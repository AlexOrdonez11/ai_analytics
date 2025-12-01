from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from datetime import datetime
from bson import ObjectId

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler

from .eda_tools import EDA_TOOLS
from db import get_mongo_client, datasets_collection

router = APIRouter()


# -----------------------------
# Pydantic models
# -----------------------------
class AnalystChatIn(BaseModel):
    project_id: str
    message: str


class AnalystChatOut(BaseModel):
    reply: str
    plots: Optional[List[Dict[str, Any]]] = None


# -----------------------------
# Helpers
# -----------------------------
def _oid(s: str) -> ObjectId:
    try:
        return ObjectId(s)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")


def _load_dataset_context(
    project_id: str,
    dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load latest dataset for the project (or a specific dataset_id if provided)
    and build a compact schema/preview context.
    """
    if dataset_id:
        ds = datasets_collection.find_one(
            {"_id": _oid(dataset_id), "project_id": project_id}
        )
    else:
        # Latest dataset for this project
        cursor = (
            datasets_collection.find({"project_id": project_id})
            .sort("createdAt", -1)
            .limit(1)
        )
        try:
            ds = next(cursor)
        except StopIteration:
            ds = None

    if not ds:
        return {}

    # Build a compact textual context (schema + few preview rows)
    schema_items = list(ds.get("schema", {}).items())
    schema_preview = ", ".join([f"{k}({v})" for k, v in schema_items[:20]])
    rows = ds.get("preview", [])[:5]

    # Simple markdown-ish table preview
    table = ""
    if rows:
        headers = list(rows[0].keys())
        headers = headers[:15]
        table += "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for r in rows[:5]:
            vals = [str(r.get(h, "")) for h in headers]
            table += "| " + " | ".join(vals) + " |\n"

    return {
        "id": str(ds["_id"]),
        "name": ds.get("name"),
        "gcs_uri": ds.get("gcs_uri"),
        "schema_text": schema_preview,
        "table_text": table,
        "createdAt": ds.get("createdAt"),
    }


# -----------------------------
# LLM + Prompt + Agent
# -----------------------------
OPENAI_MODEL = os.getenv("ANALYST_MODEL", "gpt-5.1-mini")
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)

SYSTEM_PROMPT = """
You are a senior data analyst.

You will be given:
- A compact dataset context (schema + a few preview rows)
- A conversation history with the user

Your mission: provide as many meaningful insights as possible from the context and user’s requests.
Be precise, skeptical, and transparent about uncertainty. Avoid fabricating columns that don't exist.
Prefer concise bullets. If you need clarification, ask ONE clear follow-up question at the end.

The response MUST be in markdown format.

- Dataset rows are stored as CSV files in Google Cloud Storage.
- Metadata and conversations live in MongoDB.
- EDA tools will load the CSV data by project_id and dataset_id, then save plots back to GCS and return public URLs.
- When you call any EDA tool, you MUST include the project_id exactly as provided in the conversation.
- Use EDA tools when the user asks for distributions, scatterplots, time series, correlations, or whenever they are necessary to help the user.
- Be precise and skeptical about data limitations (e.g., sampling, missing values).
- In your final answer, output concise bullet points plus short explanations of what each plot shows and any caveats.

Dataset context (may be empty):
- Name: {ds_name}
- GCS: {ds_gcs}
- Schema: {ds_schema}

Preview:
{ds_table}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        # required so the tool-calling agent can inject prior tool calls / scratchpad
        MessagesPlaceholder("agent_scratchpad"),
    ]
)


class PlotCollectorCallbackHandler(BaseCallbackHandler):
    """
    Callback to capture tool outputs that contain plot URLs.
    Assumes each EDA tool returns a dict, e.g. {"url": "...", ...}
    """

    def __init__(self):
        self.plots: List[Dict[str, Any]] = []

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        if isinstance(output, dict) and "url" in output:
            self.plots.append(output)


# This returns an LCEL Runnable agent (no AgentExecutor needed)
agent = create_tool_calling_agent(llm=llm, tools=EDA_TOOLS, prompt=prompt)


# -----------------------------
# Mongo helpers (chat history)
# -----------------------------
def _get_conversations_collection():
    client = get_mongo_client()
    db = client["analytics_ai"]
    return db["analyst_conversations"]


def _load_chat_history(project_id: str) -> List[Any]:
    coll = _get_conversations_collection()
    docs = coll.find({"project_id": project_id}).sort("timestamp", 1)
    history: List[Any] = []
    for d in docs:
        role = d.get("role")
        text = d.get("message", {}).get("text", "")
        if not text:
            continue
        if role == "user":
            history.append(HumanMessage(content=text))
        elif role == "assistant":
            history.append(AIMessage(content=text))
    return history


def _log_turn(project_id: str, user_text: str, assistant_text: str):
    coll = _get_conversations_collection()
    now = datetime.utcnow()
    coll.insert_many(
        [
            {
                "project_id": project_id,
                "role": "user",
                "message": {"text": user_text},
                "timestamp": now,
                "source": "analyst_agent",
            },
            {
                "project_id": project_id,
                "role": "assistant",
                "message": {"text": assistant_text},
                "timestamp": now,
                "source": "analyst_agent",
            },
        ]
    )


# -----------------------------
# Route
# -----------------------------
@router.post("/analyst/chat", response_model=AnalystChatOut)
def analyst_chat(body: AnalystChatIn):

    chat_history = _load_chat_history(body.project_id)

    ds_ctx = _load_dataset_context(body.project_id)

    # Fill prompt vars
    ds_name = ds_ctx.get("name", "N/A")
    ds_gcs = ds_ctx.get("gcs_uri", "N/A")
    ds_schema = ds_ctx.get("schema_text", "N/A")
    ds_table = ds_ctx.get("table_text", "(no preview)")

    ds_info = f"project_id='{body.project_id}'"
    agent_input = (
        f"User question: {body.message}\n\n"
        f"When you call any EDA tool, always pass {ds_info} in the tool arguments."
    )

    # Collect plot URLs from tools via callback
    cb = PlotCollectorCallbackHandler()

    result = agent.invoke(
        {
            "input": agent_input,
            "chat_history": chat_history,
            "ds_name": ds_name,
            "ds_gcs": ds_gcs,
            "ds_schema": ds_schema,
            "ds_table": ds_table,
        },
        config={"callbacks": [cb]},
    )

    # Handle result (AIMessage or dict)
    if isinstance(result, dict):
        reply_text = (
            result.get("output")
            or result.get("content")
            or str(result)
        )
    elif hasattr(result, "content"):
        reply_text = result.content
    else:
        reply_text = str(result)

    plots = cb.plots or None

    _log_turn(body.project_id, body.message, reply_text)
    return AnalystChatOut(reply=reply_text, plots=plots)
