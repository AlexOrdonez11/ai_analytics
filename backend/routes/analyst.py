from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from datetime import datetime
from bson import ObjectId

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .eda_tools import EDA_TOOLS
from db import client, datasets_collection

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


def _get_conversations_collection():
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
# LLM + Prompt
# -----------------------------
OPENAI_MODEL = os.getenv("ANALYST_MODEL", "gpt-5-mini")
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

# We still use ChatPromptTemplate so we can format dataset context cleanly if needed.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        # This is here for compatibility if you later want full agent-style scratchpad,
        # but our manual loop does not rely on it.
        MessagesPlaceholder("agent_scratchpad"),
    ]
)


# -----------------------------
# Manual tool-calling loop
# -----------------------------
def _run_analyst_with_tools(
    user_question: str,
    project_id: str,
    chat_history: List[Any],
    ds_name: str,
    ds_gcs: str,
    ds_schema: str,
    ds_table: str,
) -> (str, List[Dict[str, Any]]):
    """
    Minimal, non-deprecated tool-calling loop using:
      - ChatOpenAI.bind_tools(...)
      - AIMessage.tool_calls
      - ToolMessage
    """

    # 1) System + context
    system_message = SystemMessage(
        content=SYSTEM_PROMPT.format(
            ds_name=ds_name,
            ds_gcs=ds_gcs,
            ds_schema=ds_schema,
            ds_table=ds_table,
        )
    )

    ds_info = f"project_id='{project_id}'"
    human_text = (
        f"User question: {user_question}\n\n"
        f"When you call any EDA tool, always pass {ds_info} in the tool arguments."
    )
    user_message = HumanMessage(content=human_text)

    # Messages: system + prior conversation + current question
    messages: List[Any] = [system_message] + list(chat_history) + [user_message]

    # 2) Bind tools to the model
    llm_with_tools = llm.bind_tools(EDA_TOOLS)

    # 3) First call: let the model decide whether to call tools
    first_ai = llm_with_tools.invoke(messages)
    if not isinstance(first_ai, AIMessage):
        # Fallback
        return str(getattr(first_ai, "content", first_ai)), []

    plots: List[Dict[str, Any]] = []

    # 4) If no tool_calls → we're done
    if not getattr(first_ai, "tool_calls", None):
        return first_ai.content, plots

    # 5) Execute tool calls
    tool_by_name = {t.name: t for t in EDA_TOOLS}
    tool_messages: List[ToolMessage] = []

    for tool_call in first_ai.tool_calls:
        name = tool_call.get("name")
        args = tool_call.get("args", {}) or {}

        tool = tool_by_name.get(name)
        if tool is None:
            # If the model requested an unknown tool, just skip it
            continue

        # Run the tool. For LangChain tools, .invoke(args) is the standard way. :contentReference[oaicite:1]{index=1}
        try:
            result = tool.invoke(args)
        except Exception as e:
            result = {"error": str(e)}

        # Collect plots if they follow your {"url": "..."} contract
        if isinstance(result, dict) and "url" in result:
            plots.append(result)

        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call.get("id", ""),
            )
        )

    # 6) Second call: give the model tools' results to synthesize final answer
    messages_with_tools = messages + [first_ai] + tool_messages
    final_ai = llm.invoke(messages_with_tools)
    if isinstance(final_ai, AIMessage):
        reply_text = final_ai.content
    else:
        reply_text = str(getattr(final_ai, "content", final_ai))

    return reply_text, plots


# -----------------------------
# Route
# -----------------------------
@router.post("/analyst/chat", response_model=AnalystChatOut)
def analyst_chat(body: AnalystChatIn):

    chat_history = _load_chat_history(body.project_id)

    ds_ctx = _load_dataset_context(body.project_id)

    ds_name = ds_ctx.get("name", "N/A")
    ds_gcs = ds_ctx.get("gcs_uri", "N/A")
    ds_schema = ds_ctx.get("schema_text", "N/A")
    ds_table = ds_ctx.get("table_text", "(no preview)")

    reply_text, plots = _run_analyst_with_tools(
        user_question=body.message,
        project_id=body.project_id,
        chat_history=chat_history,
        ds_name=ds_name,
        ds_gcs=ds_gcs,
        ds_schema=ds_schema,
        ds_table=ds_table,
    )

    _log_turn(body.project_id, body.message, reply_text)
    return AnalystChatOut(reply=reply_text, plots=plots or None)
