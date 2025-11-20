from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId
import os

from db import datasets_collection, conversations_collection

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import MongoDBChatMessageHistory

router = APIRouter()

# ----- Models -----
class AnalystChatIn(BaseModel):
    project_id: str
    message: str
    dataset_id: Optional[str] = None

class AnalystChatOut(BaseModel):
    reply: str
    dataset_used: Optional[Dict[str, Any]] = None

# ----- Helpers -----
def _oid(s: str) -> ObjectId:
    try:
        return ObjectId(s)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")

def _load_dataset_context(project_id: str, dataset_id: Optional[str]) -> Dict[str, Any]:
    if dataset_id:
        ds = datasets_collection.find_one({"_id": _oid(dataset_id), "project_id": project_id})
    else:
        ds = datasets_collection.find({"project_id": project_id}).sort("createdAt", -1).limit(1).next()
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
        table += "| " + " | ".join(headers[:15]) + " |\n"
        table += "| " + " | ".join(["---"] * min(len(headers), 15)) + " |\n"
        for r in rows[:5]:
            vals = [str(r.get(h, "")) for h in headers[:15]]
            table += "| " + " | ".join(vals) + " |\n"

    return {
        "id": str(ds["_id"]),
        "name": ds.get("name"),
        "gcs_uri": ds.get("gcs_uri"),
        "schema_text": schema_preview,
        "table_text": table,
        "createdAt": ds.get("createdAt"),
    }

# ----- LangChain bits -----
SYSTEM_PROMPT = """You are a senior data analyst.
You will be given:
- A compact dataset context (schema + a few preview rows)
- A conversation history with the user

Your mission: provide as many meaningful insights as possible from the context and user’s requests.
Be precise, skeptical, and transparent about uncertainty. Avoid fabricating columns that don't exist.
Prefer concise bullets. If you need clarification, ask ONE clear follow-up question at the end.

The response MUST be in markdown format.

Dataset context (may be empty):
- Name: {ds_name}
- GCS: {ds_gcs}
- Schema: {ds_schema}

Preview:
{ds_table}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

# We’ll use Mongo to hold the message history (one thread per project).
def get_history(session_id: str):
    # Uses default 'messages' field in docs. You can set a custom collection too.
    return MongoDBChatMessageHistory(
        connection_string=os.environ.get("MONGO_URI", "mongodb://localhost:27017"),
        database_name=os.environ.get("MONGO_DB", "analytics_ai"),
        collection_name="lc_chat_histories",   # separate from your conversations_collection
        session_id=session_id,
    )

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm_kwargs = {"api_key": OPENAI_API_KEY}

# LLM (choose your preferred model)
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5-mini"), **llm_kwargs)

chain = prompt | llm
agent_with_history = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ----- Endpoint -----
@router.post("/analyst/chat", response_model=AnalystChatOut)
def analyst_chat(body: AnalystChatIn):
    # get dataset context
    ds_ctx = {}
    try:
        ds_ctx = _load_dataset_context(body.project_id, body.dataset_id)
    except StopIteration:
        ds_ctx = {}

    # Fill prompt vars
    ds_name = ds_ctx.get("name", "N/A")
    ds_gcs  = ds_ctx.get("gcs_uri", "N/A")
    ds_schema = ds_ctx.get("schema_text", "N/A")
    ds_table  = ds_ctx.get("table_text", "(no preview)")

    # run chain with project_id as the session id
    response = agent_with_history.invoke(
        {
            "input": body.message,
            "ds_name": ds_name,
            "ds_gcs": ds_gcs,
            "ds_schema": ds_schema,
            "ds_table": ds_table,
        },
        config={"configurable": {"session_id": f"project:{body.project_id}"}}
    )

    reply_text = getattr(response, "content", str(response))

    # log both user and assistant messages in conversations_collection
    conversations_collection.insert_many([
        {
            "project_id": body.project_id,
            "role": "user",
            "message": {"text": body.message},
            "timestamp": datetime.utcnow(),
            "source": "analyst_agent"
        },
        {
            "project_id": body.project_id,
            "role": "assistant",
            "message": {"text": reply_text},
            "timestamp": datetime.utcnow(),
            "source": "analyst_agent"
        }
    ])

    return AnalystChatOut(
        reply=reply_text,
        dataset_used=ds_ctx if ds_ctx else None
    )