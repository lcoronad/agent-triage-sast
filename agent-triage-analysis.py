import os
import json
from typing import Sequence

from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_redis import RedisChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

app = FastAPI(title="Triage Analysis Agent Platform")

MEMORY_WINDOW_K = int(os.getenv("MEMORY_WINDOW_K", "10"))
GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "50"))


def _api_base(url: str) -> str:
    normalized = url.rstrip("/")
    return normalized if normalized.endswith("/v1") else f"{normalized}/v1"


base_url = _api_base(os.getenv("QWEN_API_URL", "http://vllm-qwen-service:8000/v1"))
model_id = os.getenv("QWEN_MODEL_NAME", "qwen2.5-coder:32b-instruct")
api_key = os.getenv("QWEN_API_KEY", "")
temperature = float(os.getenv("TEMPERATURE", "0.1"))
top_p = float(os.getenv("TOP_P", "0.1"))
max_completion_tokens = int(os.getenv("MAX_COMPLETION_TOKENS", "4096"))
embedding_model = os.getenv(
    "EMBEDDINGS_MODEL_NAME",
    "sentence-transformers/ibm-granite/granite-embedding-125m-english",
)
embedding_base_url = _api_base(
    os.getenv("EMBEDDING_API_URL", "http://vllm-qwen-service:8000/v1")
)
embedding_api_key = os.getenv("EMBEDDING_API_KEY", "")
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
slack_tool_url = os.getenv("SLACK_TOOL_URL", "http://cluster.local")
github_tool_url = os.getenv("GITHUB_TOOL_URL", "http://cluster.local")
milvus_host = os.getenv("MILVUS_HOST", "milvus-service")
milvus_port = os.getenv("MILVUS_PORT", "19530")
milvus_collection_name = os.getenv("MILVUS_COLLECTION_NAME", "company_coding_standards")
milvus_uri = "http://" + milvus_host + ":" + milvus_port

triage_system_prompt = os.getenv(
    "TRIAGE_SYSTEM_PROMPT",
    (
        "Eres un agente DevSecOps experto en análisis de vulnerabilidades y remediación de código. "
        "Usa las herramientas MCP de GitHub y Slack, y consulta las normas internas de codificación "
        "cuando necesites validar lineamientos de la empresa."
    ),
)
slack_system_prompt = os.getenv(
    "SLACK_SYSTEM_PROMPT",
    (
        "Eres un asistente DevSecOps en Slack. Responde de forma clara y técnica. "
        "Usa las herramientas de Slack para publicar en el canal e hilo indicados. "
        "Consulta las normas internas de codificación cuando el desarrollador lo requiera."
    ),
)

llm = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model=model_id,
    temperature=temperature,
    top_p=top_p,
    max_completion_tokens=max_completion_tokens,
)

embeddings = OpenAIEmbeddings(
    model=embedding_model,
    openai_api_base=embedding_base_url,
    openai_api_key=embedding_api_key,
)
vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": milvus_uri},
    collection_name=milvus_collection_name,
)


def _mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "slack": {
                "transport": "http",
                "url": slack_tool_url,
            },
            "github": {
                "transport": "http",
                "url": github_tool_url,
            },
        }
    )


@tool
def query_company_coding_standards(query: str) -> str:
    """Útil para buscar lineamientos de codificación segura de la empresa."""
    docs = vector_store.similarity_search(query, k=3)
    return "\n\n".join([f"[Norma]: {d.page_content}" for d in docs])


async def _get_agent_tools():
    client = _mcp_client()
    return await client.get_tools() + [query_company_coding_standards]


async def _build_agent(system_prompt: str):
    tools = await _get_agent_tools()
    return create_agent(model=llm, tools=tools, system_prompt=system_prompt)


def _redis_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(
        redis_url=redis_url,
        session_id=session_id,
        ttl=604800,
    )


def _window_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    if MEMORY_WINDOW_K <= 0:
        return list(messages)
    return list(messages[-MEMORY_WINDOW_K * 2 :])


def _trim_redis_history(history: RedisChatMessageHistory) -> None:
    messages = history.messages
    windowed = _window_messages(messages)
    if len(windowed) >= len(messages):
        return
    history.clear()
    for message in windowed:
        history.add_message(message)


def _final_ai_message(messages: Sequence[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return message
    return None


async def _run_agent(
    agent,
    prompt: str,
    prior_messages: Sequence[BaseMessage] | None = None,
) -> dict:
    messages = list(prior_messages or []) + [HumanMessage(content=prompt)]
    return await agent.ainvoke(
        {"messages": messages},
        config={"recursion_limit": GRAPH_RECURSION_LIMIT},
    )


def _persist_slack_turn(
    session_id: str,
    prompt: str,
    result_messages: Sequence[BaseMessage],
) -> None:
    history = _redis_history(session_id)
    history.add_message(HumanMessage(content=prompt))

    final_ai = _final_ai_message(result_messages)
    if final_ai is not None:
        history.add_message(final_ai)

    _trim_redis_history(history)


class TriageRequest(BaseModel):
    repo_path: str
    pull_request_number: str
    trivy_json: dict
    opengrep_sarif: dict


async def process_async_triage(data: TriageRequest):
    agent = await _build_agent(triage_system_prompt)
    prompt = (
        f"Analiza estos reportes: TRIVY: {json.dumps(data.trivy_json)[:15000]} "
        f"y OPENGREP: {json.dumps(data.opengrep_sarif)[:20000]}.\n"
        f"1. Genera soluciones específicas de remediación.\n"
        f"2. Comenta en el repositorio '{data.repo_path}' PR '{data.pull_request_number}'.\n"
        f"3. Envía una notificación de fin de análisis a Slack."
    )
    await _run_agent(agent, prompt)


async def handle_slack_mention(event: dict):
    channel = event.get("channel")
    thread_ts = event.get("thread_ts", event.get("ts"))
    session_id = f"slack:{thread_ts}"

    history = _redis_history(session_id)
    prior_messages = _window_messages(history.messages)

    agent = await _build_agent(slack_system_prompt)
    prompt = (
        f"El desarrollador preguntó en Slack: {event.get('text')}. "
        f"Responde usando las herramientas de Slack en el canal '{channel}' "
        f"e hilo '{thread_ts}' e investiga las normativas internas si es necesario."
    )
    result = await _run_agent(agent, prompt, prior_messages=prior_messages)
    _persist_slack_turn(session_id, prompt, result["messages"])


@app.post("/api/v1/triage")
async def trigger_triage(payload: TriageRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_async_triage, payload)
    return {"status": "accepted"}


@app.post("/api/v1/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge")}
    if (
        payload.get("type") == "event_callback"
        and payload.get("event", {}).get("type") == "app_mention"
    ):
        background_tasks.add_task(handle_slack_mention, payload.get("event"))
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "7861"))
    uvicorn.run(app, host=host, port=port)
