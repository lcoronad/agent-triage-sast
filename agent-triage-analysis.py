import logging
import os
import json
from typing import Any, Sequence

from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_redis import RedisChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_MAX_CHARS = int(os.getenv("LOG_MAX_CHARS", "2000"))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Triage Analysis Agent Platform")

MEMORY_WINDOW_K = int(os.getenv("MEMORY_WINDOW_K", "10"))
GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "50"))


def _api_base(url: str) -> str:
    normalized = url.rstrip("/")
    return normalized if normalized.endswith("/v1") else f"{normalized}/v1"


def _mcp_sse_url(url: str) -> str:
    normalized = url.rstrip("/")
    return normalized if normalized.endswith("/sse") else f"{normalized}/sse"


def _truncate_text(text: str, max_len: int = LOG_MAX_CHARS) -> str:
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}... [truncado, total={len(text)} chars]"


def _message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", json.dumps(block, ensure_ascii=False)))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _log_message(message: BaseMessage, context: str) -> None:
    content = _message_content(message.content)

    if isinstance(message, HumanMessage):
        logger.info("[%s] INPUT usuario: %s", context, _truncate_text(content))
        return

    if isinstance(message, AIMessage):
        for tool_call in message.tool_calls or []:
            logger.info(
                "[%s] LLM -> invoca tool: %s | args=%s",
                context,
                tool_call.get("name"),
                _truncate_text(
                    json.dumps(tool_call.get("args", {}), ensure_ascii=False),
                    800,
                ),
            )
        if content:
            logger.info("[%s] LLM respuesta: %s", context, _truncate_text(content))
        return

    if isinstance(message, ToolMessage):
        logger.info(
            "[%s] MCP/tool resultado [%s]: %s",
            context,
            message.name or "unknown",
            _truncate_text(content),
        )
        return

    logger.info("[%s] %s: %s", context, message.__class__.__name__, _truncate_text(content))


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
triage_user_prompt_template = os.getenv(
    "TRIAGE_USER_PROMPT",
    (
        "Analiza estos reportes: TRIVY: {trivy_json} y OPENGREP: {opengrep_sarif}.\n"
        "1. Genera soluciones específicas de remediación lo más detalladas posible "
        "y usa la herramienta de query_company_coding_standards para dar mayor explicacion.\n"
        "2. Comenta el detalle de las remediaciones en el repositorio '{repo_path}' "
        "PR '{pull_request_number}' usando la herramienta de github_comment.\n"
        "3. Envía una notificación de fin de análisis a Slack indicando que el análisis "
        "ha finalizado usando la herramienta de slack_message."
    ),
)
slack_user_prompt_template = os.getenv(
    "SLACK_USER_PROMPT",
    (
        "El desarrollador preguntó en Slack: {slack_text}. "
        "Responde usando las herramientas de Slack en el canal '{channel}' "
        "e hilo '{thread_ts}' e investiga las normativas internas si es necesario."
    ),
)
TRIVY_JSON_MAX_CHARS = int(os.getenv("TRIVY_JSON_MAX_CHARS", "15000"))
OPENGREP_JSON_MAX_CHARS = int(os.getenv("OPENGREP_JSON_MAX_CHARS", "20000"))

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
                "transport": "sse",
                "url": _mcp_sse_url(slack_tool_url),
            },
            "github": {
                "transport": "sse",
                "url": _mcp_sse_url(github_tool_url),
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
    tools = await client.get_tools() + [query_company_coding_standards]
    logger.info(
        "Tools cargadas (%s): %s",
        len(tools),
        ", ".join(tool.name for tool in tools),
    )
    return tools


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
    *,
    context: str = "agent",
) -> dict:
    messages = list(prior_messages or []) + [HumanMessage(content=prompt)]
    logger.info("[%s] Iniciando ejecución del agente", context)

    result: dict | None = None
    seen_messages = 0

    async for state in agent.astream(
        {"messages": messages},
        config={"recursion_limit": GRAPH_RECURSION_LIMIT},
        stream_mode="values",
    ):
        current_messages = state.get("messages", [])
        for message in current_messages[seen_messages:]:
            _log_message(message, context)
        seen_messages = len(current_messages)
        result = state

    if result is None:
        result = await agent.ainvoke(
            {"messages": messages},
            config={"recursion_limit": GRAPH_RECURSION_LIMIT},
        )
        for message in result.get("messages", [])[seen_messages:]:
            _log_message(message, context)

    final_ai = _final_ai_message(result.get("messages", []))
    if final_ai is not None:
        logger.info(
            "[%s] Respuesta final del agente: %s",
            context,
            _truncate_text(_message_content(final_ai.content)),
        )

    logger.info(
        "[%s] Ejecución finalizada. Total mensajes=%s",
        context,
        len(result.get("messages", [])),
    )
    return result


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


def _build_triage_prompt(data: TriageRequest) -> str:
    return triage_user_prompt_template.format(
        trivy_json=json.dumps(data.trivy_json, ensure_ascii=False)[:TRIVY_JSON_MAX_CHARS],
        opengrep_sarif=json.dumps(data.opengrep_sarif, ensure_ascii=False)[
            :OPENGREP_JSON_MAX_CHARS
        ],
        repo_path=data.repo_path,
        pull_request_number=data.pull_request_number,
    )


def _build_slack_prompt(event: dict, channel: str, thread_ts: str) -> str:
    return slack_user_prompt_template.format(
        slack_text=event.get("text", ""),
        channel=channel,
        thread_ts=thread_ts,
    )


async def process_async_triage(data: TriageRequest):
    try:
        agent = await _build_agent(triage_system_prompt)
        prompt = _build_triage_prompt(data)
        await _run_agent(
            agent,
            prompt,
            context=f"triage:{data.repo_path}#PR{data.pull_request_number}",
        )
    except Exception:
        logger.exception(
            "Triage falló para repo=%s pr=%s",
            data.repo_path,
            data.pull_request_number,
        )


async def handle_slack_mention(event: dict):
    try:
        channel = event.get("channel")
        thread_ts = event.get("thread_ts", event.get("ts"))
        session_id = f"slack:{thread_ts}"

        history = _redis_history(session_id)
        prior_messages = _window_messages(history.messages)

        agent = await _build_agent(slack_system_prompt)
        prompt = _build_slack_prompt(event, channel, thread_ts)
        result = await _run_agent(
            agent,
            prompt,
            prior_messages=prior_messages,
            context=f"slack:{session_id}",
        )
        _persist_slack_turn(session_id, prompt, result["messages"])
    except Exception:
        logger.exception("Slack mention falló para evento=%s", event.get("ts"))


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
