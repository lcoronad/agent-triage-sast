import logging
import os
import json
import requests
from typing import Any, Sequence

from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_redis import RedisChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pymilvus import MilvusClient
from typing import List
from langchain_core.embeddings import Embeddings

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
PARALLEL_TOOL_CALLS = os.getenv("PARALLEL_TOOL_CALLS", "false").lower() in {
    "1",
    "true",
    "yes",
}
SINGLE_TOOL_CALL_RULE = (
    "\n\nIMPORTANTE: Invoca como máximo UNA herramienta por turno. "
    "Espera el resultado antes de llamar la siguiente herramienta."
)


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


class SingleToolCallMiddleware(AgentMiddleware):
    """Fuerza una sola tool-call por turno (requerido por vLLM/Llama)."""

    def _apply_single_tool_call_setting(self, request) -> None:
        if PARALLEL_TOOL_CALLS:
            return

        model_settings = getattr(request, "model_settings", None)
        if isinstance(model_settings, dict):
            model_settings["parallel_tool_calls"] = False
            return

        tools = getattr(request, "tools", None)
        if tools:
            request.model = request.model.bind_tools(tools, parallel_tool_calls=False)

    def _truncate_extra_tool_calls(self, state) -> dict | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        last = messages[-1]
        if (
            not isinstance(last, AIMessage)
            or not last.tool_calls
            or len(last.tool_calls) <= 1
        ):
            return None

        logger.warning(
            "Modelo devolvió %s tool_calls; truncando a 1 para compatibilidad vLLM",
            len(last.tool_calls),
        )
        last.tool_calls = [last.tool_calls[0]]
        return None

    def wrap_model_call(self, request, handler):
        self._apply_single_tool_call_setting(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        self._apply_single_tool_call_setting(request)
        return await handler(request)

    def after_model(self, state, runtime):
        return self._truncate_extra_tool_calls(state)

    async def aafter_model(self, state, runtime):
        return self._truncate_extra_tool_calls(state)


def _system_prompt_with_rules(system_prompt: str) -> str:
    if PARALLEL_TOOL_CALLS:
        return system_prompt
    if SINGLE_TOOL_CALL_RULE.strip() in system_prompt:
        return system_prompt
    return f"{system_prompt.rstrip()}{SINGLE_TOOL_CALL_RULE}"


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
milvus_search_limit = int(os.getenv("MILVUS_SEARCH_LIMIT", "3"))
milvus_metric_type = os.getenv("MILVUS_METRIC_TYPE", "L2")

_DEFAULT_GITHUB_REMEDIATION_GUIDE = """
## Plantillas obligatorias para comentarios en GitHub (Markdown)

Usa español técnico, tono profesional y contenido accionable. NO publiques listas sueltas;
cada hallazgo debe tener secciones completas.

### A) TRIVY → publicar_comentario_general_pr (UN comentario consolidado)

Estructura mínima del campo `vulnerabilidades_md`:

## Resumen ejecutivo SCA (Trivy)
- Total hallazgos: N
- Críticos: X | Altos: Y | Medios: Z | Bajos: W
- Acción prioritaria: (1-2 frases)

## Tabla de dependencias afectadas
| Severidad | ID | Paquete | Versión instalada | Versión corregida | Archivo/Target |
|-----------|----|---------|-------------------|-------------------|----------------|
| CRITICAL  | CVE-... | ... | ... | ... | pom.xml |

## Detalle por vulnerabilidad

### [SEVERIDAD] CVE-XXXX — nombre-paquete
**Target:** archivo o imagen afectada

**Descripción:** Qué es la vulnerabilidad y por qué aplica aquí.

**Impacto:** Riesgo concreto para esta aplicación (confidencialidad, integridad, disponibilidad).

**Remediación:**
1. Paso concreto (actualizar versión, reemplazar dependencia, etc.)
2. Verificar compatibilidad / breaking changes si aplica
3. Re-ejecutar escaneo Trivy tras el cambio

**Comando o cambio sugerido:**
```xml
<!-- fragmento pom.xml / package.json / etc. con versión corregida -->
```

**Referencias:**
- Enlace NVD/advisory si está en el reporte
- Norma interna (si query_company_coding_standards aportó contexto)

**Validación:**
- [ ] Dependencia actualizada en el manifest lockfile
- [ ] Build/test exitoso
- [ ] Trivy sin el CVE en el target afectado

---

(repetir bloque ### por CADA CVE de Trivy)

### B) OPENGREP → publicar_comentario_linea_pr (UN comentario POR hallazgo)

Estructura mínima del campo `recomendacion` (Markdown):

### [SEVERIDAD] rule_id — CWE-XXX

**Ubicación:** `ruta/archivo.ext:LINEA`

**Problema detectado:** Explica la línea/patrón inseguro en lenguaje claro.

**Riesgo / impacto:** Qué puede explotar un atacante (ej. RCE, SQLi, filtrado de datos).

**Causa raíz:** Por qué el código actual es vulnerable.

**Remediación recomendada:**
1. Cambio específico a aplicar
2. Buenas prácticas (validación, parametrización, allowlist, etc.)
3. Controles adicionales si aplica (tests, lint, SAST en CI)

**Código sugerido:**
```java
// ❌ Código vulnerable (resumen)
// ✅ Código corregido (fragmento concreto)
```

**Referencias:**
- CWE / rule_id
- Norma interna relevante (Milvus) si la consultaste

**Validación:**
- [ ] Corrección aplicada en la línea o bloque afectado
- [ ] Prueba unitaria o caso negativo añadido si aplica
- [ ] OpenGrep/SAST sin re-detectar el hallazgo

### Reglas de calidad
- Integra SIEMPRE datos del reporte (CVE, paquete, versión, rule_id, path, line, mensaje).
- Enriquece con query_company_coding_standards antes de redactar (una consulta por turno).
- Prioriza remediaciones prácticas para el lenguaje/framework del repo.
- Si falta fixedVersion en Trivy, indica mitigación alternativa (workaround, compensating control).
- No inventes URLs; usa solo referencias presentes en el reporte o normas internas.
""".strip()

github_remediation_guide = os.getenv(
    "GITHUB_REMEDIATION_GUIDE",
    _DEFAULT_GITHUB_REMEDIATION_GUIDE,
)

triage_system_prompt = os.getenv(
    "TRIAGE_SYSTEM_PROMPT",
    (
        "Eres un agente DevSecOps experto en triage y remediación de vulnerabilidades.\n\n"
        "Debes analizar SIEMPRE ambas fuentes:\n"
        "1) TRIVY (SCA/dependencias) — sin línea de código.\n"
        "2) OPENGREP (SAST/código) — con archivo y línea.\n\n"
        "Publicación en GitHub:\n"
        "- TRIVY → publicar_comentario_general_pr: UN comentario consolidado, "
        "estructurado con tabla + detalle por CVE (ver GITHUB_REMEDIATION_GUIDE).\n"
        "- OPENGREP → publicar_comentario_linea_pr: UN comentario POR hallazgo, "
        "con remediación detallada y código sugerido (ver GITHUB_REMEDIATION_GUIDE).\n\n"
        "Calidad de remediación:\n"
        "- Consulta query_company_coding_standards por CVE, CWE o rule_id antes de redactar.\n"
        "- Incluye impacto, pasos concretos, fragmento de código/config y checklist de validación.\n"
        "- No uses bullets genéricos; cada hallazgo lleva secciones completas.\n"
        "- NO finalices hasta publicar comentarios para TODOS los ítems del checklist.\n"
        "- Invoca como máximo UNA herramienta por turno."
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
        "Analiza TODOS los hallazgos de TRIVY y OPENGREP del PR. No omitas ninguno.\n\n"
        "## Contexto\n"
        "- Repositorio: {repo_path}\n"
        "- Owner: {repo_owner}\n"
        "- Repo: {repo_name}\n"
        "- PR: {pull_request_number}\n"
        "- Commit: {commit_id}\n\n"
        "## Checklist obligatorio\n"
        "{findings_checklist}\n\n"
        "## Flujo (una herramienta por turno)\n"
        "1. Consulta query_company_coding_standards (CVE, CWE o rule_id) para enriquecer contexto.\n"
        "2. Redacta comentario SCA estructurado y publícalo con publicar_comentario_general_pr "
        "(owner={repo_owner}, repo={repo_name}, pr_number={pull_request_number}).\n"
        "   El campo vulnerabilidades_md DEBE seguir la plantilla TRIVY de GITHUB_REMEDIATION_GUIDE.\n"
        "3. Por CADA hallazgo OPENGREP: redacta recomendacion con plantilla SAST y publica con "
        "publicar_comentario_linea_pr (owner, repo, pr_number, commit_id={commit_id}, "
        "path, line, recomendacion).\n"
        "4. Notifica fin de análisis en Slack (resumen: total Trivy, total OpenGrep, acciones).\n\n"
        "## Guía de formato para GitHub\n"
        "{github_remediation_guide}\n\n"
        "## Reportes completos\n"
        "TRIVY: {trivy_json}\n"
        "OPENGREP: {opengrep_sarif}"
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
token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

llm = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model=model_id,
    temperature=temperature,
    top_p=top_p,
    max_completion_tokens=max_completion_tokens,
)

class LlamaStackGraniteEmbeddings(Embeddings):
    """
    Adaptador personalizado para enviar textos puros a Llama Stack,
    evitando los bugs de arrays del SDK de OpenAI.
    """

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectores = []
        # Enviamos los textos uno por uno como strings puros
        for text in texts:
            # Aseguramos que sea un string válido
            texto_limpio = str(text).strip()
            payload = {
                "model": self.model,
                "input": texto_limpio
            }
            response = requests.post(
                f"{self.base_url}/embeddings",  # Endpoint /v1/embeddings
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            # Si hay un error HTTP, nos lo mostrará claramente
            response.raise_for_status() 
            # Extraemos el vector de la respuesta de Llama Stack
            vector = response.json()["data"][0]["embedding"]
            vectores.append(vector)
        return vectores

    def embed_query(self, text: str) -> List[float]:
        # Para LangChain, un query es solo un documento de 1 elemento
        return self.embed_documents([text])[0]


embeddings = LlamaStackGraniteEmbeddings(
    model=embedding_model, 
    base_url=embedding_base_url
)


def _milvus_uri() -> str:
    explicit = os.getenv("MILVUS_URI", "").strip()
    if explicit:
        return explicit

    host = milvus_host.strip()
    port = milvus_port.strip()
    if host.startswith(("http://", "https://")):
        return host if host.endswith(f":{port}") else f"{host.rstrip('/')}:{port}"
    return f"http://{host}:{port}"


def _search_company_standards(query: str) -> list[str]:
    """Consulta Milvus creando conexión local (seguro en threads de LangGraph)."""
    milvus_client = MilvusClient(uri=_milvus_uri())
    query_vector = embeddings.embed_query(query)
    results = milvus_client.search(
        collection_name=milvus_collection_name,
        data=[query_vector],
        limit=milvus_search_limit,
        search_params={"metric_type": milvus_metric_type, "params": {}},
        output_fields=["text"],
    )

    if not results or not results[0]:
        return []

    chunks: list[str] = []
    for hit in results[0]:
        text = hit.get("entity", {}).get("text", "")
        if text:
            chunks.append(text)
    return chunks


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
    try:
        docs = _search_company_standards(query)
        if not docs:
            return "No se encontraron normas internas relacionadas con la consulta."
        return "\n\n".join(f"[Norma]: {doc}" for doc in docs)
    except Exception as exc:
        logger.exception("Error consultando Milvus para query=%s", query)
        return f"Error al consultar normas en Milvus: {exc}"


@tool
def publicar_comentario_linea_pr(owner: str, repo: str, pr_number: int, commit_id: str, path: str, line: int, recomendacion: str) -> str:
    """
    Publica comentario SAST inline (OPENGREP) en el PR.

    El campo `recomendacion` debe ser Markdown estructurado con:
    Problema detectado, Riesgo/impacto, Causa raíz, Remediación (pasos),
    Código sugerido (bloque ```), Referencias (CWE/rule_id/normas) y Validación (checklist).
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/comments"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    payload = {
        "body": f"🚨 **Alerta de Seguridad (SAST)**\n\n{recomendacion}",
        "commit_id": commit_id,
        "path": path,
        "line": line,
        "side": "RIGHT"
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 201:
        return "✅ Comentario publicado exitosamente en la línea de código."
    elif response.status_code == 422 and "could not be resolved" in response.text:
        print(f"⚠️ La línea {line} no está en el diff del PR. Usando Fallback a comentario general...")
        
        url_general = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
        payload_general = {
            "body": f"🚨 **Alerta de Seguridad (SAST) - Código Preexistente**\n\n"
                    f"📍 *Se detectó una vulnerabilidad en el archivo `{path}` en la línea `{line}`, pero esta línea no fue modificada en este PR.*\n\n"
                    f"{recomendacion}"
        }
        
        fallback_resp = requests.post(url_general, headers=headers, json=payload_general)
        if fallback_resp.status_code == 201:
            return "✅ El comentario no pudo anclarse a la línea (fuera del diff), pero se publicó exitosamente como comentario general en el PR."
        else:
            return f"❌ Error crítico en el Fallback: {fallback_resp.status_code} - {fallback_resp.text}"
    else:
        return f"❌ Error {response.status_code}: {response.text}"


@tool
def publicar_comentario_general_pr(owner: str, repo: str, pr_number: int, vulnerabilidades_md: str) -> str:
    """
    Publica comentario general SCA (TRIVY) en el PR.

    El campo `vulnerabilidades_md` debe incluir: resumen ejecutivo, tabla de CVEs,
    y bloque detallado por CVE (descripción, impacto, remediación, cambio sugerido,
    referencias y checklist de validación). Ver GITHUB_REMEDIATION_GUIDE.
    """
    
    # Nota: Para GitHub, los comentarios generales de un PR se hacen en el endpoint de "issues"
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    payload = {
        "body": f"📦 **Alerta de Seguridad en Dependencias (SCA)**\n\n{vulnerabilidades_md}"
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 201:
        return "✅ Comentario general de dependencias publicado exitosamente."
    else:
        return f"❌ Error {response.status_code}: {response.text}"


async def _get_agent_tools():
    client = _mcp_client()
    tools = await client.get_tools() + [query_company_coding_standards] + [publicar_comentario_linea_pr] + [publicar_comentario_general_pr]
    logger.info(
        "Tools cargadas (%s): %s",
        len(tools),
        ", ".join(tool.name for tool in tools),
    )
    return tools


async def _build_agent(system_prompt: str):
    tools = await _get_agent_tools()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=_system_prompt_with_rules(system_prompt),
        middleware=[SingleToolCallMiddleware()],
    )


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
    repo_owner: str
    commit_id: str
    trivy_json: dict
    opengrep_sarif: dict


def _split_repo_path(repo_path: str, repo_owner: str) -> tuple[str, str]:
    if "/" in repo_path:
        owner, repo = repo_path.split("/", 1)
        return owner, repo
    return repo_owner, repo_path


def _normalize_repo_file_path(path: str, repo_name: str) -> str:
    normalized = path.lstrip("./")
    prefix = f"{repo_name}/"
    if normalized.startswith(prefix):
        return normalized[len(prefix) :]
    return normalized


def _extract_trivy_findings(trivy_json: dict) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for result in trivy_json.get("Results", []):
        target = result.get("Target", "")
        for vuln in result.get("Vulnerabilities", []) or []:
            findings.append(
                {
                    "id": vuln.get("VulnerabilityID", "UNKNOWN"),
                    "pkg": vuln.get("PkgName", ""),
                    "severity": vuln.get("Severity", ""),
                    "target": target,
                    "fixed_version": vuln.get("FixedVersion", ""),
                    "description": vuln.get("Description", vuln.get("Title", "")),
                }
            )
    return findings


def _opengrep_result_line(result: dict[str, Any]) -> int | None:
    start = result.get("start") or {}
    if isinstance(start, dict) and start.get("line") is not None:
        return int(start["line"])

    locations = result.get("locations") or []
    if locations:
        region = locations[0].get("physicalLocation", {}).get("region", {})
        if region.get("startLine") is not None:
            return int(region["startLine"])
    return None


def _opengrep_result_path(result: dict[str, Any]) -> str:
    if result.get("path"):
        return str(result["path"])

    locations = result.get("locations") or []
    if locations:
        uri = locations[0].get("physicalLocation", {}).get("artifactLocation", {}).get("uri", "")
        if uri:
            return str(uri)
    return ""


def _opengrep_result_message(result: dict[str, Any]) -> str:
    extra = result.get("extra") or {}
    if extra.get("message"):
        return str(extra["message"])
    message = result.get("message") or {}
    if isinstance(message, dict) and message.get("text"):
        return str(message["text"])
    return str(message or "")


def _extract_opengrep_findings(opengrep_sarif: dict) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for run in opengrep_sarif.get("runs", []):
        for result in run.get("results", []) or []:
            rule_id = result.get("check_id") or result.get("ruleId") or "UNKNOWN"
            extra = result.get("extra") or {}
            metadata = extra.get("metadata") or {}
            cwe_list = metadata.get("cwe") or []
            findings.append(
                {
                    "rule_id": rule_id,
                    "path": _opengrep_result_path(result),
                    "line": _opengrep_result_line(result),
                    "severity": extra.get("severity") or result.get("level", ""),
                    "message": _opengrep_result_message(result),
                    "cwe": cwe_list[0] if cwe_list else "",
                }
            )
    return findings


def _build_findings_checklist(data: TriageRequest) -> str:
    owner, repo_name = _split_repo_path(data.repo_path, data.repo_owner)
    trivy_findings = _extract_trivy_findings(data.trivy_json)
    opengrep_findings = _extract_opengrep_findings(data.opengrep_sarif)

    lines = [
        f"Owner={owner}, Repo={repo_name}, PR={data.pull_request_number}, Commit={data.commit_id}",
        "",
        f"### TRIVY ({len(trivy_findings)} hallazgos) → publicar_comentario_general_pr",
    ]

    if trivy_findings:
        for idx, finding in enumerate(trivy_findings, start=1):
            lines.append(
                f"{idx}. [{finding['severity']}] {finding['id']} en {finding['pkg']} "
                f"(target: {finding['target']}) → remediar a {finding['fixed_version'] or 'ver advisory'}"
            )
    else:
        lines.append("- Sin hallazgos Trivy.")

    lines.append("")
    lines.append(
        f"### OPENGREP ({len(opengrep_findings)} hallazgos) → publicar_comentario_linea_pr (uno por hallazgo)"
    )

    if opengrep_findings:
        for idx, finding in enumerate(opengrep_findings, start=1):
            path = _normalize_repo_file_path(finding["path"], repo_name)
            line = finding["line"] if finding["line"] is not None else "?"
            lines.append(
                f"{idx}. [{finding['severity']}] {finding['rule_id']} "
                f"en {path}:{line} — {finding['message']}"
            )
            lines.append(
                f"   → publicar_comentario_linea_pr(owner={owner}, repo={repo_name}, "
                f"pr_number={data.pull_request_number}, commit_id={data.commit_id}, "
                f"path={path}, line={line}, recomendacion=...)"
            )
    else:
        lines.append("- Sin hallazgos OpenGrep.")

    lines.append("")
    lines.append("### Formato esperado en GitHub")
    lines.append("- TRIVY: resumen + tabla + detalle por CVE con remediación y validación.")
    lines.append(
        "- OPENGREP: un comentario por hallazgo con impacto, causa, fix, código sugerido y validación."
    )
    lines.append("")
    lines.append(
        "IMPORTANTE: Debes publicar comentarios para TODOS los hallazgos anteriores antes de Slack."
    )
    return "\n".join(lines)


def _build_triage_prompt(data: TriageRequest) -> str:
    owner, repo_name = _split_repo_path(data.repo_path, data.repo_owner)
    guide = github_remediation_guide
    if "{github_remediation_guide}" not in triage_user_prompt_template:
        guide = ""

    return triage_user_prompt_template.format(
        trivy_json=json.dumps(data.trivy_json, ensure_ascii=False)[:TRIVY_JSON_MAX_CHARS],
        opengrep_sarif=json.dumps(data.opengrep_sarif, ensure_ascii=False)[
            :OPENGREP_JSON_MAX_CHARS
        ],
        repo_path=data.repo_path,
        pull_request_number=data.pull_request_number,
        repo_owner=owner,
        repo_name=repo_name,
        commit_id=data.commit_id,
        findings_checklist=_build_findings_checklist(data),
        github_remediation_guide=guide or github_remediation_guide,
    )


def _build_slack_prompt(event: dict, channel: str, thread_ts: str) -> str:
    return slack_user_prompt_template.format(
        slack_text=event.get("text", ""),
        channel=channel,
        thread_ts=thread_ts,
    )


async def process_async_triage(data: TriageRequest):
    try:
        trivy_count = len(_extract_trivy_findings(data.trivy_json))
        opengrep_count = len(_extract_opengrep_findings(data.opengrep_sarif))
        logger.info(
            "Iniciando triage repo=%s pr=%s | hallazgos trivy=%s opengrep=%s",
            data.repo_path,
            data.pull_request_number,
            trivy_count,
            opengrep_count,
        )
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
