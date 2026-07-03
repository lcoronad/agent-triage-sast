import logging
import os
import json
import uuid
import re
import requests
from typing import Any, Sequence
from urllib.parse import urlparse

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


_GITHUB_TOOLS = frozenset(
    {"publicar_comentario_general_pr", "publicar_comentario_linea_pr"}
)


def _split_repo_path(repo_path: str, repo_owner: str) -> tuple[str, str]:
    path = repo_path.strip().rstrip("/").removesuffix(".git")
    if path.startswith("//"):
        path = f"https:{path}"
    if not path.startswith(("http://", "https://", "git@")) and "github.com/" in path:
        path = f"https://{path.lstrip('/')}"

    if path.startswith(("http://", "https://")):
        segments = [segment for segment in urlparse(path).path.split("/") if segment]
        if len(segments) >= 2:
            return segments[-2], segments[-1]

    if path.startswith("git@"):
        _, _, rest = path.partition(":")
        segments = [segment for segment in rest.split("/") if segment]
        if len(segments) >= 2:
            return segments[-2], segments[-1]

    segments = [
        segment
        for segment in path.split("/")
        if segment and segment not in {"github.com", "https:", "http:"}
    ]
    if len(segments) >= 2:
        return segments[-2], segments[-1]
    if len(segments) == 1 and repo_owner:
        return repo_owner, segments[0]

    return repo_owner, path


def _normalize_github_owner_repo(owner: str, repo: str, fallback_owner: str = "") -> tuple[str, str]:
    owner = (owner or "").strip()
    repo = (repo or "").strip()

    if "github.com" in repo or repo.startswith(("http://", "https://", "//")):
        return _split_repo_path(repo, fallback_owner or owner)

    if owner in {"https:", "http:"} or owner.startswith(("http://", "https://")):
        combined = repo
        if combined.startswith("//"):
            combined = f"https:{combined}"
        elif combined.startswith("/"):
            combined = f"https://{combined.lstrip('/')}"
        elif not combined.startswith("http"):
            combined = f"{owner.rstrip(':')}//{repo.lstrip('/')}"
        return _split_repo_path(combined, fallback_owner)

    if "/" in repo and not repo.startswith("http"):
        parsed_owner, parsed_repo = repo.split("/", 1)
        return parsed_owner, parsed_repo.removesuffix(".git")

    clean_owner = owner if owner not in {"https:", "http:"} else fallback_owner
    return clean_owner, repo.removesuffix(".git")


def _normalize_github_tool_args(tool_call: dict[str, Any], fallback_owner: str = "") -> None:
    if tool_call.get("name") not in _GITHUB_TOOLS:
        return

    args = tool_call.setdefault("args", {})
    owner, repo = _normalize_github_owner_repo(
        str(args.get("owner", "")),
        str(args.get("repo", "")),
        fallback_owner,
    )
    args["owner"] = owner
    args["repo"] = repo


def _strip_code_fence(content: str) -> str:
    content = content.strip()
    if not content.startswith("```"):
        return content

    lines = content.split("\n")
    if lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return "\n".join(lines[1:]).strip()


def _extract_json_object_candidate(content: str) -> str:
    content = _strip_code_fence(content)
    if content.startswith("{"):
        return content

    for marker in ('{"type"', '{"name"', '{"function"'):
        idx = content.find(marker)
        if idx != -1:
            return content[idx:]
    return content


def _decode_json_string_body(raw: str, *, allow_truncated: bool) -> str:
    out: list[str] = []
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == '"' and not allow_truncated:
            break
        if ch == '"' and allow_truncated:
            ahead = raw[i + 1 :].lstrip()
            if ahead.startswith((",", "}")):
                break
        if ch == "\\" and i + 1 < len(raw):
            nxt = raw[i + 1]
            escapes = {
                "n": "\n",
                "r": "\r",
                "t": "\t",
                '"': '"',
                "\\": "\\",
                "/": "/",
            }
            if nxt in escapes:
                out.append(escapes[nxt])
                i += 2
                continue
            if nxt == "u" and i + 5 < len(raw):
                try:
                    out.append(chr(int(raw[i + 2 : i + 6], 16)))
                    i += 6
                    continue
                except ValueError:
                    pass
        out.append(ch)
        i += 1

    text = "".join(out).rstrip("\\")
    if allow_truncated and text and "informe truncado" not in text.lower():
        text += "\n\n> Nota: informe truncado por límite de respuesta del modelo."
    return text


def _parse_text_tool_call_lenient(content: str) -> tuple[str, dict[str, Any]] | None:
    content = _extract_json_object_candidate(content)
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', content)
    if not name_match:
        function_match = re.search(
            r'"function"\s*:\s*\{[^}]*"name"\s*:\s*"([^"]+)"',
            content,
        )
        if not function_match:
            return None
        name = function_match.group(1)
    else:
        name = name_match.group(1)

    args: dict[str, Any] = {}
    for key in (
        "owner",
        "repo",
        "channel",
        "thread_ts",
        "path",
        "file_path",
        "rule_id",
        "cve_id",
        "message",
        "body",
        "text",
    ):
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)"', content)
        if match:
            args[key] = match.group(1)

    pr_match = re.search(r'"pr_number"\s*:\s*"?([^",}\s]+)"?', content)
    if pr_match:
        args["pr_number"] = pr_match.group(1)

    line_match = re.search(r'"line(?:_number)?"\s*:\s*(\d+)', content)
    if line_match:
        args["line"] = int(line_match.group(1))

    for long_field in ("vulnerabilidades_md", "comment", "markdown"):
        field_match = re.search(rf'"{re.escape(long_field)}"\s*:\s*"', content)
        if not field_match:
            continue
        body = content[field_match.end() :]
        tail = body.rstrip()
        truncated = not tail.endswith(('"', '"}', '",'))
        args[long_field] = _decode_json_string_body(body, allow_truncated=truncated)

    if name in _GITHUB_TOOLS:
        if not all(args.get(key) for key in ("owner", "repo", "pr_number")):
            return None
        if not args.get("vulnerabilidades_md"):
            return None

    if not args:
        return None
    return name, args


def _apply_text_tool_call(
    messages: list[BaseMessage],
    last: AIMessage,
    name: str,
    args: dict[str, Any],
    fallback_owner: str = "",
) -> AIMessage:
    tool_call = {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "name": name,
        "args": args or {},
        "type": "tool_call",
    }
    _normalize_github_tool_args(tool_call, fallback_owner)
    new_ai = AIMessage(
        content="",
        tool_calls=[tool_call],
        id=last.id,
        response_metadata=getattr(last, "response_metadata", {}) or {},
    )
    messages[-1] = new_ai
    logger.warning(
        "Tool-call en texto convertida a invocación estructurada: %s",
        name,
    )
    return new_ai


def _coerce_text_tool_calls(
    messages: list[BaseMessage],
    fallback_owner: str = "",
) -> AIMessage | None:
    if not messages:
        return False

    last = messages[-1]
    if not isinstance(last, AIMessage) or last.tool_calls:
        return None

    content = _extract_json_object_candidate(_message_content(last.content).strip())

    if not content.startswith("{"):
        parsed = _parse_text_tool_call_lenient(content)
        if parsed:
            name, args = parsed
            return _apply_text_tool_call(messages, last, name, args, fallback_owner)
        return None

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        try:
            payload, _ = decoder.raw_decode(content)
        except json.JSONDecodeError:
            parsed = _parse_text_tool_call_lenient(content)
            if parsed:
                name, args = parsed
                return _apply_text_tool_call(messages, last, name, args, fallback_owner)
            logger.warning("No se pudo parsear tool-call JSON del LLM")
            return None

    if not isinstance(payload, dict):
        return None

    name = payload.get("name")
    args = payload.get("parameters") or payload.get("args")
    if payload.get("type") == "function" and name:
        args = args or {}
    elif isinstance(payload.get("function"), dict):
        function = payload["function"]
        name = function.get("name")
        raw_args = function.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
        else:
            args = raw_args or {}
    elif name and args is not None:
        pass
    else:
        return None

    if not name:
        return None

    return _apply_text_tool_call(messages, last, name, args or {}, fallback_owner)


class SingleToolCallMiddleware(AgentMiddleware):
    """Fuerza una sola tool-call por turno (requerido por vLLM/Llama)."""

    def __init__(self, github_owner_fallback: str = ""):
        self.github_owner_fallback = github_owner_fallback

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

    def _postprocess_model_response(self, state) -> dict | None:
        messages = state.get("messages", [])
        coerced_ai = _coerce_text_tool_calls(messages, self.github_owner_fallback)

        if not messages:
            return {"messages": [coerced_ai]} if coerced_ai is not None else None

        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            for tool_call in last.tool_calls:
                _normalize_github_tool_args(tool_call, self.github_owner_fallback)
            if len(last.tool_calls) > 1:
                self._truncate_extra_tool_calls(state)

        if coerced_ai is not None:
            return {"messages": [coerced_ai]}
        return None

    def wrap_model_call(self, request, handler):
        self._apply_single_tool_call_setting(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        self._apply_single_tool_call_setting(request)
        return await handler(request)

    def after_model(self, state, runtime):
        return self._postprocess_model_response(state)

    async def aafter_model(self, state, runtime):
        return self._postprocess_model_response(state)


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
_model_max_context = int(os.getenv("MODEL_MAX_CONTEXT_TOKENS", "20000"))
_input_token_reserve = int(os.getenv("INPUT_TOKEN_RESERVE", "17000"))
_configured_max_completion = int(os.getenv("MAX_COMPLETION_TOKENS", "4096"))
max_completion_tokens = min(
    _configured_max_completion,
    max(256, _model_max_context - _input_token_reserve),
)
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

(repetir bloque ### por CADA CVE de Trivy dentro del MISMO comentario)

### B) OPENGREP → publicar_comentario_general_pr (UN comentario consolidado)

Usa la MISMA herramienta `publicar_comentario_general_pr` con el campo `vulnerabilidades_md`.
NO uses `publicar_comentario_linea_pr` en el flujo de triage.

Estructura mínima del campo `vulnerabilidades_md` (sección SAST):

## Resumen ejecutivo SAST (OpenGrep)
- Total hallazgos: N
- Críticos: X | Altos: Y | Medios: Z | Bajos: W
- Acción prioritaria: (1-2 frases)

## Tabla de hallazgos de código
| Severidad | Rule ID | CWE | Archivo:Línea | Mensaje |
|-----------|---------|-----|---------------|---------|
| ERROR     | rule-id | CWE-78 | src/Foo.java:42 | ... |

## Detalle por hallazgo SAST

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

---

(repetir bloque ### por CADA hallazgo OpenGrep dentro del MISMO comentario)

### Reglas de publicación en GitHub (OBLIGATORIAS)
- Máximo 2 comentarios por análisis: 1 para TRIVY (si hay hallazgos) y 1 para OPENGREP (si hay hallazgos).
- PROHIBIDO publicar un comentario GitHub por cada CVE, GHSA o hallazgo individual.
- PROHIBIDO usar publicar_comentario_linea_pr para CVE/GHSA, dependencias (pom.xml, package.json) o paquetes Maven/npm.
- PROHIBIDO inventar archivos o líneas de código para hallazgos de dependencias (SCA).
- Todos los hallazgos de una fuente van consolidados en un único `vulnerabilidades_md`.

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
        "Publicación en GitHub (máximo 2 comentarios por PR):\n"
        "- TRIVY → publicar_comentario_general_pr: EXACTAMENTE UN comentario con TODOS los CVE "
        "en vulnerabilidades_md (tabla + detalle por CVE).\n"
        "- OPENGREP → publicar_comentario_general_pr: EXACTAMENTE UN comentario con TODOS "
        "los hallazgos SAST en vulnerabilidades_md (tabla + detalle por rule_id).\n"
        "- NO uses publicar_comentario_linea_pr en este flujo de triage.\n"
        "- PROHIBIDO: un comentario por CVE/hallazgo; PROHIBIDO: line comments para "
        "dependencias, pom.xml, CVE o GHSA.\n\n"
        "Calidad de remediación:\n"
        "- Consulta query_company_coding_standards por CVE, CWE o rule_id antes de redactar.\n"
        "- Incluye impacto, pasos concretos, fragmento de código/config y checklist de validación.\n"
        "- Consolida todos los hallazgos de cada fuente en un solo Markdown antes de publicar.\n"
        "- NO finalices hasta publicar los comentarios consolidados (SCA y/o SAST) y notificar Slack.\n"
        "- Invoca como máximo UNA herramienta por turno.\n"
        "- Usa SIEMPRE tool_calls nativas del API; NO escribas JSON de funciones en el texto."
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
        "1. (Opcional) Consulta query_company_coding_standards para enriquecer contexto.\n"
        "2. Si hay hallazgos TRIVY (>0 en checklist): redacta UN SOLO comentario SCA consolidado y publícalo "
        "con publicar_comentario_general_pr (owner={repo_owner}, repo={repo_name}, pr_number={pull_request_number}). "
        "Incluye TODOS los CVE en vulnerabilidades_md. NO llames esta tool más de una vez para Trivy.\n"
        "   Si TRIVY=0 hallazgos: NO publiques comentario SCA.\n"
        "3. Si hay hallazgos OPENGREP (>0 en checklist): redacta UN SOLO comentario SAST consolidado y publícalo "
        "con publicar_comentario_general_pr (mismos owner/repo/pr). "
        "Incluye TODOS los hallazgos OpenGrep en vulnerabilidades_md con plantilla SAST. "
        "NO uses publicar_comentario_linea_pr.\n"
        "   Si OPENGREP=0 hallazgos: NO publiques comentario SAST.\n"
        "4. Notifica fin de análisis en Slack (resumen: total Trivy, total OpenGrep, acciones).\n"
        "Usa SOLO owner y repo cortos (ej. lcoronad, unsecure-quarkus-app); nunca URLs completas.\n\n"
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
TRIAGE_FINDING_DESC_MAX_CHARS = int(os.getenv("TRIAGE_FINDING_DESC_MAX_CHARS", "180"))
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
    owner, repo = _normalize_github_owner_repo(owner, repo)
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
    Publica comentario consolidado en el PR (SCA y/o SAST).

    Usa esta herramienta para TODOS los hallazgos de Trivy y OpenGrep.
    Máximo 1 llamada por fuente (1 para SCA, 1 para SAST).

    El campo `vulnerabilidades_md` debe incluir TODOS los hallazgos de la fuente:
    resumen ejecutivo, tabla, y bloque detallado por CVE (Trivy) o rule_id (OpenGrep).
    Ver GITHUB_REMEDIATION_GUIDE. NO publiques un comentario por hallazgo individual.
    """
    owner, repo = _normalize_github_owner_repo(owner, repo)

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


async def _build_agent(system_prompt: str, github_owner_fallback: str = ""):
    tools = await _get_agent_tools()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=_system_prompt_with_rules(system_prompt),
        middleware=[SingleToolCallMiddleware(github_owner_fallback=github_owner_fallback)],
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


def _has_unresolved_tool_calls(messages: Sequence[BaseMessage]) -> bool:
    if not messages:
        return False

    last = messages[-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return False

    pending_ids = [
        tool_call.get("id")
        for tool_call in last.tool_calls
        if tool_call.get("id")
    ]
    if not pending_ids:
        return True

    executed_ids = {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolMessage) and message.tool_call_id
    }
    return any(tool_id not in executed_ids for tool_id in pending_ids)


def _final_ai_message(messages: Sequence[BaseMessage]) -> AIMessage | None:
    last_ai: AIMessage | None = None
    for message in messages:
        if isinstance(message, AIMessage):
            last_ai = message
    if last_ai is None or last_ai.tool_calls:
        return None
    return last_ai if last_ai.content else None


async def _execute_tool_call(tool_call: dict[str, Any], tools: list) -> ToolMessage:
    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {}) or {}
    tool_call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:12]}"
    selected = next((tool for tool in tools if tool.name == tool_name), None)
    if selected is None:
        content = f"Error: herramienta no encontrada: {tool_name}"
    else:
        try:
            if hasattr(selected, "ainvoke"):
                content = await selected.ainvoke(tool_args)
            else:
                content = selected.invoke(tool_args)
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
        except Exception as exc:
            logger.exception("Error ejecutando tool %s manualmente", tool_name)
            content = f"Error ejecutando {tool_name}: {exc}"
    return ToolMessage(
        content=str(content),
        tool_call_id=tool_call_id,
        name=tool_name,
    )


async def _manual_run_pending_tools(
    messages: list[BaseMessage],
    tools: list,
    github_owner_fallback: str = "",
) -> list[BaseMessage] | None:
    patched = list(messages)
    coerced_ai = _coerce_text_tool_calls(patched, github_owner_fallback)
    if coerced_ai is not None:
        pending = coerced_ai.tool_calls or []
    else:
        last = patched[-1] if patched else None
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return None
        pending = last.tool_calls

    if not pending:
        return None

    for tool_call in pending:
        patched.append(await _execute_tool_call(tool_call, tools))
    return patched


async def _run_agent(
    agent,
    prompt: str,
    prior_messages: Sequence[BaseMessage] | None = None,
    *,
    context: str = "agent",
    github_owner_fallback: str = "",
    tools: list | None = None,
) -> dict:
    agent_tools = tools or await _get_agent_tools()
    messages = list(prior_messages or []) + [HumanMessage(content=prompt)]
    agent_config = {"recursion_limit": GRAPH_RECURSION_LIMIT}
    logger.info("[%s] Iniciando ejecución del agente", context)

    async def _drain_agent(input_messages: list[BaseMessage]) -> dict:
        result: dict | None = None
        seen_messages = 0

        try:
            async for state in agent.astream(
                {"messages": input_messages},
                config=agent_config,
                stream_mode="values",
            ):
                current_messages = state.get("messages", [])
                if _coerce_text_tool_calls(current_messages, github_owner_fallback):
                    logger.warning(
                        "[%s] Tool-call en texto reparada durante stream",
                        context,
                    )
                for message in current_messages[seen_messages:]:
                    _log_message(message, context)
                seen_messages = len(current_messages)
                result = state
        except ValueError as exc:
            if "No AIMessage found in input" not in str(exc):
                raise
            logger.warning(
                "[%s] ToolNode falló (%s); ejecutando tools manualmente",
                context,
                exc,
            )
            base_messages = list(result.get("messages", [])) if result else list(input_messages)
            patched_messages = await _manual_run_pending_tools(
                base_messages,
                agent_tools,
                github_owner_fallback,
            )
            if patched_messages is None:
                raise
            return {"messages": patched_messages}

        if result is None:
            try:
                result = await agent.ainvoke(
                    {"messages": input_messages},
                    config=agent_config,
                )
            except ValueError as exc:
                if "No AIMessage found in input" not in str(exc):
                    raise
                logger.warning(
                    "[%s] ToolNode falló en ainvoke; ejecutando tools manualmente",
                    context,
                )
                patched_messages = await _manual_run_pending_tools(
                    list(input_messages),
                    agent_tools,
                    github_owner_fallback,
                )
                if patched_messages is None:
                    raise
                return {"messages": patched_messages}

            current_messages = result.get("messages", [])
            if _coerce_text_tool_calls(current_messages, github_owner_fallback):
                logger.warning("[%s] Tool-call en texto reparada tras ainvoke", context)
            for message in current_messages[seen_messages:]:
                _log_message(message, context)

        return result

    result = await _drain_agent(messages)
    continuation_step = 0

    while (
        _has_unresolved_tool_calls(result.get("messages", []))
        and continuation_step < GRAPH_RECURSION_LIMIT
    ):
        continuation_step += 1
        current_messages = list(result.get("messages", []))
        if _coerce_text_tool_calls(current_messages, github_owner_fallback):
            logger.warning(
                "[%s] Tool-call en texto reparada antes de continuar (paso %s)",
                context,
                continuation_step,
            )
        logger.warning(
            "[%s] Hay tool-call pendiente; continuando ejecución del agente (paso %s)",
            context,
            continuation_step,
        )
        result = await _drain_agent(current_messages)

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
    trivy_json: list[dict[str, Any]] | dict[str, Any]
    opengrep_sarif: list[dict[str, Any]] | dict[str, Any]


def _normalize_repo_file_path(path: str, repo_name: str) -> str:
    normalized = path.lstrip("./")
    prefix = f"{repo_name}/"
    if normalized.startswith(prefix):
        return normalized[len(prefix) :]
    return normalized


def _extract_trivy_findings(trivy_json: list | dict) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if isinstance(trivy_json, list):
        for result in trivy_json:
            findings.append(
                {
                    "id": result.get("id", "UNKNOWN"),
                    "pkg": result.get("pkg", ""),
                    "severity": result.get("severity", ""),
                    "target": result.get("target", ""),
                    "fixed_version": result.get("fixed_version", ""),
                    "description": result.get("description", ""),
                }
            )
        return findings

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


def _truncate_field(text: str, max_len: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len].rstrip()}..."


def _compact_trivy_for_prompt(trivy_json: list | dict) -> list[dict[str, Any]]:
    return [
        {
            "id": finding["id"],
            "pkg": finding["pkg"],
            "severity": finding["severity"],
            "target": finding["target"],
            "fixed_version": finding["fixed_version"],
            "description": _truncate_field(
                finding.get("description", ""), TRIAGE_FINDING_DESC_MAX_CHARS
            ),
        }
        for finding in _extract_trivy_findings(trivy_json)
    ]


def _compact_opengrep_for_prompt(opengrep_sarif: list | dict) -> list[dict[str, Any]]:
    return [
        {
            "rule_id": finding["rule_id"],
            "path": finding["path"],
            "line": finding["line"],
            "severity": finding["severity"],
            "cwe": finding["cwe"],
            "message": _truncate_field(finding.get("message", ""), TRIAGE_FINDING_DESC_MAX_CHARS),
        }
        for finding in _extract_opengrep_findings(opengrep_sarif)
    ]


def _severity_summary(findings: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for finding in findings:
        severity = str(finding.get("severity") or "UNKNOWN").upper()
        counts[severity] = counts.get(severity, 0) + 1
    if not counts:
        return "sin hallazgos"
    return ", ".join(f"{severity}={count}" for severity, count in sorted(counts.items()))


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


def _extract_opengrep_findings(opengrep_sarif: list | dict) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if isinstance(opengrep_sarif, list):
        for result in opengrep_sarif:
            findings.append(
                {
                    "rule_id": result.get("rule_id", "UNKNOWN"),
                    "path": result.get("path", ""),
                    "line": result.get("line", 0),
                    "severity": result.get("severity", ""),
                    "message": result.get("message", ""),
                    "cwe": result.get("cwe", ""),
                }
            )
        return findings

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
        f"### TRIVY ({len(trivy_findings)} hallazgos) → UN SOLO publicar_comentario_general_pr",
    ]

    if trivy_findings:
        lines.append(
            f"   Resumen severidad: {_severity_summary(trivy_findings)}. "
            f"Incluir los {len(trivy_findings)} CVE/GHSA en un único vulnerabilidades_md."
        )
        trivy_ids = ", ".join(finding["id"] for finding in trivy_findings)
        lines.append(f"   IDs: {trivy_ids}")
    else:
        lines.append("- Sin hallazgos Trivy. Omitir comentario SCA.")

    lines.append("")
    lines.append(
        f"### OPENGREP ({len(opengrep_findings)} hallazgos) → UN SOLO publicar_comentario_general_pr"
    )

    if opengrep_findings:
        lines.append(
            f"   Incluir los {len(opengrep_findings)} hallazgos SAST en un único vulnerabilidades_md "
            f"(tabla + detalle). NO usar publicar_comentario_linea_pr."
        )
        for idx, finding in enumerate(opengrep_findings, start=1):
            path = _normalize_repo_file_path(finding["path"], repo_name)
            line = finding["line"] if finding["line"] is not None else "?"
            lines.append(
                f"{idx}. [{finding['severity']}] {finding['rule_id']} "
                f"en {path}:{line} — {finding['message']}"
            )
    else:
        lines.append("- Sin hallazgos OpenGrep. Omitir comentario SAST.")

    lines.append("")
    lines.append("### Formato esperado en GitHub")
    lines.append("- Máximo 2 comentarios: 1 consolidado SCA (Trivy) + 1 consolidado SAST (OpenGrep).")
    lines.append("- Cada comentario incluye resumen, tabla y detalle de TODOS sus hallazgos.")
    lines.append("")
    lines.append(
        "IMPORTANTE: Consolida todos los hallazgos antes de publicar. "
        "NO publiques un comentario GitHub por cada CVE o rule_id."
    )
    return "\n".join(lines)


def _build_triage_prompt(data: TriageRequest) -> str:
    owner, repo_name = _split_repo_path(data.repo_path, data.repo_owner)
    guide = github_remediation_guide
    if "{github_remediation_guide}" not in triage_user_prompt_template:
        guide = ""

    return triage_user_prompt_template.format(
        trivy_json=json.dumps(
            _compact_trivy_for_prompt(data.trivy_json), ensure_ascii=False
        )[:TRIVY_JSON_MAX_CHARS],
        opengrep_sarif=json.dumps(
            _compact_opengrep_for_prompt(data.opengrep_sarif), ensure_ascii=False
        )[:OPENGREP_JSON_MAX_CHARS],
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
        owner, _ = _split_repo_path(data.repo_path, data.repo_owner)
        tools = await _get_agent_tools()
        agent = await _build_agent(triage_system_prompt, github_owner_fallback=owner)
        prompt = _build_triage_prompt(data)
        await _run_agent(
            agent,
            prompt,
            context=f"triage:{data.repo_path}#PR{data.pull_request_number}",
            github_owner_fallback=owner,
            tools=tools,
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
