"""
Funciones de ayuda transversales del agente de triage.

Agrupa utilidades sin estado para:
- Logging estructurado de mensajes LangChain.
- Normalización de rutas y repositorios GitHub.
- Coerción de tool-calls emitidas como JSON en texto (compatibilidad Llama/vLLM).
- Extracción y compactación de hallazgos Trivy/OpenGrep.
- Construcción de prompts y checklists para el LLM.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Sequence
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agent_triage.constants import GITHUB_TOOLS, SINGLE_TOOL_CALL_RULE, settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging y contenido de mensajes
# ---------------------------------------------------------------------------


def truncate_text(text: str, max_len: int | None = None) -> str:
    """Trunca texto largo para logs legibles sin saturar el output."""
    limit = max_len if max_len is not None else settings.log_max_chars
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncado, total={len(text)} chars]"


def message_content(content: Any) -> str:
    """Convierte el contenido multimodal de LangChain a string plano."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", json.dumps(block, ensure_ascii=False)))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def log_message(message: BaseMessage, context: str) -> None:
    """Registra un mensaje del grafo con formato consistente según su tipo."""
    content = message_content(message.content)

    if isinstance(message, HumanMessage):
        logger.info("[%s] INPUT usuario: %s", context, truncate_text(content))
        return

    if isinstance(message, AIMessage):
        for tool_call in message.tool_calls or []:
            logger.info(
                "[%s] LLM -> invoca tool: %s | args=%s",
                context,
                tool_call.get("name"),
                truncate_text(
                    json.dumps(tool_call.get("args", {}), ensure_ascii=False),
                    800,
                ),
            )
        if content:
            logger.info("[%s] LLM respuesta: %s", context, truncate_text(content))
        return

    if isinstance(message, ToolMessage):
        logger.info(
            "[%s] MCP/tool resultado [%s]: %s",
            context,
            message.name or "unknown",
            truncate_text(content),
        )
        return

    logger.info("[%s] %s: %s", context, message.__class__.__name__, truncate_text(content))


# ---------------------------------------------------------------------------
# URLs y repositorios GitHub
# ---------------------------------------------------------------------------


def mcp_sse_url(url: str) -> str:
    """Asegura que la URL del servidor MCP termine en /sse."""
    normalized = url.rstrip("/")
    return normalized if normalized.endswith("/sse") else f"{normalized}/sse"


def split_repo_path(repo_path: str, repo_owner: str) -> tuple[str, str]:
    """
    Extrae (owner, repo) desde URLs GitHub, SSH o rutas relativas.

    Soporta formatos comunes en webhooks de Tekton y payloads del pipeline.
    """
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


def normalize_github_owner_repo(
    owner: str,
    repo: str,
    fallback_owner: str = "",
) -> tuple[str, str]:
    """Corrige owner/repo malformados que el LLM a veces envía en tool args."""
    owner = (owner or "").strip()
    repo = (repo or "").strip()

    if "github.com" in repo or repo.startswith(("http://", "https://", "//")):
        return split_repo_path(repo, fallback_owner or owner)

    if owner in {"https:", "http:"} or owner.startswith(("http://", "https://")):
        combined = repo
        if combined.startswith("//"):
            combined = f"https:{combined}"
        elif combined.startswith("/"):
            combined = f"https://{combined.lstrip('/')}"
        elif not combined.startswith("http"):
            combined = f"{owner.rstrip(':')}//{repo.lstrip('/')}"
        return split_repo_path(combined, fallback_owner)

    if "/" in repo and not repo.startswith("http"):
        parsed_owner, parsed_repo = repo.split("/", 1)
        return parsed_owner, parsed_repo.removesuffix(".git")

    clean_owner = owner if owner not in {"https:", "http:"} else fallback_owner
    return clean_owner, repo.removesuffix(".git")


def normalize_github_tool_args(tool_call: dict[str, Any], fallback_owner: str = "") -> None:
    """Normaliza in-place los argumentos owner/repo de tools GitHub."""
    if tool_call.get("name") not in GITHUB_TOOLS:
        return

    args = tool_call.setdefault("args", {})
    owner, repo = normalize_github_owner_repo(
        str(args.get("owner", "")),
        str(args.get("repo", "")),
        fallback_owner,
    )
    args["owner"] = owner
    args["repo"] = repo


def system_prompt_with_rules(system_prompt: str) -> str:
    """Añade la regla de una sola tool-call por turno si no está ya presente."""
    if settings.parallel_tool_calls:
        return system_prompt
    if SINGLE_TOOL_CALL_RULE.strip() in system_prompt:
        return system_prompt
    return f"{system_prompt.rstrip()}{SINGLE_TOOL_CALL_RULE}"


# ---------------------------------------------------------------------------
# Coerción de tool-calls en texto (redhataillama / vLLM)
# ---------------------------------------------------------------------------


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
    """Decodifica un string JSON escapado; tolera truncamiento al final."""
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
            escapes = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\", "/": "/"}
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


def parse_text_tool_call_lenient(content: str) -> tuple[str, dict[str, Any]] | None:
    """
    Extrae name y args de un JSON de tool-call aunque esté truncado o malformado.

    Necesario porque algunos modelos Llama devuelven la función como texto en lugar
    de tool_calls nativas del API OpenAI.
    """
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

    if name in GITHUB_TOOLS:
        if not all(args.get(key) for key in ("owner", "repo", "pr_number")):
            return None
        if not args.get("vulnerabilidades_md"):
            return None

    if not args:
        return None
    return name, args


def apply_text_tool_call(
    messages: list[BaseMessage],
    last: AIMessage,
    name: str,
    args: dict[str, Any],
    fallback_owner: str = "",
) -> AIMessage:
    """
    Reemplaza el AIMessage de texto por uno con tool_calls estructuradas.

    LangGraph requiere un nuevo mensaje (estado inmutable), no mutación in-place.
    """
    tool_call = {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "name": name,
        "args": args or {},
        "type": "tool_call",
    }
    normalize_github_tool_args(tool_call, fallback_owner)
    new_ai = AIMessage(
        content="",
        tool_calls=[tool_call],
        id=last.id,
        response_metadata=getattr(last, "response_metadata", {}) or {},
    )
    messages[-1] = new_ai
    logger.warning("Tool-call en texto convertida a invocación estructurada: %s", name)
    return new_ai


def coerce_text_tool_calls(
    messages: list[BaseMessage],
    fallback_owner: str = "",
) -> AIMessage | None:
    """
    Detecta y convierte tool-calls en texto del último AIMessage.

    Retorna el nuevo AIMessage si hubo coerción, o None si no aplica.
    """
    if not messages:
        return None

    last = messages[-1]
    if not isinstance(last, AIMessage) or last.tool_calls:
        return None

    content = _extract_json_object_candidate(message_content(last.content).strip())

    if not content.startswith("{"):
        parsed = parse_text_tool_call_lenient(content)
        if parsed:
            name, args = parsed
            return apply_text_tool_call(messages, last, name, args, fallback_owner)
        return None

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        try:
            payload, _ = decoder.raw_decode(content)
        except json.JSONDecodeError:
            parsed = parse_text_tool_call_lenient(content)
            if parsed:
                name, args = parsed
                return apply_text_tool_call(messages, last, name, args, fallback_owner)
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

    return apply_text_tool_call(messages, last, name, args or {}, fallback_owner)


# ---------------------------------------------------------------------------
# Estado del grafo (mensajes)
# ---------------------------------------------------------------------------


def window_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Aplica ventana deslizante al historial para no exceder contexto."""
    if settings.memory_window_k <= 0:
        return list(messages)
    return list(messages[-settings.memory_window_k * 2 :])


def has_unresolved_tool_calls(messages: Sequence[BaseMessage]) -> bool:
    """Indica si el último AIMessage tiene tool_calls sin ToolMessage de respuesta."""
    if not messages:
        return False

    last = messages[-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return False

    pending_ids = [
        tool_call.get("id") for tool_call in last.tool_calls if tool_call.get("id")
    ]
    if not pending_ids:
        return True

    executed_ids = {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolMessage) and message.tool_call_id
    }
    return any(tool_id not in executed_ids for tool_id in pending_ids)


def final_ai_message(messages: Sequence[BaseMessage]) -> AIMessage | None:
    """Obtiene la última respuesta AI con contenido textual (sin tool_calls pendientes)."""
    last_ai: AIMessage | None = None
    for message in messages:
        if isinstance(message, AIMessage):
            last_ai = message
    if last_ai is None or last_ai.tool_calls:
        return None
    return last_ai if last_ai.content else None


# ---------------------------------------------------------------------------
# Hallazgos Trivy / OpenGrep
# ---------------------------------------------------------------------------


def normalize_repo_file_path(path: str, repo_name: str) -> str:
    normalized = path.lstrip("./")
    prefix = f"{repo_name}/"
    if normalized.startswith(prefix):
        return normalized[len(prefix) :]
    return normalized


def extract_trivy_findings(trivy_json: list | dict) -> list[dict[str, Any]]:
    """Normaliza reportes Trivy (raw JSON o lista pre-procesada del pipeline)."""
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


def truncate_field(text: str, max_len: int | None = None) -> str:
    limit = max_len if max_len is not None else settings.triage_finding_desc_max_chars
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def compact_trivy_for_prompt(trivy_json: list | dict) -> list[dict[str, Any]]:
    """Reduce el payload Trivy enviado al LLM conservando campos de triage."""
    return [
        {
            "id": finding["id"],
            "pkg": finding["pkg"],
            "severity": finding["severity"],
            "target": finding["target"],
            "fixed_version": finding["fixed_version"],
            "description": truncate_field(finding.get("description", "")),
        }
        for finding in extract_trivy_findings(trivy_json)
    ]


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


def extract_opengrep_findings(opengrep_sarif: list | dict) -> list[dict[str, Any]]:
    """Normaliza SARIF OpenGrep (raw o lista pre-procesada del pipeline)."""
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


def compact_opengrep_for_prompt(opengrep_sarif: list | dict) -> list[dict[str, Any]]:
    return [
        {
            "rule_id": finding["rule_id"],
            "path": finding["path"],
            "line": finding["line"],
            "severity": finding["severity"],
            "cwe": finding["cwe"],
            "message": truncate_field(finding.get("message", "")),
        }
        for finding in extract_opengrep_findings(opengrep_sarif)
    ]


def severity_summary(findings: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for finding in findings:
        severity = str(finding.get("severity") or "UNKNOWN").upper()
        counts[severity] = counts.get(severity, 0) + 1
    if not counts:
        return "sin hallazgos"
    return ", ".join(f"{severity}={count}" for severity, count in sorted(counts.items()))


def build_findings_checklist(data: Any) -> str:
    """
    Genera checklist estructurado que guía al LLM en la consolidación de comentarios.

    `data` debe exponer: repo_path, repo_owner, pull_request_number, commit_id,
    trivy_json y opengrep_sarif (compatible con TriageRequest).
    """
    owner, repo_name = split_repo_path(data.repo_path, data.repo_owner)
    trivy_findings = extract_trivy_findings(data.trivy_json)
    opengrep_findings = extract_opengrep_findings(data.opengrep_sarif)

    lines = [
        f"Owner={owner}, Repo={repo_name}, PR={data.pull_request_number}, Commit={data.commit_id}",
        "",
        f"### TRIVY ({len(trivy_findings)} hallazgos) → UN SOLO publicar_comentario_general_pr",
    ]

    if trivy_findings:
        lines.append(
            f"   Resumen severidad: {severity_summary(trivy_findings)}. "
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
            path = normalize_repo_file_path(finding["path"], repo_name)
            line = finding["line"] if finding["line"] is not None else "?"
            lines.append(
                f"{idx}. [{finding['severity']}] {finding['rule_id']} "
                f"en {path}:{line} — {finding['message']}"
            )
    else:
        lines.append("- Sin hallazgos OpenGrep. Omitir comentario SAST.")

    lines.extend(
        [
            "",
            "### Formato esperado en GitHub",
            "- Máximo 2 comentarios: 1 consolidado SCA (Trivy) + 1 consolidado SAST (OpenGrep).",
            "- Cada comentario incluye resumen, tabla y detalle de TODOS sus hallazgos.",
            "",
            "IMPORTANTE: Consolida todos los hallazgos antes de publicar. "
            "NO publiques un comentario GitHub por cada CVE o rule_id.",
        ]
    )
    return "\n".join(lines)


def build_triage_prompt(data: Any) -> str:
    """Construye el prompt de usuario para el flujo de triage de un PR."""
    owner, repo_name = split_repo_path(data.repo_path, data.repo_owner)
    template = settings.triage_user_prompt_template
    guide = settings.github_remediation_guide
    if "{github_remediation_guide}" not in template:
        guide = ""

    return template.format(
        trivy_json=json.dumps(compact_trivy_for_prompt(data.trivy_json), ensure_ascii=False)[
            : settings.trivy_json_max_chars
        ],
        opengrep_sarif=json.dumps(
            compact_opengrep_for_prompt(data.opengrep_sarif), ensure_ascii=False
        )[: settings.opengrep_json_max_chars],
        repo_path=data.repo_path,
        pull_request_number=data.pull_request_number,
        repo_owner=owner,
        repo_name=repo_name,
        commit_id=data.commit_id,
        findings_checklist=build_findings_checklist(data),
        github_remediation_guide=guide or settings.github_remediation_guide,
    )


def build_slack_prompt(event: dict, channel: str, thread_ts: str) -> str:
    return settings.slack_user_prompt_template.format(
        slack_text=event.get("text", ""),
        channel=channel,
        thread_ts=thread_ts,
    )
