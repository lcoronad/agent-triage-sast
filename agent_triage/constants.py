"""
Constantes y configuración del agente de triage.

Centraliza variables de entorno, límites operativos y plantillas de prompts.
Toda la configuración se carga una sola vez al importar el módulo, lo que
facilita inyección en tests y despliegues (ConfigMap de OpenShift).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _api_base(url: str) -> str:
    """Normaliza la URL base del API OpenAI-compatible añadiendo /v1 si falta."""
    normalized = url.rstrip("/")
    return normalized if normalized.endswith("/v1") else f"{normalized}/v1"


def _bool_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# Plantillas de prompts (valores por defecto; sobreescribibles vía env)
# ---------------------------------------------------------------------------

DEFAULT_GITHUB_REMEDIATION_GUIDE = """
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

DEFAULT_TRIAGE_SYSTEM_PROMPT = (
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
)

DEFAULT_SLACK_SYSTEM_PROMPT = (
    "Eres un asistente DevSecOps en Slack. Responde de forma clara y técnica. "
    "Usa las herramientas de Slack para publicar en el canal e hilo indicados. "
    "Consulta las normas internas de codificación cuando el desarrollador lo requiera."
)

DEFAULT_TRIAGE_USER_PROMPT = (
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
)

DEFAULT_SLACK_USER_PROMPT = (
    "El desarrollador preguntó en Slack: {slack_text}. "
    "Responde usando las herramientas de Slack en el canal '{channel}' "
    "e hilo '{thread_ts}' e investiga las normativas internas si es necesario."
)

SINGLE_TOOL_CALL_RULE = (
    "\n\nIMPORTANTE: Invoca como máximo UNA herramienta por turno. "
    "Espera el resultado antes de llamar la siguiente herramienta."
)

# Tools de GitHub que requieren normalización de owner/repo en los argumentos.
GITHUB_TOOLS = frozenset({"publicar_comentario_general_pr", "publicar_comentario_linea_pr"})


@dataclass(frozen=True)
class Settings:
    """
    Configuración inmutable del servicio.

    Agrupa credenciales, URLs de servicios externos y límites de contexto del LLM.
    """

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())
    log_max_chars: int = field(default_factory=lambda: int(os.getenv("LOG_MAX_CHARS", "2000")))

    # Agente / grafo
    memory_window_k: int = field(default_factory=lambda: int(os.getenv("MEMORY_WINDOW_K", "10")))
    graph_recursion_limit: int = field(
        default_factory=lambda: int(os.getenv("GRAPH_RECURSION_LIMIT", "50"))
    )
    parallel_tool_calls: bool = field(
        default_factory=lambda: _bool_env("PARALLEL_TOOL_CALLS", "false")
    )

    # LLM (vLLM / Red Hat AI Llama)
    qwen_api_url: str = field(
        default_factory=lambda: _api_base(
            os.getenv("QWEN_API_URL", "http://vllm-qwen-service:8000/v1")
        )
    )
    qwen_model_name: str = field(
        default_factory=lambda: os.getenv("QWEN_MODEL_NAME", "qwen2.5-coder:32b-instruct")
    )
    qwen_api_key: str = field(default_factory=lambda: os.getenv("QWEN_API_KEY", ""))
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.1")))
    top_p: float = field(default_factory=lambda: float(os.getenv("TOP_P", "0.1")))
    model_max_context_tokens: int = field(
        default_factory=lambda: int(os.getenv("MODEL_MAX_CONTEXT_TOKENS", "20000"))
    )
    input_token_reserve: int = field(
        default_factory=lambda: int(os.getenv("INPUT_TOKEN_RESERVE", "17000"))
    )
    max_completion_tokens: int = field(
        default_factory=lambda: min(
            int(os.getenv("MAX_COMPLETION_TOKENS", "4096")),
            max(
                256,
                int(os.getenv("MODEL_MAX_CONTEXT_TOKENS", "20000"))
                - int(os.getenv("INPUT_TOKEN_RESERVE", "17000")),
            ),
        )
    )

    # Embeddings / RAG (Milvus)
    embedding_api_url: str = field(
        default_factory=lambda: _api_base(
            os.getenv("EMBEDDING_API_URL", "http://vllm-qwen-service:8000/v1")
        )
    )
    embeddings_model_name: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDINGS_MODEL_NAME",
            "sentence-transformers/ibm-granite/granite-embedding-125m-english",
        )
    )
    embedding_api_key: str = field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY", ""))
    milvus_host: str = field(default_factory=lambda: os.getenv("MILVUS_HOST", "milvus-service"))
    milvus_port: str = field(default_factory=lambda: os.getenv("MILVUS_PORT", "19530"))
    milvus_uri: str = field(default_factory=lambda: os.getenv("MILVUS_URI", "").strip())
    milvus_collection_name: str = field(
        default_factory=lambda: os.getenv("MILVUS_COLLECTION_NAME", "company_coding_standards")
    )
    milvus_search_limit: int = field(
        default_factory=lambda: int(os.getenv("MILVUS_SEARCH_LIMIT", "3"))
    )
    milvus_metric_type: str = field(
        default_factory=lambda: os.getenv("MILVUS_METRIC_TYPE", "L2")
    )

    # Integraciones externas
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
    slack_tool_url: str = field(default_factory=lambda: os.getenv("SLACK_TOOL_URL", "http://cluster.local"))
    github_tool_url: str = field(default_factory=lambda: os.getenv("GITHUB_TOOL_URL", "http://cluster.local"))
    github_token: str = field(default_factory=lambda: os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", ""))

    # Límites de payload al LLM
    trivy_json_max_chars: int = field(
        default_factory=lambda: int(os.getenv("TRIVY_JSON_MAX_CHARS", "15000"))
    )
    opengrep_json_max_chars: int = field(
        default_factory=lambda: int(os.getenv("OPENGREP_JSON_MAX_CHARS", "20000"))
    )
    triage_finding_desc_max_chars: int = field(
        default_factory=lambda: int(os.getenv("TRIAGE_FINDING_DESC_MAX_CHARS", "180"))
    )

    # Prompts (sobreescribibles vía ConfigMap)
    github_remediation_guide: str = field(
        default_factory=lambda: os.getenv("GITHUB_REMEDIATION_GUIDE", DEFAULT_GITHUB_REMEDIATION_GUIDE)
    )
    triage_system_prompt: str = field(
        default_factory=lambda: os.getenv("TRIAGE_SYSTEM_PROMPT", DEFAULT_TRIAGE_SYSTEM_PROMPT)
    )
    slack_system_prompt: str = field(
        default_factory=lambda: os.getenv("SLACK_SYSTEM_PROMPT", DEFAULT_SLACK_SYSTEM_PROMPT)
    )
    triage_user_prompt_template: str = field(
        default_factory=lambda: os.getenv("TRIAGE_USER_PROMPT", DEFAULT_TRIAGE_USER_PROMPT)
    )
    slack_user_prompt_template: str = field(
        default_factory=lambda: os.getenv("SLACK_USER_PROMPT", DEFAULT_SLACK_USER_PROMPT)
    )

    # Servidor HTTP
    host: str = field(default_factory=lambda: os.getenv("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "7861")))


# Instancia global de configuración usada por el resto del paquete.
settings = Settings()
