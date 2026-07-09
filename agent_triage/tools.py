"""
Herramientas (tools) disponibles para el agente de triage.

Integra tres fuentes de capacidades:
1. MCP (Slack, GitHub adicionales) vía SSE.
2. RAG interno (Milvus + embeddings Granite) para normas de codificación.
3. Publicación directa en GitHub REST API para comentarios de PR.
"""

from __future__ import annotations

import logging
from typing import List

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pymilvus import MilvusClient

from agent_triage.constants import settings
from agent_triage.helpers import mcp_sse_url, normalize_github_owner_repo

logger = logging.getLogger(__name__)


class LlamaStackGraniteEmbeddings(Embeddings):
    """
    Adaptador de embeddings para Llama Stack.

    Envía textos como strings puros al endpoint /v1/embeddings, evitando
    incompatibilidades del SDK OpenAI con ciertos formatos de input.
    """

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            payload = {"model": self.model, "input": str(text).strip()}
            response = requests.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            response.raise_for_status()
            vectors.append(response.json()["data"][0]["embedding"])
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class ToolRegistry:
    """
    Registro y fábrica de herramientas del agente.

    Encapsula la conexión a Milvus, MCP y la creación del conjunto completo
    de tools que LangGraph expone al LLM.
    """

    def __init__(self) -> None:
        self._embeddings = LlamaStackGraniteEmbeddings(
            base_url=settings.embedding_api_url,
            model=settings.embeddings_model_name,
        )

    def milvus_uri(self) -> str:
        if settings.milvus_uri:
            return settings.milvus_uri
        host = settings.milvus_host.strip()
        port = settings.milvus_port.strip()
        if host.startswith(("http://", "https://")):
            return host if host.endswith(f":{port}") else f"{host.rstrip('/')}:{port}"
        return f"http://{host}:{port}"

    def search_company_standards(self, query: str) -> list[str]:
        """Búsqueda semántica en la colección de normas internas de la empresa."""
        client = MilvusClient(uri=self.milvus_uri())
        query_vector = self._embeddings.embed_query(query)
        results = client.search(
            collection_name=settings.milvus_collection_name,
            data=[query_vector],
            limit=settings.milvus_search_limit,
            search_params={"metric_type": settings.milvus_metric_type, "params": {}},
            output_fields=["text"],
        )
        if not results or not results[0]:
            return []
        return [
            hit.get("entity", {}).get("text", "")
            for hit in results[0]
            if hit.get("entity", {}).get("text", "")
        ]

    def mcp_client(self) -> MultiServerMCPClient:
        """Cliente multi-servidor MCP para Slack y GitHub."""
        return MultiServerMCPClient(
            {
                "slack": {
                    "transport": "sse",
                    "url": mcp_sse_url(settings.slack_tool_url),
                },
                "github": {
                    "transport": "sse",
                    "url": mcp_sse_url(settings.github_tool_url),
                },
            }
        )

    async def get_all_tools(self) -> list:
        """
        Combina tools MCP remotas con las locales de triage/GitHub/RAG.

        El orden no es crítico; LangGraph las registra por nombre.
        """
        client = self.mcp_client()
        tools = (
            await client.get_tools()
            + [
                query_company_coding_standards,
                publicar_comentario_linea_pr,
                publicar_comentario_general_pr,
            ]
        )
        logger.info(
            "Tools cargadas (%s): %s",
            len(tools),
            ", ".join(tool.name for tool in tools),
        )
        return tools


# Instancia compartida del registro de tools.
tool_registry = ToolRegistry()


@tool
def query_company_coding_standards(query: str) -> str:
    """Útil para buscar lineamientos de codificación segura de la empresa."""
    try:
        docs = tool_registry.search_company_standards(query)
        if not docs:
            return "No se encontraron normas internas relacionadas con la consulta."
        return "\n\n".join(f"[Norma]: {doc}" for doc in docs)
    except Exception as exc:
        logger.exception("Error consultando Milvus para query=%s", query)
        return f"Error al consultar normas en Milvus: {exc}"


@tool
def publicar_comentario_linea_pr(
    owner: str,
    repo: str,
    pr_number: int,
    commit_id: str,
    path: str,
    line: int,
    recomendacion: str,
) -> str:
    """
    Publica comentario SAST inline (OPENGREP) en el PR.

    El campo `recomendacion` debe ser Markdown estructurado con:
    Problema detectado, Riesgo/impacto, Causa raíz, Remediación (pasos),
    Código sugerido (bloque ```), Referencias (CWE/rule_id/normas) y Validación (checklist).
    """
    owner, repo = normalize_github_owner_repo(owner, repo)
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {settings.github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {
        "body": f"🚨 **Alerta de Seguridad (SAST)**\n\n{recomendacion}",
        "commit_id": commit_id,
        "path": path,
        "line": line,
        "side": "RIGHT",
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)

    if response.status_code == 201:
        return "✅ Comentario publicado exitosamente en la línea de código."

    if response.status_code == 422 and "could not be resolved" in response.text:
        logger.warning(
            "Línea %s fuera del diff del PR %s/%s#%s; usando fallback general",
            line,
            owner,
            repo,
            pr_number,
        )
        url_general = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
        payload_general = {
            "body": (
                f"🚨 **Alerta de Seguridad (SAST) - Código Preexistente**\n\n"
                f"📍 *Se detectó una vulnerabilidad en el archivo `{path}` en la línea `{line}`, "
                f"pero esta línea no fue modificada en este PR.*\n\n"
                f"{recomendacion}"
            )
        }
        fallback_resp = requests.post(url_general, headers=headers, json=payload_general, timeout=30)
        if fallback_resp.status_code == 201:
            return (
                "✅ El comentario no pudo anclarse a la línea (fuera del diff), "
                "pero se publicó exitosamente como comentario general en el PR."
            )
        return f"❌ Error crítico en el Fallback: {fallback_resp.status_code} - {fallback_resp.text}"

    return f"❌ Error {response.status_code}: {response.text}"


@tool
def publicar_comentario_general_pr(
    owner: str,
    repo: str,
    pr_number: int,
    vulnerabilidades_md: str,
) -> str:
    """
    Publica comentario consolidado en el PR (SCA y/o SAST).

    Usa esta herramienta para TODOS los hallazgos de Trivy y OpenGrep.
    Máximo 1 llamada por fuente (1 para SCA, 1 para SAST).
    """
    owner, repo = normalize_github_owner_repo(owner, repo)
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {settings.github_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    payload = {
        "body": f"📦 **Alerta de Seguridad en Dependencias (SCA)**\n\n{vulnerabilidades_md}"
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    if response.status_code == 201:
        return "✅ Comentario general de dependencias publicado exitosamente."
    return f"❌ Error {response.status_code}: {response.text}"
