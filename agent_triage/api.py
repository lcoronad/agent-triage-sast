"""
Capa REST del agente de triage.

Expone endpoints HTTP para:
- POST /api/v1/triage: dispara análisis asíncrono de hallazgos de un PR.
- POST /api/v1/slack/events: webhook de eventos Slack (url_verification + mentions).

Los handlers delegan en TriageAgentService; la API solo orquesta BackgroundTasks.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Request
from pydantic import BaseModel

from agent_triage.agent_service import TriageAgentService
from agent_triage.constants import settings

logger = logging.getLogger(__name__)


class TriageRequest(BaseModel):
    """
    Payload de entrada del pipeline Tekton (consume-agent-task).

    Acepta trivy_json y opengrep_sarif como lista (formato pipeline) o dict
    (formato raw de Trivy/SARIF).
    """

    repo_path: str
    pull_request_number: str
    repo_owner: str
    commit_id: str
    trivy_json: list[dict[str, Any]] | dict[str, Any]
    opengrep_sarif: list[dict[str, Any]] | dict[str, Any]


def configure_logging() -> None:
    """Configura logging estructurado una sola vez al arrancar la aplicación."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )


def create_app() -> FastAPI:
    """
    Factory de la aplicación FastAPI.

    Permite reutilizar la app en tests y en el entry point sin efectos colaterales
    al importar el módulo.
    """
    configure_logging()
    app = FastAPI(title="Triage Analysis Agent Platform")
    agent_service = TriageAgentService()

    async def _process_triage_safe(data: TriageRequest) -> None:
        try:
            await agent_service.process_triage(data)
        except Exception:
            logger.exception(
                "Triage falló para repo=%s pr=%s",
                data.repo_path,
                data.pull_request_number,
            )

    async def _handle_slack_safe(event: dict) -> None:
        try:
            await agent_service.handle_slack_mention(event)
        except Exception:
            logger.exception("Slack mention falló para evento=%s", event.get("ts"))

    @app.post("/api/v1/triage")
    async def trigger_triage(payload: TriageRequest, background_tasks: BackgroundTasks):
        """
        Acepta hallazgos de escaneo y encola el análisis del agente en background.

        Responde 202 Accepted inmediatamente para no bloquear el pipeline Tekton.
        """
        background_tasks.add_task(_process_triage_safe, payload)
        return {"status": "accepted"}

    @app.post("/api/v1/slack/events")
    async def slack_events(request: Request, background_tasks: BackgroundTasks):
        """
        Webhook de Slack Events API.

        - url_verification: responde el challenge para validar el endpoint.
        - app_mention: encola respuesta del agente en el hilo correspondiente.
        """
        payload = await request.json()
        if payload.get("type") == "url_verification":
            return {"challenge": payload.get("challenge")}
        if (
            payload.get("type") == "event_callback"
            and payload.get("event", {}).get("type") == "app_mention"
        ):
            background_tasks.add_task(_handle_slack_safe, payload.get("event"))
        return {"status": "ok"}

    return app


# Instancia por defecto para uvicorn y compatibilidad con imports directos.
app = create_app()
