#!/usr/bin/env python3
"""
Punto de entrada del servicio de triage DevSecOps.

Mantiene compatibilidad con el despliegue existente (Containerfile CMD).
La lógica de negocio reside en el paquete `agent_triage/`.
"""

from __future__ import annotations

import uvicorn

from agent_triage.api import app
from agent_triage.constants import settings

# Re-exportaciones para compatibilidad con imports legacy.
from agent_triage.api import TriageRequest  # noqa: F401
from agent_triage.agent_service import TriageAgentService  # noqa: F401

__all__ = ["app", "TriageRequest", "TriageAgentService"]


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
