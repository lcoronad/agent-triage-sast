"""
Lógica del agente LangGraph para triage DevSecOps.

Contiene:
- SingleToolCallMiddleware: adapta respuestas del LLM a una tool-call por turno.
- TriageAgentService: construcción del grafo, ejecución con streaming y fallback
  manual de tools cuando LangGraph no reconoce AIMessage con tool_calls coerced.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_openai import ChatOpenAI
from langchain_redis import RedisChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agent_triage.constants import settings
from agent_triage.helpers import (
    coerce_text_tool_calls,
    final_ai_message,
    has_unresolved_tool_calls,
    log_message,
    message_content,
    normalize_github_tool_args,
    system_prompt_with_rules,
    truncate_text,
    window_messages,
)
from agent_triage.tools import tool_registry

logger = logging.getLogger(__name__)


class SingleToolCallMiddleware(AgentMiddleware):
    """
    Middleware de compatibilidad con vLLM / Red Hat AI Llama.

    Responsabilidades:
    1. Desactivar parallel_tool_calls en el binding del modelo.
    2. Convertir tool-calls en texto a invocaciones estructuradas.
    3. Truncar a una sola tool-call si el modelo devuelve varias.
    4. Normalizar argumentos owner/repo de tools GitHub.
    """

    def __init__(self, github_owner_fallback: str = "") -> None:
        self.github_owner_fallback = github_owner_fallback

    def _apply_single_tool_call_setting(self, request) -> None:
        if settings.parallel_tool_calls:
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
        coerced_ai = coerce_text_tool_calls(messages, self.github_owner_fallback)

        if not messages:
            return {"messages": [coerced_ai]} if coerced_ai is not None else None

        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            for tool_call in last.tool_calls:
                normalize_github_tool_args(tool_call, self.github_owner_fallback)
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


class TriageAgentService:
    """
    Servicio de orquestación del agente de triage.

    Gestiona el ciclo completo: construcción del grafo LangGraph, ejecución
    iterativa con resolución de tool-calls y persistencia de historial Slack.
    """

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            base_url=settings.qwen_api_url,
            api_key=settings.qwen_api_key,
            model=settings.qwen_model_name,
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_completion_tokens=settings.max_completion_tokens,
        )

    async def build_agent(self, system_prompt: str, github_owner_fallback: str = ""):
        """Crea el grafo LangGraph con tools y middleware de compatibilidad."""
        tools = await tool_registry.get_all_tools()
        return create_agent(
            model=self._llm,
            tools=tools,
            system_prompt=system_prompt_with_rules(system_prompt),
            middleware=[SingleToolCallMiddleware(github_owner_fallback=github_owner_fallback)],
        )

    def _redis_history(self, session_id: str) -> RedisChatMessageHistory:
        return RedisChatMessageHistory(
            redis_url=settings.redis_url,
            session_id=session_id,
            ttl=604800,
        )

    def _trim_redis_history(self, history: RedisChatMessageHistory) -> None:
        messages = history.messages
        windowed = window_messages(messages)
        if len(windowed) >= len(messages):
            return
        history.clear()
        for message in windowed:
            history.add_message(message)

    async def _execute_tool_call(self, tool_call: dict[str, Any], tools: list) -> ToolMessage:
        """Ejecuta una tool manualmente cuando el ToolNode de LangGraph falla."""
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

        return ToolMessage(content=str(content), tool_call_id=tool_call_id, name=tool_name)

    async def _manual_run_pending_tools(
        self,
        messages: list[BaseMessage],
        tools: list,
        github_owner_fallback: str = "",
    ) -> list[BaseMessage] | None:
        """Fallback: ejecuta tool_calls pendientes fuera del grafo LangGraph."""
        patched = list(messages)
        coerced_ai = coerce_text_tool_calls(patched, github_owner_fallback)
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
            patched.append(await self._execute_tool_call(tool_call, tools))
        return patched

    async def run(
        self,
        agent,
        prompt: str,
        prior_messages: Sequence[BaseMessage] | None = None,
        *,
        context: str = "agent",
        github_owner_fallback: str = "",
        tools: list | None = None,
    ) -> dict:
        """
        Ejecuta el agente hasta completar tool-calls o alcanzar el límite de recursión.

        Flujo:
        1. Envía prompt + historial al grafo.
        2. Durante el stream, coerciona tool-calls en texto si aparecen.
        3. Si ToolNode falla con "No AIMessage found", ejecuta tools manualmente.
        4. Continúa mientras haya tool_calls sin respuesta.
        """
        agent_tools = tools or await tool_registry.get_all_tools()
        messages = list(prior_messages or []) + [HumanMessage(content=prompt)]
        agent_config = {"recursion_limit": settings.graph_recursion_limit}
        logger.info("[%s] Iniciando ejecución del agente", context)

        async def drain(input_messages: list[BaseMessage]) -> dict:
            result: dict | None = None
            seen_messages = 0

            try:
                async for state in agent.astream(
                    {"messages": input_messages},
                    config=agent_config,
                    stream_mode="values",
                ):
                    current_messages = state.get("messages", [])
                    if coerce_text_tool_calls(current_messages, github_owner_fallback):
                        logger.warning(
                            "[%s] Tool-call en texto reparada durante stream",
                            context,
                        )
                    for message in current_messages[seen_messages:]:
                        log_message(message, context)
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
                base_messages = (
                    list(result.get("messages", [])) if result else list(input_messages)
                )
                patched_messages = await self._manual_run_pending_tools(
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
                    patched_messages = await self._manual_run_pending_tools(
                        list(input_messages),
                        agent_tools,
                        github_owner_fallback,
                    )
                    if patched_messages is None:
                        raise
                    return {"messages": patched_messages}

                current_messages = result.get("messages", [])
                if coerce_text_tool_calls(current_messages, github_owner_fallback):
                    logger.warning("[%s] Tool-call en texto reparada tras ainvoke", context)
                for message in current_messages[seen_messages:]:
                    log_message(message, context)

            return result

        result = await drain(messages)
        continuation_step = 0

        while (
            has_unresolved_tool_calls(result.get("messages", []))
            and continuation_step < settings.graph_recursion_limit
        ):
            continuation_step += 1
            current_messages = list(result.get("messages", []))
            if coerce_text_tool_calls(current_messages, github_owner_fallback):
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
            result = await drain(current_messages)

        final_ai = final_ai_message(result.get("messages", []))
        if final_ai is not None:
            logger.info(
                "[%s] Respuesta final del agente: %s",
                context,
                truncate_text(message_content(final_ai.content)),
            )

        logger.info(
            "[%s] Ejecución finalizada. Total mensajes=%s",
            context,
            len(result.get("messages", [])),
        )
        return result

    def persist_slack_turn(
        self,
        session_id: str,
        prompt: str,
        result_messages: Sequence[BaseMessage],
    ) -> None:
        """Guarda el turno de Slack en Redis con ventana deslizante."""
        history = self._redis_history(session_id)
        history.add_message(HumanMessage(content=prompt))
        final_ai = final_ai_message(result_messages)
        if final_ai is not None:
            history.add_message(final_ai)
        self._trim_redis_history(history)

    async def process_triage(self, data: Any) -> None:
        """
        Flujo principal de triage para un PR.

        Recibe un TriageRequest (o objeto compatible) con hallazgos Trivy/OpenGrep.
        """
        from agent_triage.helpers import (
            build_triage_prompt,
            extract_opengrep_findings,
            extract_trivy_findings,
            split_repo_path,
        )

        trivy_count = len(extract_trivy_findings(data.trivy_json))
        opengrep_count = len(extract_opengrep_findings(data.opengrep_sarif))
        logger.info(
            "Iniciando triage repo=%s pr=%s | hallazgos trivy=%s opengrep=%s",
            data.repo_path,
            data.pull_request_number,
            trivy_count,
            opengrep_count,
        )

        owner, _ = split_repo_path(data.repo_path, data.repo_owner)
        tools = await tool_registry.get_all_tools()
        agent = await self.build_agent(settings.triage_system_prompt, github_owner_fallback=owner)
        prompt = build_triage_prompt(data)

        await self.run(
            agent,
            prompt,
            context=f"triage:{data.repo_path}#PR{data.pull_request_number}",
            github_owner_fallback=owner,
            tools=tools,
        )

    async def handle_slack_mention(self, event: dict) -> None:
        """Procesa menciones @bot en Slack con memoria de hilo en Redis."""
        from agent_triage.helpers import build_slack_prompt

        channel = event.get("channel")
        thread_ts = event.get("thread_ts", event.get("ts"))
        session_id = f"slack:{thread_ts}"

        history = self._redis_history(session_id)
        prior_messages = window_messages(history.messages)

        agent = await self.build_agent(settings.slack_system_prompt)
        prompt = build_slack_prompt(event, channel, thread_ts)
        result = await self.run(
            agent,
            prompt,
            prior_messages=prior_messages,
            context=f"slack:{session_id}",
        )
        self.persist_slack_turn(session_id, prompt, result["messages"])
