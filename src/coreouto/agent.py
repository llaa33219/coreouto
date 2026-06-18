from __future__ import annotations

import asyncio
import inspect
from typing import Any

from coreouto._types import (
    AgentConfig,
    AudioBlock,
    DocumentBlock,
    ImageBlock,
    Message,
    Response,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResult,
    Usage,
    VideoBlock,
)
from coreouto.hooks import (
    AFTER_LLM_CALL,
    AFTER_TOOL_CALL,
    BEFORE_LLM_CALL,
    BEFORE_TOOL_CALL,
    ON_FINISH,
    ON_ITERATION,
    ON_USER_INJECTION,
    trigger,
)
from coreouto.providers import get_provider
from coreouto.settings import normalize_provider_config
from coreouto.tools import Tool, get_tool

CONTINUE_LOOP_TOOL_NAME = "continue_loop"
_CONTENT_BLOCK_TYPES = (TextBlock, ImageBlock, DocumentBlock, VideoBlock, AudioBlock)

# Provider-specific sets of stop_reason / finish_reason / status values that mean
# "the loop should end now". For Anthropic and OpenAI Chat Completions, the
# END/CONTINUE split is encoded in the value itself (`tool_use` / `tool_calls`
# are the only CONTINUE values); for Google Gemini and OpenAI Responses, the
# field always carries an END-style value, and the loop must additionally
# consult whether the response contained tool calls.
_ANTHROPIC_TERMINATE_REASONS = frozenset(
    {"end_turn", "max_tokens", "stop_sequence", "pause_turn", "refusal"}
)
_ANTHROPIC_CONTINUE_REASONS = frozenset({"tool_use"})

_OPENAI_CHAT_TERMINATE_REASONS = frozenset({"stop", "length", "content_filter", "function_call"})
_OPENAI_CHAT_CONTINUE_REASONS = frozenset({"tool_calls"})

# OpenAI Responses API: status="completed" is the natural end-of-turn signal,
# and tool calls live in `response.output` items. Other terminal statuses
# (failed / cancelled / incomplete) are also END regardless of tool_calls.
_OPENAI_RESPONSES_TERMINATE_STATUSES = frozenset(
    {
        "failed",
        "cancelled",
        "incomplete",
        "incomplete:max_output_tokens",
        "incomplete:content_filter",
    }
)

# Google Gemini: every documented `finish_reason` is a terminal signal. The
# SDK has no "I called a tool" finish reason — when the model calls a tool,
# `finish_reason="STOP"` and the function call appears in
# `candidate.content.parts`. So for Gemini the loop must additionally consult
# `has_tool_calls` (STOP without tool calls = end; STOP with tool calls =
# continue; any other finish_reason = end regardless of tool_calls).
_GOOGLE_TERMINATE_REASONS = frozenset(
    {
        "STOP",
        "MAX_TOKENS",
        "SAFETY",
        "RECITATION",
        "LANGUAGE",
        "OTHER",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "MALFORMED_FUNCTION_CALL",
        "UNEXPECTED_TOOL_CALL",
        "FINISH_REASON_UNSPECIFIED",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
        "IMAGE_OTHER",
        "NO_IMAGE",
    }
)


def _should_terminate(provider: str, stop_reason: str | None, has_tool_calls: bool) -> bool:
    """Classify a provider response to decide whether the agent loop ends now.

    Rule: "if it's not END, keep going." The END set is defined per-provider
    from each provider's documented end-of-turn vocabulary. For Google Gemini
    and OpenAI Responses the field is always END-like, so `has_tool_calls` is
    consulted as a second condition to disambiguate "natural finish" from
    "model called a tool, loop should continue".

    - **Anthropic** (`stop_reason`): `end_turn`, `max_tokens`, `stop_sequence`,
      `pause_turn`, `refusal` = END. `tool_use` = CONTINUE. Any other value
      (including `None`) = CONTINUE.
    - **OpenAI Chat Completions** (`finish_reason`): `stop`, `length`,
      `content_filter`, `function_call` = END. `tool_calls` = CONTINUE. Any
      other value = CONTINUE.
    - **OpenAI Responses API** (`status`): `failed`, `cancelled`, `incomplete`
      (and the `incomplete:<reason>` variants) = END. `completed` is END
      only when there are no tool calls in the response.
    - **Google Gemini** (`finish_reason`): any explicit non-STOP value
      (`MAX_TOKENS`, `SAFETY`, `RECITATION`, ...) = END. `STOP` (or
      `FINISH_REASON_UNSPECIFIED` / `None`) is END only when there are no
      tool calls in the response.

    Unknown providers fall back to the historical "no tool calls = end" rule
    so that custom providers without documented stop_reason semantics keep
    working.
    """
    if provider == "anthropic":
        # Trust stop_reason: explicit END values end the loop; tool_use keeps
        # the loop going; anything else is treated as CONTINUE so we never
        # silently end the loop on an unrecognized value.
        return stop_reason in _ANTHROPIC_TERMINATE_REASONS

    if provider == "openai":
        return stop_reason in _OPENAI_CHAT_TERMINATE_REASONS

    if provider == "openai-response":
        if stop_reason in _OPENAI_RESPONSES_TERMINATE_STATUSES:
            return True
        # `completed` (or any other non-terminal status) — END only if no tool
        # calls were produced.
        return not has_tool_calls

    if provider == "google":
        if stop_reason in _GOOGLE_TERMINATE_REASONS and stop_reason != "STOP":
            # Explicit non-STOP terminal reason (SAFETY, MAX_TOKENS, ...).
            return True
        # STOP / None / unknown — END only if no tool calls were produced.
        return not has_tool_calls

    # Unknown provider: preserve historical "no tool calls = end" behavior.
    return not has_tool_calls


def _classify_finish(provider: str, stop_reason: str | None) -> StopReason:
    """Map a raw provider stop_reason to the Response.stop_reason literal.

    `"finish"` is the generic "the model finished naturally" value. The other
    literals surface non-clean terminations so callers can distinguish a
    refusal from a length cap from a successful finish.
    """
    if provider == "anthropic":
        if stop_reason == "max_tokens":
            return "max_tokens"
        if stop_reason == "refusal":
            return "refusal"
        return "finish"

    if provider == "openai":
        if stop_reason == "length":
            return "length"
        if stop_reason == "content_filter":
            return "content_filter"
        return "finish"

    if provider == "openai-response":
        if stop_reason == "failed":
            return "failed"
        if stop_reason == "cancelled":
            return "cancelled"
        if stop_reason and stop_reason.startswith("incomplete"):
            return "incomplete"
        return "finish"

    # Google and unknown providers don't surface distinct terminator categories
    # through stop_reason — they all collapse to "finish" or get reflected
    # verbatim in `Response.stop_reason` if a hook needs to see it.
    return "finish"

    if provider == "openai":
        if stop_reason == "length":
            return "length"
        if stop_reason == "content_filter":
            return "content_filter"
        return "finish"

    if provider == "openai-response":
        if stop_reason == "failed":
            return "failed"
        if stop_reason == "cancelled":
            return "cancelled"
        if stop_reason and stop_reason.startswith("incomplete"):
            return "incomplete"
        return "finish"

    # Google and unknown providers don't surface distinct terminator categories
    # through stop_reason — they all collapse to "finish" or get reflected
    # verbatim in `Response.stop_reason` if a hook needs to see it.
    return "finish"


_CONTINUE_LOOP_TOOL = Tool(
    name=CONTINUE_LOOP_TOOL_NAME,
    description=(
        "Send text to the user mid-task without ending the agent loop. Use for "
        "progress updates when you still intend to call more tools. The `content` "
        "argument is shown to the user; the loop continues."
    ),
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The text to deliver to the user this turn.",
            },
        },
        "required": [],
    },
    handler=lambda content="": content,
    parallelizable=True,
)


_DEFAULT_SYSTEM_PROMPT = (
    "You are an agent. Use tools to gather information as needed.\n\n"
    "The `continue_loop` tool sends text to the user mid-task but keeps the loop "
    "running for more tools. Use it for progress updates before calling more tools.\n"
)


async def _run_one_tool_call(tool_call: ToolCall, tool: Any) -> ToolResult:
    """Execute a single tool call, fire the BEFORE/AFTER hooks, return a ToolResult.

    Sync handlers are run via asyncio.to_thread so they don't block the event
    loop when this helper is awaited concurrently from asyncio.gather. Async
    handlers are awaited directly.
    """
    assert isinstance(tool_call, ToolCall)

    await trigger(
        BEFORE_TOOL_CALL,
        name=tool_call.name,
        arguments=tool_call.arguments,
    )

    if tool is None:
        result = ToolResult(
            tool_call_id=tool_call.id,
            content=f"tool not found: {tool_call.name}",
            is_error=True,
        )
    else:
        try:
            if inspect.iscoroutinefunction(tool.handler):
                raw_result = await tool.handler(**tool_call.arguments)
            else:
                raw_result = await asyncio.to_thread(
                    _invoke_sync_handler, tool.handler, tool_call.arguments
                )
            result = _coerce_tool_result(tool_call.id, raw_result)
        except Exception as exc:
            result = ToolResult(
                tool_call_id=tool_call.id,
                content=f"{type(exc).__name__}: {exc}",
                is_error=True,
            )

    await trigger(
        AFTER_TOOL_CALL,
        name=tool_call.name,
        result=result,
    )
    return result


def _invoke_sync_handler(handler: Any, arguments: dict[str, Any]) -> Any:
    return handler(**arguments)


def _coerce_tool_result(tool_call_id: str, raw_result: Any) -> ToolResult:
    """Wrap a tool handler's return value into a ToolResult.

    - A ToolResult is passed through (caller is responsible for the tool_call_id).
    - A `str` becomes a text-only ToolResult.
    - `list[ContentBlock]` becomes a multimodal ToolResult.
    - Anything else is stringified.
    """
    if isinstance(raw_result, ToolResult):
        return raw_result.model_copy(update={"tool_call_id": tool_call_id})
    if isinstance(raw_result, str):
        return ToolResult(tool_call_id=tool_call_id, content=raw_result, is_error=False)
    if isinstance(raw_result, list) and all(
        isinstance(item, _CONTENT_BLOCK_TYPES) for item in raw_result
    ):
        return ToolResult(tool_call_id=tool_call_id, blocks=raw_result, is_error=False)
    return ToolResult(tool_call_id=tool_call_id, content=str(raw_result), is_error=False)


class MaxIterationsError(Exception):
    pass


class Agent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.provider_name = config.provider
        self._pending_user_messages: asyncio.Queue[str] = asyncio.Queue()

    def inject_user_message(self, content: str) -> None:
        """Queue a user message to be inserted on the next loop iteration.

        Thread-safe and async-safe: callable from any thread, from another
        async task, or from a hook callback. The message is inserted into
        the conversation at the start of the next iteration and fires the
        `on_user_injection` hook.
        """
        self._pending_user_messages.put_nowait(content)

    def call_sync(
        self,
        user_message: str,
        *,
        override: AgentConfig | None = None,
        history: list[Message] | None = None,
    ) -> Response:
        from coreouto import sync

        return sync.call_sync(self, user_message, override=override, history=history)

    async def call(
        self,
        user_message: str,
        *,
        override: AgentConfig | None = None,
        history: list[Message] | None = None,
    ) -> Response:
        cfg = override or self.config
        provider = get_provider(cfg.provider)

        resolved_tools: list[Tool] = []
        for name in cfg.tools:
            tool = get_tool(name)
            if tool is None:
                raise KeyError(f"tool not registered: {name!r}")
            resolved_tools.append(tool)

        effective_tools: list[Tool] = [*resolved_tools, _CONTINUE_LOOP_TOOL]

        messages: list[Message] = []
        if cfg.system_prompt:
            messages.append(Message(role="system", content=cfg.system_prompt))
        else:
            messages.append(Message(role="system", content=_DEFAULT_SYSTEM_PROMPT))
        if history:
            messages.extend(history)
        messages.append(Message(role="user", content=user_message))

        iterations = 0
        all_usage: list[Usage] = []

        while True:
            await asyncio.sleep(0)
            while not self._pending_user_messages.empty():
                content = self._pending_user_messages.get_nowait()
                injected = Message(role="user", content=content)
                messages.append(injected)
                await trigger(
                    ON_USER_INJECTION,
                    message=injected,
                    messages=messages,
                )

            iterations += 1
            if cfg.max_iterations is not None and iterations > cfg.max_iterations:
                raise MaxIterationsError(
                    f"max_iterations ({cfg.max_iterations}) reached without "
                    f"terminating the loop (the model kept producing tool calls)"
                )

            await trigger(
                BEFORE_LLM_CALL,
                messages=messages,
                model=cfg.model,
                tools=effective_tools,
            )

            normalized = normalize_provider_config(cfg.provider, cfg.provider_config)
            merged = {**normalized, **cfg.provider_passthrough}
            response = await provider.create(
                messages=messages,
                model=cfg.model,
                tools=effective_tools,
                system_prompt=None,
                **merged,
            )

            await trigger(AFTER_LLM_CALL, response=response, messages=messages)
            if response.usage:
                all_usage.append(response.usage)

            assistant_msg = provider.format_assistant_message(response)
            messages.append(assistant_msg)

            await trigger(
                ON_ITERATION,
                iteration=iterations,
                messages=messages,
                response=response,
            )

            tool_calls = list(response.tool_calls)

            if _should_terminate(cfg.provider, response.stop_reason, bool(tool_calls)):
                final_answer = response.content or ""

                await trigger(
                    ON_FINISH,
                    content=final_answer,
                    messages=messages,
                    iterations=iterations,
                )
                stop_reason: StopReason = _classify_finish(cfg.provider, response.stop_reason)
                return Response(
                    content=final_answer,
                    messages=messages,
                    iterations=iterations,
                    usage=all_usage,
                    stop_reason=stop_reason,
                )

            resolved: list[tuple[ToolCall, Tool | None]] = []
            for tc in tool_calls:
                user_tool = get_tool(tc.name)
                if user_tool is not None:
                    resolved.append((tc, user_tool))
                else:
                    injected_tool = next((t for t in effective_tools if t.name == tc.name), None)
                    resolved.append((tc, injected_tool))
            all_parallelizable = cfg.parallel_tool_calls and all(
                tool is not None and tool.parallelizable for _, tool in resolved
            )

            if all_parallelizable and len(tool_calls) > 1:
                results = await asyncio.gather(
                    *(_run_one_tool_call(tc, tool) for tc, tool in resolved)
                )
            else:
                results = []
                for tc, tool in resolved:
                    results.append(await _run_one_tool_call(tc, tool))

            for tool_call, result in zip(tool_calls, results, strict=False):
                tool_msg = provider.format_tool_result(tool_call, result)
                messages.append(tool_msg)
