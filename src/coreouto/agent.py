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
FINISH_TOOL_NAME = "finish"
_CONTENT_BLOCK_TYPES = (TextBlock, ImageBlock, DocumentBlock, VideoBlock, AudioBlock)

# Provider-specific sets of stop_reason / finish_reason / status values that mean
# "the loop should end now, regardless of whether the model called `finish`".
# These are unrecoverable terminations — the provider refused, the run was
# cancelled, the response was filtered for content, the output hit a hard
# limit. coreouto surfaces them on `Response.stop_reason` so callers can
# distinguish a non-clean termination from a clean `finish`.
#
# A clean natural-end signal from the provider (`end_turn`, `stop`,
# `completed`, `STOP`, ...) is NOT an unconditional END under the new
# policy — the model must explicitly call the `finish` tool to close the
# loop. Without `finish`, the loop re-prompts the model with the empty
# content, giving it a chance to call more tools or `finish`. This is
# the "explicitness" guarantee: the model's intent to end the loop is
# declared through a tool call, not inferred from a provider's natural
# end-of-turn field.
_ANTHROPIC_TERMINATE_REASONS = frozenset({"max_tokens", "refusal"})
# `pause_turn` is a continuation signal: the server-side sampling loop hit
# its iteration limit (default 10) while running server tools like web
# search / web fetch. The official guidance is to send the assistant
# response back as-is in a new request so the loop can continue — see the
# Anthropic docs and SDK issue #1170.
_ANTHROPIC_CONTINUE_REASONS = frozenset({"tool_use", "pause_turn"})

_OPENAI_CHAT_TERMINATE_REASONS = frozenset({"length", "content_filter"})
# `tool_calls` and the legacy `function_call` are explicit "I called a
# tool" signals (the latter being the singular predecessor of the modern
# `tool_calls` field). They are CONTINUE because the loop must execute
# the call(s); whether the loop then ends depends on whether `finish` is
# also in the same response.
_OPENAI_CHAT_CONTINUE_REASONS = frozenset({"tool_calls", "function_call"})

# OpenAI Responses API: these are the truly terminal statuses — server
# rejected, user cancelled, content filter tripped. They are END
# regardless of `finish`. All other statuses (including `completed`,
# `incomplete`, `incomplete:max_output_tokens`) are NOT unconditional
# END — the model must call `finish` to close the loop.
_OPENAI_RESPONSES_TERMINATE_STATUSES = frozenset(
    {
        "failed",
        "cancelled",
        "incomplete:content_filter",
    }
)

# Google Gemini: these are the truly terminal finish_reasons. All other
# values (including `STOP`, `FINISH_REASON_UNSPECIFIED`,
# `UNEXPECTED_TOOL_CALL`, `MALFORMED_FUNCTION_CALL`) are NOT
# unconditional END — the model must call `finish` to close the loop.
_GOOGLE_TERMINATE_REASONS = frozenset(
    {
        "MAX_TOKENS",
        "SAFETY",
        "RECITATION",
        "LANGUAGE",
        "OTHER",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
        "IMAGE_OTHER",
        "NO_IMAGE",
    }
)


def _called_finish(tool_calls: list[ToolCall]) -> bool:
    """True iff the response included a `finish` tool call.

    The `finish` tool is the model's explicit declaration that the loop
    should end and that the supplied `content` is the final answer. Any
    other tool call without `finish` keeps the loop running.
    """
    return any(tc.name == FINISH_TOOL_NAME for tc in tool_calls)


def _should_terminate(provider: str, stop_reason: str | None, tool_calls: list[ToolCall]) -> bool:
    """Classify a provider response to decide whether the agent loop ends now.

    The rule is **model-driven, not provider-driven**: the loop ends when
    the model explicitly calls the `finish` tool (regardless of the
    provider's stop field) OR when the provider's stop field carries an
    unrecoverable termination value that cannot be papered over by
    re-prompting (`max_tokens`, `refusal`, `length`, `content_filter`,
    `SAFETY`, `failed`, `cancelled`, `incomplete:content_filter`).

    Natural end-of-turn values from the provider (`end_turn`, `stop`,
    `completed`, `STOP`, `pause_turn`, `tool_calls`, ...) are NOT
    unconditional END anymore — they are CONTINUE unless the model also
    called `finish`. This implements the "explicitness" guarantee: the
    model declares its intent to end the loop through a tool call,
    rather than the agent inferring it from a provider signal that the
    model didn't author.

    Per-provider detail:

    - **Anthropic** (`stop_reason`): unrecoverable values
      (`max_tokens`, `refusal`) are END. `end_turn`, `stop_sequence`,
      `tool_use`, `pause_turn` are all CONTINUE — only `finish` ends the
      loop. Unknown values are CONTINUE.
    - **OpenAI Chat Completions** (`finish_reason`): unrecoverable
      values (`length`, `content_filter`) are END. `stop`, `tool_calls`,
      legacy `function_call` are CONTINUE — only `finish` ends.
    - **OpenAI Responses API** (`status`): unrecoverable values
      (`failed`, `cancelled`, `incomplete:content_filter`) are END.
      `completed`, `incomplete`, `incomplete:max_output_tokens` are
      CONTINUE — only `finish` ends.
    - **Google Gemini** (`finish_reason`): unrecoverable values
      (`MAX_TOKENS`, `SAFETY`, `RECITATION`, ...) are END. `STOP`,
      `FINISH_REASON_UNSPECIFIED`, `UNEXPECTED_TOOL_CALL`,
      `MALFORMED_FUNCTION_CALL` are CONTINUE — only `finish` ends.
    - **Unknown providers**: only the explicit `finish` tool call
      ends the loop. Without it, the loop re-prompts the model.
    """
    if _called_finish(tool_calls):
        # Explicit termination — the model called the `finish` tool.
        # This is the canonical end-of-loop signal and overrides the
        # provider's stop_reason (including the unrecoverable values,
        # because the model may have called `finish` and then hit a
        # token cap simultaneously).
        return True

    if provider == "anthropic":
        return stop_reason in _ANTHROPIC_TERMINATE_REASONS

    if provider == "openai":
        return stop_reason in _OPENAI_CHAT_TERMINATE_REASONS

    if provider == "openai-response":
        return stop_reason in _OPENAI_RESPONSES_TERMINATE_STATUSES

    if provider == "google":
        return stop_reason in _GOOGLE_TERMINATE_REASONS

    # Unknown provider: only the `finish` tool call ends the loop.
    return False


def _classify_finish(provider: str, stop_reason: str | None) -> StopReason:
    """Map a raw provider stop_reason to the Response.stop_reason literal.

    `"finish"` is the generic "the model finished naturally" value — but
    under the new model-driven policy, the only way to reach this
    function is via the model's `finish` tool call (the
    non-recoverable terminations still go through this classifier when
    no `finish` was involved, and they surface their own literals).
    The other literals surface non-clean terminations so callers can
    distinguish a refusal from a length cap from a successful finish.
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

    if provider == "google":
        if stop_reason == "MAX_TOKENS":
            return "max_tokens"
        return "finish"

    # Unknown providers — fall back to the generic "finish" literal.
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


_FINISH_TOOL = Tool(
    name=FINISH_TOOL_NAME,
    description=(
        "End the agent loop and return `content` to the caller as the final "
        "answer. Call this when you are done. After calling `finish` the loop "
        "stops — this is the only way to close the loop under coreouto's "
        "model-driven termination policy (a text-only turn without `finish` "
        "will be re-prompted)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The final answer to return to the user.",
            },
        },
        "required": [],
    },
    handler=lambda content="": content,
    parallelizable=True,
)


def _extract_finish_content(tool_calls: list[ToolCall]) -> str:
    """Return the `content` argument of the first `finish` tool call in the
    response. The model may emit `finish` alongside other tool calls in a
    single turn; we take the first `finish` call's content as the final
    answer.
    """
    for tc in tool_calls:
        if tc.name == FINISH_TOOL_NAME:
            value = tc.arguments.get("content", "")
            return value if isinstance(value, str) else str(value)
    return ""


_DEFAULT_SYSTEM_PROMPT = (
    "You are an agent. Use tools to gather information as needed.\n\n"
    "The `finish` tool ends the agent loop and returns its `content` argument "
    "to the caller as the final answer. Call `finish` when you are done. A "
    "text-only turn without `finish` will be re-prompted, so you must call "
    "`finish` to actually close the loop.\n\n"
    "The `continue_loop` tool sends text to the user mid-task but keeps the "
    "loop running for more tools. Use it for progress updates before calling "
    "more tools or before calling `finish`.\n"
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

        effective_tools: list[Tool] = [*resolved_tools, _CONTINUE_LOOP_TOOL, _FINISH_TOOL]

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

            if _should_terminate(cfg.provider, response.stop_reason, tool_calls):
                # Final-answer extraction: the model's `finish` tool call's
                # `content` argument is the canonical final answer. If the
                # loop ended via an unrecoverable provider termination
                # (e.g. max_tokens) without a `finish` call, fall back to
                # the response text.
                finish_content = _extract_finish_content(tool_calls)
                final_answer = finish_content or (response.content or "")

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
