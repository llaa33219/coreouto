from __future__ import annotations

import asyncio
import inspect
from typing import Any

from coreouto._types import (
    AgentConfig,
    AudioBlock,
    DocumentBlock,
    ErrorRule,
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
    ON_PROVIDER_ERROR,
    ON_STREAM_TEXT,
    ON_STREAM_THINKING,
    ON_THINKING,
    ON_USER_INJECTION,
    trigger,
)
from coreouto.providers import get_provider
from coreouto.settings import normalize_provider_config
from coreouto.tools import Tool, get_tool

_CONTENT_BLOCK_TYPES = (TextBlock, ImageBlock, DocumentBlock, VideoBlock, AudioBlock)

# Provider-specific stop_reason / finish_reason / status values that mean
# "the provider terminated unrecoverably". These end the loop immediately,
# even if the response contains tool calls — the provider refused,
# truncated, filtered, or cancelled the response, so coreouto must NOT
# execute any tool calls it may carry.
_ANTHROPIC_TERMINATE_REASONS = frozenset({"max_tokens", "refusal"})
_OPENAI_CHAT_TERMINATE_REASONS = frozenset({"length", "content_filter"})
_OPENAI_RESPONSES_TERMINATE_STATUSES = frozenset(
    {"failed", "cancelled", "incomplete:content_filter"}
)
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


def _is_unrecoverable_termination(provider: str, stop_reason: str | None) -> bool:
    """True iff the provider's stop field carries a value that forces
    immediate loop termination.

    Unrecoverable terminations end the loop immediately. Tool calls in the
    same response are NOT executed (the provider flagged the response as
    unsafe / rejected / truncated).
    """
    if provider == "anthropic":
        return stop_reason in _ANTHROPIC_TERMINATE_REASONS
    if provider == "openai":
        return stop_reason in _OPENAI_CHAT_TERMINATE_REASONS
    if provider == "openai-response":
        return stop_reason in _OPENAI_RESPONSES_TERMINATE_STATUSES
    if provider == "google":
        return stop_reason in _GOOGLE_TERMINATE_REASONS
    return False


def _classify_finish(provider: str, stop_reason: str | None) -> StopReason:
    """Map a raw provider stop_reason to the Response.stop_reason literal.

    `"finish"` is the generic clean-termination value. The other literals
    surface non-clean terminations so callers can distinguish a refusal
    from a length cap from a successful finish.
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


_DEFAULT_SYSTEM_PROMPT = (
    "You are an agent. Use tools to gather information and take actions.\n\n"
    "The loop runs as long as you call tools. To deliver your final answer and "
    "end the loop, respond with text and no tool calls — that text becomes the "
    "response. To share progress while still working, include the text alongside "
    "a tool call in the same turn.\n"
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


def _match_error_rule(exc: Exception, rules: list[ErrorRule]) -> ErrorRule | None:
    """Find the first rule whose set fields all match the exception."""
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(exc, "code", None)
    content = str(exc)
    for rule in rules:
        if rule.status_code is not None and status_code != rule.status_code:
            continue
        if rule.content_contains is not None and rule.content_contains not in content:
            continue
        return rule
    return None


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

        effective_tools: list[Tool] = list(resolved_tools)

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

        async def _on_stream_text(text: str) -> None:
            await trigger(ON_STREAM_TEXT, text=text, messages=messages, model=cfg.model)

        async def _on_stream_thinking(text: str) -> None:
            await trigger(ON_STREAM_THINKING, text=text, messages=messages, model=cfg.model)

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
            extra_kwargs = {"system_prompt": None, **merged}
            if hasattr(provider, "_stream"):
                extra_kwargs["_on_stream_text"] = _on_stream_text
                extra_kwargs["_on_stream_thinking"] = _on_stream_thinking

            try:
                response = await provider.create(
                    messages=messages,
                    model=cfg.model,
                    tools=effective_tools,
                    **extra_kwargs,
                )
            except Exception as exc:
                error_handling = getattr(provider, "error_handling", None)
                if not error_handling:
                    raise
                rule = _match_error_rule(exc, error_handling)
                if rule is None:
                    raise

                status_code = getattr(exc, "status_code", None)
                if status_code is None:
                    status_code = getattr(exc, "code", None)
                error_message = getattr(exc, "message", None) or str(exc)

                if rule.reaction == "retry":
                    delay = rule.retry_after
                    for _ in range(rule.retry_max):
                        await trigger(
                            ON_PROVIDER_ERROR,
                            error=exc,
                            status_code=status_code,
                            error_message=error_message,
                            reaction="retry",
                            reaction_message=rule.message,
                            messages=messages,
                            model=cfg.model,
                        )
                        await asyncio.sleep(delay)
                        delay *= rule.retry_backoff
                        try:
                            response = await provider.create(
                                messages=messages,
                                model=cfg.model,
                                tools=effective_tools,
                                **extra_kwargs,
                            )
                            break
                        except Exception as retry_exc:
                            exc = retry_exc
                            if _match_error_rule(retry_exc, error_handling) is not rule:
                                raise
                            status_code = getattr(exc, "status_code", None)
                            if status_code is None:
                                status_code = getattr(exc, "code", None)
                            error_message = getattr(exc, "message", None) or str(exc)
                    else:
                        raise exc
                else:
                    await trigger(
                        ON_PROVIDER_ERROR,
                        error=exc,
                        status_code=status_code,
                        error_message=error_message,
                        reaction=rule.reaction,
                        reaction_message=rule.message,
                        messages=messages,
                        model=cfg.model,
                    )
                    if rule.reaction == "terminate":
                        await trigger(
                            ON_FINISH,
                            content=rule.message,
                            messages=messages,
                            iterations=iterations,
                        )
                        return Response(
                            content=rule.message,
                            messages=messages,
                            iterations=iterations,
                            usage=all_usage,
                            stop_reason="failed",
                        )
                    if rule.reaction == "user_message":
                        self._pending_user_messages.put_nowait(rule.message)
                        continue
                    if rule.reaction == "tool_result":
                        for msg in reversed(messages):
                            if msg.role == "assistant" and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    messages.append(
                                        provider.format_tool_result(
                                            tc,
                                            ToolResult(
                                                tool_call_id=tc.id,
                                                content=rule.message,
                                                is_error=True,
                                            ),
                                        )
                                    )
                                break
                        continue

            await trigger(AFTER_LLM_CALL, response=response, messages=messages)
            if response.thinking:
                await trigger(
                    ON_THINKING,
                    thinking=response.thinking,
                    messages=messages,
                    model=cfg.model,
                )
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
            is_unrecoverable = _is_unrecoverable_termination(cfg.provider, response.stop_reason)

            if is_unrecoverable:
                # Unrecoverable provider termination (max_tokens, refusal,
                # content_filter, SAFETY, failed, cancelled, ...). The provider
                # refused/truncated the response, so do NOT execute any tool
                # calls it may contain. Terminate and surface the provider
                # signal on Response.stop_reason.
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

            if not tool_calls:
                if response.content:
                    # Content (text) + no tool calls — the model is done. The
                    # assistant text is the final answer. This is the canonical
                    # end-of-loop signal under coreouto's termination policy.
                    final_answer = response.content

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
                # No content and no tool calls (e.g. a thinking-only turn, or
                # an otherwise empty response). The model produced neither an
                # answer nor an action, so treat it as "still working": continue
                # the loop (re-prompt) rather than terminating with an empty
                # answer. The loop is still bounded by max_iterations.
                continue

            # Tool calls present on a recoverable turn — execute them and
            # continue the loop. The model is mid-task.
            resolved: list[tuple[ToolCall, Tool | None]] = []
            for tc in tool_calls:
                user_tool = get_tool(tc.name)
                resolved.append((tc, user_tool))
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
            # loop continues to next iteration
