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


_CONTINUE_LOOP_TOOL = Tool(
    name=CONTINUE_LOOP_TOOL_NAME,
    description=(
        "Output a text-only turn to the user without ending the agent loop. Use "
        "this when you want to communicate something to the user but intend to call "
        "more tools afterward. The text in the `content` argument is delivered to "
        "the user but the loop continues."
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
    "You are an agent. You can use tools to gather information.\n\n"
    "Termination model:\n"
    "  - The loop ends when you produce a turn with no tool calls. The text in "
    "such a turn is returned to the user as the final answer.\n"
    "  - If you want to output text to the user but keep working (e.g. share "
    "progress before calling more tools), call the `continue_loop` tool with "
    "`content`.\n\n"
    "Rules:\n"
    "  - do NOT call `continue_loop` if you intend to end the loop. Just respond "
    "with text and no tool call when you're done.\n"
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

            if not tool_calls:
                # No tool call = loop terminates. Use the model's text as the final answer.
                final_answer = response.content or ""

                await trigger(
                    ON_FINISH,
                    content=final_answer,
                    messages=messages,
                    iterations=iterations,
                )
                stop_reason: StopReason = "finish"
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
