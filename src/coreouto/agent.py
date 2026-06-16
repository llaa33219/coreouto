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

FINISH_TOOL_NAME = "finish"
_FINISH_REMINDER = (
    "Reminder: when you are done, call the `finish` tool with your final answer in the "
    "`content` argument. Example: finish(content='Paris is the capital of France.')."
)
_CONTENT_BLOCK_TYPES = (TextBlock, ImageBlock, DocumentBlock, VideoBlock, AudioBlock)


_FINISH_TOOL = Tool(
    name=FINISH_TOOL_NAME,
    description=(
        "Signal that the task is complete and return the final answer to the user. "
        "Pass the user-facing answer in the `content` argument."
    ),
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": ("The final user-facing answer. Omit for an empty response."),
            },
        },
        "required": [],
    },
    handler=lambda content="": content,
    parallelizable=True,
)


_DEFAULT_SYSTEM_PROMPT = (
    "You are an agent. You can use tools to gather information. When the task is fully "
    "complete and you have a self-contained answer for the user, signal completion by "
    "calling the `finish` tool with your final answer in the `content` argument.\n\n"
    "Rules:\n"
    "  - Call `finish` only when the work is fully complete and you are returning the "
    "final answer to the user. Once you call `finish`, the loop ends and your answer is "
    "delivered.\n"
    "  - Do NOT call `finish` if you still intend to call more tools, refine your "
    "answer, or continue working. Do NOT call it for intermediate summaries, partial "
    "progress, or thinking aloud.\n"
    "  - If you respond with text but no `finish` tool call (and no other tool calls), "
    "the loop will inject a reminder asking you to call `finish`.\n"
    "  - If you call `finish` together with other tool calls in the same turn, the other "
    "tools will still execute; the `finish` content will only be returned on a clean "
    "subsequent turn.\n\n"
    "Example:\n"
    "User: What is the capital of France?\n"
    "You: I'll answer from general knowledge. <finish tool call with content='Paris is "
    "the capital of France.'>\n"
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

        effective_tools: list[Tool] = [*resolved_tools, _FINISH_TOOL]

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
                    f"max_iterations ({cfg.max_iterations}) reached without a "
                    f"`{FINISH_TOOL_NAME}` tool call"
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
            finish_call: ToolCall | None = next(
                (tc for tc in tool_calls if tc.name == FINISH_TOOL_NAME),
                None,
            )
            other_calls: list[ToolCall] = [tc for tc in tool_calls if tc.name != FINISH_TOOL_NAME]

            if finish_call is not None and not other_calls:
                content_value = finish_call.arguments.get("content", "")
                if content_value is None:
                    content_value = ""
                final_answer = str(content_value)

                await trigger(
                    ON_FINISH,
                    content=final_answer,
                    messages=messages,
                    iterations=iterations,
                    tool_call_id=finish_call.id,
                )
                return Response(
                    content=final_answer,
                    messages=messages,
                    iterations=iterations,
                    usage=all_usage,
                    finish_called=True,
                )

            if not tool_calls:
                messages.append(Message(role="user", content=_FINISH_REMINDER))
                continue

            resolved = [(tc, get_tool(tc.name)) for tc in other_calls]
            all_parallelizable = cfg.parallel_tool_calls and all(
                tool is not None and tool.parallelizable for _, tool in resolved
            )

            if all_parallelizable and len(other_calls) > 1:
                results = await asyncio.gather(
                    *(_run_one_tool_call(tc, tool) for tc, tool in resolved)
                )
            else:
                results = []
                for tc, tool in resolved:
                    results.append(await _run_one_tool_call(tc, tool))

            for tool_call, result in zip(other_calls, results, strict=False):
                tool_msg = provider.format_tool_result(tool_call, result)
                messages.append(tool_msg)
