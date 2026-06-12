from __future__ import annotations

import asyncio
import inspect
import re
from typing import Any

from coreouto._types import (
    AgentConfig,
    AudioBlock,
    DocumentBlock,
    ImageBlock,
    Message,
    Response,
    TextBlock,
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
from coreouto.tools import get_tool

_FINISH_RE = re.compile(r"<finish>(.*?)</finish>", re.DOTALL)
_FINISH_REMINDER = (
    "Reminder: wrap your final answer in <finish>...</finish> tags so I can return it."
)
_CONTENT_BLOCK_TYPES = (TextBlock, ImageBlock, DocumentBlock, VideoBlock, AudioBlock)


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


_CONTENT_BLOCK_TYPES = (TextBlock, ImageBlock, DocumentBlock, VideoBlock, AudioBlock)


_DEFAULT_SYSTEM_PROMPT = (
    "You are an agent. Use tools to gather information, then return your final answer to the user.\n\n"
    "CRITICAL: When you are done, your final user-facing answer MUST be wrapped in <finish>...</finish> tags. "
    "The text inside the tags is what the user will see.\n\n"
    "Example:\n"
    "The capital of France is Paris.\n"
    "<finish>Paris is the capital of France.</finish>\n\n"
    "If you respond with text but no <finish> tags, the loop will continue and you'll be asked to retry."
)


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

        resolved_tools: list[Any] = []
        for name in cfg.tools:
            tool = get_tool(name)
            if tool is None:
                raise KeyError(f"tool not registered: {name!r}")
            resolved_tools.append(tool)

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
            if iterations > cfg.max_iterations:
                raise MaxIterationsError(
                    f"max_iterations ({cfg.max_iterations}) reached without a <finish> tag"
                )

            await trigger(
                BEFORE_LLM_CALL,
                messages=messages,
                model=cfg.model,
                tools=resolved_tools,
            )

            normalized = normalize_provider_config(cfg.provider, cfg.provider_config)
            merged = {**normalized, **cfg.provider_passthrough}
            response = await provider.create(
                messages=messages,
                model=cfg.model,
                tools=resolved_tools,
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

            last_assistant_text = response.content or ""
            match = _FINISH_RE.search(last_assistant_text)
            if match:
                final_answer = match.group(1).strip()
                await trigger(
                    ON_FINISH,
                    content=final_answer,
                    raw_content=last_assistant_text,
                    messages=messages,
                    iterations=iterations,
                )
                return Response(
                    content=final_answer,
                    messages=messages,
                    iterations=iterations,
                    usage=all_usage,
                    finish_called=True,
                )

            if not response.tool_calls:
                messages.append(Message(role="user", content=_FINISH_REMINDER))
                continue

            for tool_call in response.tool_calls:
                tool = get_tool(tool_call.name)

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
                        raw_result = tool.handler(**tool_call.arguments)
                        if inspect.iscoroutine(raw_result):
                            raw_result = await raw_result
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

                tool_msg = provider.format_tool_result(tool_call, result)
                messages.append(tool_msg)
