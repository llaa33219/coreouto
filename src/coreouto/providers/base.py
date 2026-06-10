from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from coreouto._types import LLMResponse, Message, ToolCall, ToolResult
from coreouto.tools import Tool


@runtime_checkable
class Provider(Protocol):
    async def create(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    def format_assistant_message(self, response: LLMResponse) -> Message: ...

    def format_tool_result(self, tool_call: ToolCall, result: ToolResult) -> Message: ...
