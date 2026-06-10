from __future__ import annotations

from typing import Any

from coreouto._types import LLMResponse, Message, ToolCall, ToolResult
from coreouto.providers.base import Provider
from coreouto.tools import Tool


class FullProvider:
    async def create(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content="ok")

    def format_assistant_message(self, response: LLMResponse) -> Message:
        return Message(role="assistant", content=response.content or "")

    def format_tool_result(self, tool_call: ToolCall, result: ToolResult) -> Message:
        return Message(role="tool", content=str(result.content))


class MissingCreate:
    def format_assistant_message(self, response: LLMResponse) -> Message:
        return Message(role="assistant", content="")

    def format_tool_result(self, tool_call: ToolCall, result: ToolResult) -> Message:
        return Message(role="tool", content="")


class MissingFormatAssistant:
    async def create(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content="ok")

    def format_tool_result(self, tool_call: ToolCall, result: ToolResult) -> Message:
        return Message(role="tool", content="")


class MissingFormatToolResult:
    async def create(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content="ok")

    def format_assistant_message(self, response: LLMResponse) -> Message:
        return Message(role="assistant", content="")


def test_full_provider_satisfies_protocol() -> None:
    provider = FullProvider()
    assert isinstance(provider, Provider)


def test_missing_create_fails_check() -> None:
    assert not isinstance(MissingCreate(), Provider)


def test_missing_format_assistant_fails_check() -> None:
    assert not isinstance(MissingFormatAssistant(), Provider)


def test_missing_format_tool_result_fails_check() -> None:
    assert not isinstance(MissingFormatToolResult(), Provider)


def test_mock_provider_satisfies_protocol() -> None:
    from tests.conftest import MockProvider

    assert isinstance(MockProvider(), Provider)


def test_protocol_has_exactly_three_methods() -> None:
    protocol_methods = {name for name in dir(Provider) if not name.startswith("_")}
    assert protocol_methods == {"create", "format_assistant_message", "format_tool_result"}
