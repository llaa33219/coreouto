from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from coreouto._types import LLMResponse, Message, ToolCall, ToolResult
from coreouto.providers.openai_response import OpenAIResponseProvider
from coreouto.tools import Tool


@dataclass
class FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class FakeResponseItem:
    type: str
    role: str | None = None
    content: list[dict[str, Any]] = field(default_factory=list)
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    text: str | None = None


@dataclass
class FakeResponse:
    output: list[FakeResponseItem]
    usage: FakeUsage


class FakeResponsesClient:
    def __init__(self, responses: list[FakeResponse] | None = None) -> None:
        self._responses = list(responses or [])
        self._index = 0
        self.calls: list[dict[str, Any]] = []

    async def create(
        self,
        *,
        model: str,
        instructions: str | None = None,
        input: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> FakeResponse:
        self.calls.append(
            {
                "model": model,
                "instructions": instructions,
                "input": input,
                "tools": tools,
                "kwargs": kwargs,
            }
        )
        if self._index >= len(self._responses):
            raise AssertionError("FakeResponsesClient exhausted")
        resp = self._responses[self._index]
        self._index += 1
        return resp


class FakeAsyncOpenAI:
    def __init__(self) -> None:
        self.responses = FakeResponsesClient()


@pytest.fixture
def fake_client() -> FakeAsyncOpenAI:
    return FakeAsyncOpenAI()


@pytest.fixture
def provider(fake_client: FakeAsyncOpenAI) -> OpenAIResponseProvider:
    return OpenAIResponseProvider(client=fake_client)


async def test_create_simple_user_message(
    provider: OpenAIResponseProvider, fake_client: FakeAsyncOpenAI
) -> None:
    fake_client.responses = FakeResponsesClient(
        responses=[
            FakeResponse(
                output=[
                    FakeResponseItem(
                        type="message",
                        role="assistant",
                        content=[{"type": "output_text", "text": "Hello!"}],
                    )
                ],
                usage=FakeUsage(input_tokens=5, output_tokens=2),
            )
        ]
    )

    messages = [Message(role="user", content="Say hello")]
    result = await provider.create(messages=messages, model="gpt-4o")

    assert result.content == "Hello!"
    assert result.tool_calls == []
    assert result.usage is not None
    assert result.usage.prompt_tokens == 5
    assert result.usage.completion_tokens == 2
    assert result.usage.total_tokens == 7
    assert result.raw is not None

    call = fake_client.responses.calls[0]
    assert call["model"] == "gpt-4o"
    assert call["instructions"] is None
    assert call["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Say hello"}],
        }
    ]


async def test_create_with_system_prompt(
    provider: OpenAIResponseProvider, fake_client: FakeAsyncOpenAI
) -> None:
    fake_client.responses = FakeResponsesClient(
        responses=[
            FakeResponse(
                output=[
                    FakeResponseItem(
                        type="message",
                        role="assistant",
                        content=[{"type": "output_text", "text": "OK"}],
                    )
                ],
                usage=FakeUsage(input_tokens=3, output_tokens=1),
            )
        ]
    )

    messages = [Message(role="user", content="test")]
    result = await provider.create(messages=messages, model="gpt-4o", system_prompt="Be helpful")

    assert result.content == "OK"
    call = fake_client.responses.calls[0]
    assert call["instructions"] == "Be helpful"


async def test_create_with_system_messages(
    provider: OpenAIResponseProvider, fake_client: FakeAsyncOpenAI
) -> None:
    fake_client.responses = FakeResponsesClient(
        responses=[
            FakeResponse(
                output=[
                    FakeResponseItem(
                        type="message",
                        role="assistant",
                        content=[{"type": "output_text", "text": "Done"}],
                    )
                ],
                usage=FakeUsage(input_tokens=4, output_tokens=1),
            )
        ]
    )

    messages = [
        Message(role="system", content="First instruction"),
        Message(role="system", content="Second instruction"),
        Message(role="user", content="Go"),
    ]
    result = await provider.create(messages=messages, model="gpt-4o")

    assert result.content == "Done"
    call = fake_client.responses.calls[0]
    assert call["instructions"] == "First instruction\nSecond instruction"
    assert len(call["input"]) == 1
    assert call["input"][0]["role"] == "user"


async def test_create_with_system_prompt_and_system_messages(
    provider: OpenAIResponseProvider, fake_client: FakeAsyncOpenAI
) -> None:
    fake_client.responses = FakeResponsesClient(
        responses=[
            FakeResponse(
                output=[
                    FakeResponseItem(
                        type="message",
                        role="assistant",
                        content=[{"type": "output_text", "text": "Yes"}],
                    )
                ],
                usage=FakeUsage(input_tokens=2, output_tokens=1),
            )
        ]
    )

    messages = [
        Message(role="system", content="Context"),
        Message(role="user", content="Q"),
    ]
    result = await provider.create(messages=messages, model="gpt-4o", system_prompt="Base prompt")

    assert result.content == "Yes"
    call = fake_client.responses.calls[0]
    assert call["instructions"] == "Base prompt\nContext"


async def test_create_with_tools(
    provider: OpenAIResponseProvider, fake_client: FakeAsyncOpenAI
) -> None:
    fake_client.responses = FakeResponsesClient(
        responses=[
            FakeResponse(
                output=[
                    FakeResponseItem(
                        type="message",
                        role="assistant",
                        content=[{"type": "output_text", "text": "OK"}],
                    )
                ],
                usage=FakeUsage(input_tokens=10, output_tokens=2),
            )
        ]
    )

    tool = Tool(
        name="get_weather",
        description="Get weather",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        handler=lambda city: "sunny",
    )
    messages = [Message(role="user", content="What's the weather?")]
    result = await provider.create(messages=messages, model="gpt-4o", tools=[tool])

    assert result.content == "OK"
    call = fake_client.responses.calls[0]
    assert call["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]


async def test_create_with_tool_result_message(
    provider: OpenAIResponseProvider, fake_client: FakeAsyncOpenAI
) -> None:
    fake_client.responses = FakeResponsesClient(
        responses=[
            FakeResponse(
                output=[
                    FakeResponseItem(
                        type="message",
                        role="assistant",
                        content=[{"type": "output_text", "text": "Thanks"}],
                    )
                ],
                usage=FakeUsage(input_tokens=5, output_tokens=2),
            )
        ]
    )

    messages = [
        Message(role="user", content="Call tool"),
        Message(
            role="tool",
            content='{"temp": 72}',
            tool_call_id="call_123",
            name="get_weather",
        ),
    ]
    result = await provider.create(messages=messages, model="gpt-4o")

    assert result.content == "Thanks"
    call = fake_client.responses.calls[0]
    assert call["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Call tool"}],
        },
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": '{"temp": 72}',
        },
    ]


async def test_create_with_assistant_tool_calls(
    provider: OpenAIResponseProvider, fake_client: FakeAsyncOpenAI
) -> None:
    fake_client.responses = FakeResponsesClient(
        responses=[
            FakeResponse(
                output=[
                    FakeResponseItem(
                        type="function_call",
                        call_id="call_456",
                        name="search",
                        arguments=json.dumps({"query": "cats"}),
                    )
                ],
                usage=FakeUsage(input_tokens=8, output_tokens=4),
            )
        ]
    )

    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="call_456", name="search", arguments={"query": "cats"})],
        ),
    ]
    result = await provider.create(messages=messages, model="gpt-4o")

    call = fake_client.responses.calls[0]
    assert call["input"] == [
        {
            "type": "function_call",
            "call_id": "call_456",
            "name": "search",
            "arguments": json.dumps({"query": "cats"}),
        }
    ]
    assert result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_456"
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].arguments == {"query": "cats"}


async def test_create_with_assistant_text_and_tool_calls(
    provider: OpenAIResponseProvider, fake_client: FakeAsyncOpenAI
) -> None:
    fake_client.responses = FakeResponsesClient(
        responses=[
            FakeResponse(
                output=[
                    FakeResponseItem(
                        type="message",
                        role="assistant",
                        content=[{"type": "output_text", "text": "Let me search."}],
                    ),
                    FakeResponseItem(
                        type="function_call",
                        call_id="call_789",
                        name="search",
                        arguments=json.dumps({"query": "dogs"}),
                    ),
                ],
                usage=FakeUsage(input_tokens=6, output_tokens=5),
            )
        ]
    )

    messages = [Message(role="user", content="Search")]
    result = await provider.create(messages=messages, model="gpt-4o")

    assert result.content == "Let me search."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_789"
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].arguments == {"query": "dogs"}


async def test_create_with_multiple_function_calls(
    provider: OpenAIResponseProvider, fake_client: FakeAsyncOpenAI
) -> None:
    fake_client.responses = FakeResponsesClient(
        responses=[
            FakeResponse(
                output=[
                    FakeResponseItem(
                        type="function_call",
                        call_id="call_a",
                        name="tool_a",
                        arguments=json.dumps({"x": 1}),
                    ),
                    FakeResponseItem(
                        type="function_call",
                        call_id="call_b",
                        name="tool_b",
                        arguments=json.dumps({"y": 2}),
                    ),
                ],
                usage=FakeUsage(input_tokens=4, output_tokens=6),
            )
        ]
    )

    messages = [Message(role="user", content="Do it")]
    result = await provider.create(messages=messages, model="gpt-4o")

    assert result.content == ""
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].id == "call_a"
    assert result.tool_calls[1].id == "call_b"


def test_format_assistant_message_text_only(provider: OpenAIResponseProvider) -> None:
    response = LLMResponse(content="Hello there")
    msg = provider.format_assistant_message(response)

    assert msg.role == "assistant"
    assert msg.content == "Hello there"
    assert msg.tool_calls is None


def test_format_assistant_message_with_tool_calls(
    provider: OpenAIResponseProvider,
) -> None:
    response = LLMResponse(
        content="Using tool",
        tool_calls=[
            ToolCall(id="tc_1", name="calc", arguments={"a": 1}),
        ],
    )
    msg = provider.format_assistant_message(response)

    assert msg.role == "assistant"
    assert msg.content == "Using tool"
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].id == "tc_1"
    assert msg.tool_calls[0].name == "calc"
    assert msg.tool_calls[0].arguments == {"a": 1}


def test_format_tool_result(provider: OpenAIResponseProvider) -> None:
    tool_call = ToolCall(id="tc_99", name="fetch", arguments={"url": "http://x"})
    result = ToolResult(tool_call_id="tc_99", content="data", is_error=False)
    msg = provider.format_tool_result(tool_call, result)

    assert msg.role == "tool"
    assert msg.content == "data"
    assert msg.tool_call_id == "tc_99"
    assert msg.name == "fetch"


def test_provider_satisfies_protocol() -> None:
    from coreouto.providers.base import Provider

    fake_client = FakeAsyncOpenAI()
    provider = OpenAIResponseProvider(client=fake_client)
    assert isinstance(provider, Provider)
