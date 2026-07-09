from __future__ import annotations

import json
import types
from typing import Any

import pytest

from coreouto._types import (
    ImageBlock,
    LLMResponse,
    Message,
    ToolCall,
    ToolResult,
    Usage,
)
from coreouto.providers import openai as openai_provider
from coreouto.tools import Tool


class FakeChatStreamManager:
    def __init__(self, completion: Any, events: list[Any] | None = None) -> None:
        self._completion = completion
        self._events = events or []

    async def __aenter__(self) -> FakeChatStreamManager:
        return self

    async def __aiter__(self) -> Any:
        for event in self._events:
            yield event

    async def get_final_completion(self) -> Any:
        return self._completion

    async def __aexit__(self, *args: Any) -> bool:
        return False


class FakeCompletions:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.stream_events: list[Any] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response

    def stream(self, **kwargs: Any) -> FakeChatStreamManager:
        self.stream_calls.append(kwargs)
        return FakeChatStreamManager(self.response, self.stream_events)


class FakeAsyncOpenAI:
    def __init__(self, response: Any) -> None:
        self.chat = types.SimpleNamespace(completions=FakeCompletions(response))


def _text_response(content: str, usage: dict[str, int] | None = None) -> Any:
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(message=types.SimpleNamespace(content=content, tool_calls=None))
        ],
        usage=types.SimpleNamespace(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
        )
        if usage
        else None,
    )


def _tool_response(tool_calls: list[dict[str, Any]], usage: dict[str, int] | None = None) -> Any:
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content=None,
                    tool_calls=[
                        types.SimpleNamespace(
                            id=tc["id"],
                            type="function",
                            function=types.SimpleNamespace(
                                name=tc["name"],
                                arguments=json.dumps(tc["arguments"]),
                            ),
                        )
                        for tc in tool_calls
                    ],
                )
            )
        ],
        usage=types.SimpleNamespace(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
        )
        if usage
        else None,
    )


@pytest.mark.asyncio
async def test_create_simple_user_message() -> None:
    response = _text_response(
        "hello", {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    )
    fake = FakeAsyncOpenAI(response)
    provider = openai_provider.OpenAIProvider(client=fake)
    messages = [Message(role="user", content="hi")]

    result = await provider.create(messages=messages, model="gpt-4")

    assert result.content == "hello"
    assert result.tool_calls == []
    assert result.usage == Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    assert result.raw is response
    call = fake.chat.completions.calls[0]
    assert call["model"] == "gpt-4"
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    assert call["tools"] is None


@pytest.mark.asyncio
async def test_create_assistant_with_tool_calls_roundtrip() -> None:
    fake = FakeAsyncOpenAI(
        _text_response("ok", {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7})
    )
    provider = openai_provider.OpenAIProvider(client=fake)
    messages = [
        Message(
            role="assistant",
            content="calling",
            tool_calls=[ToolCall(id="tc1", name="search", arguments={"query": "x"})],
        )
    ]

    await provider.create(messages=messages, model="gpt-4")

    call = fake.chat.completions.calls[0]
    assert call["messages"] == [
        {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": json.dumps({"query": "x"}),
                    },
                }
            ],
        }
    ]


@pytest.mark.asyncio
async def test_create_tool_result_message() -> None:
    fake = FakeAsyncOpenAI(
        _text_response("ok", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
    )
    provider = openai_provider.OpenAIProvider(client=fake)
    messages = [Message(role="tool", content="result", tool_call_id="tc1", name="search")]

    await provider.create(messages=messages, model="gpt-4")

    call = fake.chat.completions.calls[0]
    assert call["messages"] == [{"role": "tool", "tool_call_id": "tc1", "content": "result"}]


@pytest.mark.asyncio
async def test_create_system_prompt_prepended() -> None:
    fake = FakeAsyncOpenAI(
        _text_response("ok", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
    )
    provider = openai_provider.OpenAIProvider(client=fake)
    messages = [Message(role="user", content="hi")]

    await provider.create(messages=messages, model="gpt-4", system_prompt="be helpful")

    call = fake.chat.completions.calls[0]
    assert call["messages"] == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
    ]


@pytest.mark.asyncio
async def test_create_tools_sent() -> None:
    fake = FakeAsyncOpenAI(
        _text_response("ok", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
    )
    provider = openai_provider.OpenAIProvider(client=fake)
    tool = Tool(
        name="search",
        description="search the web",
        parameters={"type": "object", "properties": {}},
        handler=lambda: None,
    )

    await provider.create(messages=[], model="gpt-4", tools=[tool])

    call = fake.chat.completions.calls[0]
    assert call["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "search the web",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


@pytest.mark.asyncio
async def test_create_usage_populated() -> None:
    fake = FakeAsyncOpenAI(
        _text_response("ok", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    )
    provider = openai_provider.OpenAIProvider(client=fake)

    result = await provider.create(messages=[], model="gpt-4")

    assert result.usage == Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def test_format_assistant_message_text_only() -> None:
    provider = openai_provider.OpenAIProvider(client=FakeAsyncOpenAI(None))
    response = LLMResponse(content="hello")

    message = provider.format_assistant_message(response)

    assert message == Message(role="assistant", content="hello")


def test_format_assistant_message_with_tool_calls() -> None:
    provider = openai_provider.OpenAIProvider(client=FakeAsyncOpenAI(None))
    response = LLMResponse(
        content="calling",
        tool_calls=[ToolCall(id="tc1", name="search", arguments={"query": "x"})],
    )

    message = provider.format_assistant_message(response)

    assert message.role == "assistant"
    assert message.content == "calling"
    assert message.tool_calls == [ToolCall(id="tc1", name="search", arguments={"query": "x"})]


def test_format_tool_result() -> None:
    provider = openai_provider.OpenAIProvider(client=FakeAsyncOpenAI(None))
    tool_call = ToolCall(id="tc1", name="search", arguments={"query": "x"})
    result = ToolResult(tool_call_id="tc1", content="found")

    message = provider.format_tool_result(tool_call, result)

    assert message == Message(
        role="tool",
        tool_call_id="tc1",
        content="found",
        name="search",
    )


def test_format_tool_result_text_only_unchanged() -> None:
    provider = openai_provider.OpenAIProvider(client=FakeAsyncOpenAI(None))
    tool_call = ToolCall(id="tc1", name="search", arguments={"query": "x"})
    result = ToolResult(tool_call_id="tc1", content="just text")

    message = provider.format_tool_result(tool_call, result)

    assert message.role == "tool"
    assert message.tool_call_id == "tc1"
    assert message.content == "just text"
    assert message.name == "search"


def test_format_tool_result_multimodal_raises() -> None:
    provider = openai_provider.OpenAIProvider(client=FakeAsyncOpenAI(None))
    tool_call = ToolCall(id="tc1", name="search", arguments={"query": "x"})
    result = ToolResult(
        tool_call_id="tc1",
        blocks=[ImageBlock(url="https://example.com/cat.png", mime_type="image/png")],
    )

    with pytest.raises(ValueError) as excinfo:
        provider.format_tool_result(tool_call, result)

    msg = str(excinfo.value)
    assert "image block detected" in msg
    assert "openai-response" in msg


def test_import_error_when_openai_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openai_provider, "AsyncOpenAI", None)
    with pytest.raises(ImportError, match="pip install coreouto\\[openai\\]"):
        openai_provider.OpenAIProvider(api_key="fake")


@pytest.mark.asyncio
async def test_stream_off_by_default_uses_create() -> None:
    fake = FakeAsyncOpenAI(
        _text_response("ok", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
    )
    provider = openai_provider.OpenAIProvider(client=fake)
    await provider.create(messages=[Message(role="user", content="hi")], model="gpt-4")
    assert fake.chat.completions.calls
    assert not fake.chat.completions.stream_calls


@pytest.mark.asyncio
async def test_stream_true_routes_to_streaming_path() -> None:
    fake = FakeAsyncOpenAI(
        _text_response("streamed", {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})
    )
    provider = openai_provider.OpenAIProvider(client=fake, stream=True)
    result = await provider.create(messages=[Message(role="user", content="hi")], model="gpt-4")
    assert fake.chat.completions.stream_calls
    assert not fake.chat.completions.calls
    assert result.content == "streamed"
    assert result.usage == Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5)


@pytest.mark.asyncio
async def test_stream_per_call_override_enables_streaming() -> None:
    fake = FakeAsyncOpenAI(
        _text_response("ok", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
    )
    provider = openai_provider.OpenAIProvider(client=fake, stream=False)
    await provider.create(messages=[Message(role="user", content="hi")], model="gpt-4", stream=True)
    assert fake.chat.completions.stream_calls
    assert not fake.chat.completions.calls
    assert "stream" not in fake.chat.completions.stream_calls[0]


@pytest.mark.asyncio
async def test_stream_invokes_text_callback() -> None:
    fake = FakeAsyncOpenAI(
        _text_response("ok", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
    )
    fake.chat.completions.stream_events = [
        types.SimpleNamespace(type="content.delta", delta="Hello "),
        types.SimpleNamespace(type="content.delta", delta="world"),
    ]
    provider = openai_provider.OpenAIProvider(client=fake, stream=True)
    received: list[str] = []

    async def cb(text: str) -> None:
        received.append(text)

    await provider.create(
        messages=[Message(role="user", content="hi")], model="gpt-4", _on_stream_text=cb
    )
    assert received == ["Hello ", "world"]


@pytest.mark.asyncio
async def test_create_extracts_reasoning_content() -> None:
    response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content="ok",
                    tool_calls=None,
                    reasoning_content="Deep reasoning here.",
                )
            )
        ],
        usage=None,
    )
    fake = FakeAsyncOpenAI(response)
    provider = openai_provider.OpenAIProvider(client=fake)
    result = await provider.create(messages=[Message(role="user", content="hi")], model="gpt-4")
    assert result.thinking == "Deep reasoning here."


@pytest.mark.asyncio
async def test_stream_invokes_thinking_callback() -> None:
    fake = FakeAsyncOpenAI(
        _text_response("ok", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
    )
    fake.chat.completions.stream_events = [
        types.SimpleNamespace(
            type="chunk",
            chunk=types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        delta=types.SimpleNamespace(content=None, reasoning_content="Thinking...")
                    )
                ]
            ),
        ),
    ]
    provider = openai_provider.OpenAIProvider(client=fake, stream=True)
    received: list[str] = []

    async def cb(text: str) -> None:
        received.append(text)

    await provider.create(
        messages=[Message(role="user", content="hi")], model="gpt-4", _on_stream_thinking=cb
    )
    assert received == ["Thinking..."]
