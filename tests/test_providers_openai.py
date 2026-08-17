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


class FakeCompletions:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.stream_chunks: list[Any] = []

    async def create(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            self.stream_calls.append(kwargs)
            return self._iterate_chunks()
        self.calls.append(kwargs)
        return self.response

    async def _iterate_chunks(self) -> Any:
        for chunk in self.stream_chunks:
            yield chunk


class FakeAsyncOpenAI:
    def __init__(self, response: Any) -> None:
        self.chat = types.SimpleNamespace(completions=FakeCompletions(response))


def _usage(prompt_tokens: int, completion_tokens: int) -> Any:
    from openai.types.completion_usage import CompletionUsage

    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _chunk(
    content: str | None = None,
    *,
    reasoning: str | None = None,
    tool_deltas: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
    usage: Any = None,
) -> Any:
    delta = types.SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        tool_calls=[
            types.SimpleNamespace(
                index=td.get("index", 0),
                id=td.get("id"),
                function=types.SimpleNamespace(
                    name=td.get("name"),
                    arguments=td.get("arguments"),
                ),
            )
            for td in (tool_deltas or [])
        ]
        or None,
    )
    return types.SimpleNamespace(
        id="chatcmpl-test",
        created=1,
        model="gpt-4",
        choices=[types.SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)],
        usage=usage,
    )


def _text_response(content: str, usage: dict[str, int] | None = None) -> Any:
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=content, tool_calls=None),
                finish_reason="stop",
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
                ),
                finish_reason="tool_calls",
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
    fake = FakeAsyncOpenAI(None)
    fake.chat.completions.stream_chunks = [
        _chunk("streamed", finish_reason="stop", usage=_usage(3, 2)),
    ]
    provider = openai_provider.OpenAIProvider(client=fake, stream=True)
    result = await provider.create(messages=[Message(role="user", content="hi")], model="gpt-4")
    assert fake.chat.completions.stream_calls
    assert not fake.chat.completions.calls
    assert result.content == "streamed"
    assert result.usage == Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    assert result.stop_reason == "stop"


@pytest.mark.asyncio
async def test_stream_per_call_override_enables_streaming() -> None:
    fake = FakeAsyncOpenAI(None)
    fake.chat.completions.stream_chunks = [_chunk("ok", finish_reason="stop")]
    provider = openai_provider.OpenAIProvider(client=fake, stream=False)
    await provider.create(messages=[Message(role="user", content="hi")], model="gpt-4", stream=True)
    assert fake.chat.completions.stream_calls
    assert not fake.chat.completions.calls
    assert fake.chat.completions.stream_calls[0]["stream"] is True


@pytest.mark.asyncio
async def test_stream_invokes_text_callback() -> None:
    fake = FakeAsyncOpenAI(None)
    fake.chat.completions.stream_chunks = [
        _chunk("Hello "),
        _chunk("world", finish_reason="stop"),
    ]
    provider = openai_provider.OpenAIProvider(client=fake, stream=True)
    received: list[str] = []

    async def cb(text: str) -> None:
        received.append(text)

    result = await provider.create(
        messages=[Message(role="user", content="hi")], model="gpt-4", _on_stream_text=cb
    )
    assert received == ["Hello ", "world"]
    assert result.content == "Hello world"


@pytest.mark.asyncio
async def test_stream_accumulates_tool_call_fragments() -> None:
    fake = FakeAsyncOpenAI(None)
    fake.chat.completions.stream_chunks = [
        _chunk(tool_deltas=[{"id": "call_1", "name": "search", "arguments": '{"que'}]),
        _chunk(tool_deltas=[{"arguments": 'ry": "x"}'}], finish_reason="tool_calls"),
    ]
    provider = openai_provider.OpenAIProvider(client=fake, stream=True)

    result = await provider.create(messages=[Message(role="user", content="hi")], model="gpt-4")

    assert result.content is None
    assert result.tool_calls == [ToolCall(id="call_1", name="search", arguments={"query": "x"})]
    assert result.stop_reason == "tool_calls"


@pytest.mark.asyncio
async def test_stream_with_non_strict_tool_schema() -> None:
    # Regression: the SDK's .stream() helper raises ValueError on tools
    # without `strict: True` before any HTTP request. The provider must use
    # low-level create(stream=True), which performs no such validation.
    fake = FakeAsyncOpenAI(None)
    fake.chat.completions.stream_chunks = [_chunk("ok", finish_reason="stop")]
    provider = openai_provider.OpenAIProvider(client=fake, stream=True)
    tool = Tool(
        name="search",
        description="search the web",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=lambda query: "result",
    )

    result = await provider.create(
        messages=[Message(role="user", content="hi")], model="gpt-4", tools=[tool]
    )

    assert result.content == "ok"
    sent_tool = fake.chat.completions.stream_calls[0]["tools"][0]
    assert "strict" not in sent_tool["function"]


@pytest.mark.asyncio
async def test_create_surfaces_finish_reason() -> None:
    response = _text_response(
        "truncated", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    )
    response.choices[0].finish_reason = "length"
    fake = FakeAsyncOpenAI(response)
    provider = openai_provider.OpenAIProvider(client=fake)

    result = await provider.create(messages=[Message(role="user", content="hi")], model="gpt-4")

    assert result.stop_reason == "length"


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
    fake = FakeAsyncOpenAI(None)
    fake.chat.completions.stream_chunks = [
        _chunk(reasoning="Thinking..."),
        _chunk("ok", finish_reason="stop"),
    ]
    provider = openai_provider.OpenAIProvider(client=fake, stream=True)
    received: list[str] = []

    async def cb(text: str) -> None:
        received.append(text)

    result = await provider.create(
        messages=[Message(role="user", content="hi")], model="gpt-4", _on_stream_thinking=cb
    )
    assert received == ["Thinking..."]
    assert result.thinking == "Thinking..."


def test_timeout_forwarded_to_client(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class RecordingOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(openai_provider, "AsyncOpenAI", RecordingOpenAI)
    openai_provider.OpenAIProvider(api_key="k", timeout=12.5)
    assert captured["timeout"] == 12.5


def test_timeout_omitted_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class RecordingOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(openai_provider, "AsyncOpenAI", RecordingOpenAI)
    openai_provider.OpenAIProvider(api_key="k")
    assert "timeout" not in captured


def test_register_forwards_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    from coreouto.providers import clear_providers, get_provider

    captured: dict[str, Any] = {}

    class RecordingOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(openai_provider, "AsyncOpenAI", RecordingOpenAI)
    clear_providers()
    openai_provider.register(api_key="k", name="openai-timeout", timeout=7.5)
    assert get_provider("openai-timeout") is not None
    assert captured["timeout"] == 7.5
    clear_providers()
