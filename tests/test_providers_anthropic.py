from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any

import pytest

from coreouto._types import (
    AudioBlock,
    DocumentBlock,
    ImageBlock,
    LLMResponse,
    Message,
    TextBlock,
    ToolCall,
    ToolResult,
    VideoBlock,
)
from coreouto.providers.base import Provider
from coreouto.tools import Tool

# Fake SDK objects so tests run without `anthropic` installed.


@dataclass
class FakeContentBlock:
    type: str
    text: str | None = None
    thinking: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class FakeMessage:
    content: list[FakeContentBlock] | None
    usage: FakeUsage


@dataclass
class FakeMessages:
    _response: FakeMessage | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    stream_calls: list[dict[str, Any]] = field(default_factory=list)
    stream_text_deltas: list[str] = field(default_factory=list)
    stream_thinking_deltas: list[str] = field(default_factory=list)

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> FakeMessage:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "system": system,
                "tools": tools,
                "max_tokens": max_tokens,
                "kwargs": kwargs,
            }
        )
        if self._response is None:
            raise AssertionError("FakeMessages.create called without a queued response")
        return self._response

    def stream(self, **kwargs: Any) -> FakeMessageStream:
        self.stream_calls.append(kwargs)
        return FakeMessageStream(
            self._response, self.stream_text_deltas, self.stream_thinking_deltas
        )

    def queue(self, response: FakeMessage) -> None:
        self._response = response


class FakeMessageStream:
    """Async context manager mimicking Anthropic's AsyncMessageStream."""

    def __init__(
        self,
        message: FakeMessage | None,
        text_deltas: list[str] | None = None,
        thinking_deltas: list[str] | None = None,
    ) -> None:
        self._message = message
        self._text_deltas = text_deltas or []
        self._thinking_deltas = thinking_deltas or []

    async def __aenter__(self) -> FakeMessageStream:
        return self

    async def __aiter__(self) -> Any:
        for text in self._text_deltas:
            yield types.SimpleNamespace(
                type="content_block_delta",
                delta=types.SimpleNamespace(type="text_delta", text=text),
            )
        for thinking in self._thinking_deltas:
            yield types.SimpleNamespace(
                type="content_block_delta",
                delta=types.SimpleNamespace(type="thinking_delta", thinking=thinking),
            )

    async def get_final_message(self) -> FakeMessage:
        if self._message is None:
            raise AssertionError("FakeMessageStream entered without a queued response")
        return self._message

    async def __aexit__(self, *args: Any) -> bool:
        return False


@dataclass
class FakeAsyncAnthropic:
    messages: FakeMessages = field(default_factory=FakeMessages)


@pytest.fixture
def fake_client():
    return FakeAsyncAnthropic()


@pytest.fixture
def provider(fake_client):
    from coreouto.providers.anthropic import AnthropicProvider

    return AnthropicProvider(client=fake_client)


@pytest.mark.asyncio
async def test_create_simple_user_message(provider, fake_client):
    fake_client.messages.queue(
        FakeMessage(
            content=[FakeContentBlock(type="text", text="Hello back")],
            usage=FakeUsage(input_tokens=5, output_tokens=3),
        )
    )
    result = await provider.create(
        messages=[Message(role="user", content="Hello")],
        model="claude-3-haiku-20240307",
    )
    assert isinstance(result, LLMResponse)
    assert result.content == "Hello back"
    assert result.tool_calls == []
    assert result.usage.prompt_tokens == 5
    assert result.usage.completion_tokens == 3
    assert result.usage.total_tokens == 8
    assert result.raw is not None

    call = fake_client.messages.calls[0]
    assert call["model"] == "claude-3-haiku-20240307"
    assert call["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    assert call["system"] is None


@pytest.mark.asyncio
async def test_create_with_tool_result(provider, fake_client):
    fake_client.messages.queue(
        FakeMessage(
            content=[FakeContentBlock(type="text", text="Done")],
            usage=FakeUsage(input_tokens=10, output_tokens=2),
        )
    )
    result = await provider.create(
        messages=[
            Message(role="user", content="Call a tool"),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="tu_1", name="my_tool", arguments={"x": 1})],
            ),
            Message(role="tool", content="result1", tool_call_id="tu_1", name="my_tool"),
        ],
        model="claude-3-haiku-20240307",
    )
    assert result.content == "Done"

    call = fake_client.messages.calls[0]
    assert call["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Call a tool"}]},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tu_1", "name": "my_tool", "input": {"x": 1}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "result1"}],
        },
    ]


@pytest.mark.asyncio
async def test_create_with_assistant_tool_calls(provider, fake_client):
    fake_client.messages.queue(
        FakeMessage(
            content=[
                FakeContentBlock(type="text", text="Let me calculate"),
                FakeContentBlock(
                    type="tool_use",
                    id="tu_2",
                    name="calc",
                    input={"a": 2, "b": 3},
                ),
            ],
            usage=FakeUsage(input_tokens=8, output_tokens=6),
        )
    )
    result = await provider.create(
        messages=[Message(role="user", content="Add 2+3")],
        model="claude-3-haiku-20240307",
    )
    assert result.content == "Let me calculate"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "tu_2"
    assert result.tool_calls[0].name == "calc"
    assert result.tool_calls[0].arguments == {"a": 2, "b": 3}


@pytest.mark.asyncio
async def test_create_with_system_prompt_arg(provider, fake_client):
    fake_client.messages.queue(
        FakeMessage(
            content=[FakeContentBlock(type="text", text="ok")],
            usage=FakeUsage(input_tokens=1, output_tokens=1),
        )
    )
    await provider.create(
        messages=[Message(role="user", content="hi")],
        model="claude-3-haiku-20240307",
        system_prompt="You are a test bot",
    )
    call = fake_client.messages.calls[0]
    assert call["system"] == "You are a test bot"


@pytest.mark.asyncio
async def test_create_with_system_role_messages(provider, fake_client):
    fake_client.messages.queue(
        FakeMessage(
            content=[FakeContentBlock(type="text", text="ok")],
            usage=FakeUsage(input_tokens=1, output_tokens=1),
        )
    )
    await provider.create(
        messages=[
            Message(role="system", content="Sys1"),
            Message(role="system", content="Sys2"),
            Message(role="user", content="hi"),
        ],
        model="claude-3-haiku-20240307",
    )
    call = fake_client.messages.calls[0]
    assert call["system"] == "Sys1\nSys2"
    assert all(m["role"] != "system" for m in call["messages"])


@pytest.mark.asyncio
async def test_create_with_tools(provider, fake_client):
    fake_client.messages.queue(
        FakeMessage(
            content=[FakeContentBlock(type="text", text="tool response")],
            usage=FakeUsage(input_tokens=2, output_tokens=2),
        )
    )
    tool = Tool(
        name="search",
        description="Search the web",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        handler=lambda q: q,
    )
    await provider.create(
        messages=[Message(role="user", content="Search for cats")],
        model="claude-3-haiku-20240307",
        tools=[tool],
    )
    call = fake_client.messages.calls[0]
    assert call["tools"] == [
        {
            "name": "search",
            "description": "Search the web",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
    ]


@pytest.mark.asyncio
async def test_create_mixed_response(provider, fake_client):
    fake_client.messages.queue(
        FakeMessage(
            content=[
                FakeContentBlock(type="text", text="Mixed"),
                FakeContentBlock(type="tool_use", id="tu_3", name="foo", input={"bar": 1}),
            ],
            usage=FakeUsage(input_tokens=4, output_tokens=5),
        )
    )
    result = await provider.create(
        messages=[Message(role="user", content="Go")],
        model="claude-3-haiku-20240307",
    )
    assert result.content == "Mixed"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0] == ToolCall(id="tu_3", name="foo", arguments={"bar": 1})


def test_format_assistant_message_text_only(provider):
    response = LLMResponse(content="Just text", usage=None)
    msg = provider.format_assistant_message(response)
    assert msg.role == "assistant"
    assert msg.content == "Just text"
    assert msg.tool_calls is None


def test_format_assistant_message_with_tool_calls(provider):
    response = LLMResponse(
        content="Using tool",
        tool_calls=[ToolCall(id="tc1", name="t1", arguments={"a": 1})],
    )
    msg = provider.format_assistant_message(response)
    assert msg.role == "assistant"
    assert isinstance(msg.content, list)
    assert msg.content[0] == TextBlock(text="Using tool")
    assert msg.content[1] == ToolCall(id="tc1", name="t1", arguments={"a": 1})
    assert msg.tool_calls == [ToolCall(id="tc1", name="t1", arguments={"a": 1})]


def test_format_tool_result(provider):
    tc = ToolCall(id="tc1", name="t1", arguments={})
    tr = ToolResult(tool_call_id="tc1", content="res", is_error=False)
    msg = provider.format_tool_result(tc, tr)
    assert msg.role == "tool"
    assert msg.content == "res"
    assert msg.tool_call_id == "tc1"
    assert msg.name == "t1"


@dataclass
class FakeAsyncAnthropicWithRecording:
    recorded_args: dict[str, Any] = field(default_factory=dict)
    messages: FakeMessages = field(default_factory=FakeMessages)

    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None, **kwargs: Any
    ) -> None:
        self.recorded_args = {"api_key": api_key, "base_url": base_url, **kwargs}
        self.messages = FakeMessages()


def test_provider_satisfies_protocol(provider):
    assert isinstance(provider, Provider)


def test_anthropic_provider_accepts_base_url_construction(fake_client):
    from coreouto.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(
        api_key="test", base_url="https://proxy.example.com", client=fake_client
    )
    assert provider._client is fake_client


def test_anthropic_provider_passes_base_url_to_client(monkeypatch: pytest.MonkeyPatch):
    from coreouto.providers.anthropic import AnthropicProvider

    monkeypatch.setattr(
        "coreouto.providers.anthropic._import_anthropic",
        lambda: FakeAsyncAnthropicWithRecording,
    )
    provider = AnthropicProvider(api_key="test", base_url="https://proxy.example.com")
    assert provider._client.recorded_args["base_url"] == "https://proxy.example.com"


def test_anthropic_provider_works_without_base_url(monkeypatch: pytest.MonkeyPatch):
    from coreouto.providers.anthropic import AnthropicProvider

    monkeypatch.setattr(
        "coreouto.providers.anthropic._import_anthropic",
        lambda: FakeAsyncAnthropicWithRecording,
    )
    provider = AnthropicProvider(api_key="test")
    assert provider._client.recorded_args["base_url"] is None


def test_anthropic_register_function_accepts_base_url(monkeypatch: pytest.MonkeyPatch):
    from coreouto.providers import clear_providers, get_provider
    from coreouto.providers.anthropic import register

    monkeypatch.setattr(
        "coreouto.providers.anthropic._import_anthropic",
        lambda: FakeAsyncAnthropicWithRecording,
    )
    clear_providers()
    register(api_key="x", base_url="https://proxy.example.com", name="anthropic-proxy")
    provider = get_provider("anthropic-proxy")
    assert provider is not None
    assert provider._client.recorded_args["base_url"] == "https://proxy.example.com"
    clear_providers()


@pytest.mark.asyncio
async def test_create_groups_consecutive_tool_messages(provider, fake_client):
    fake_client.messages.queue(
        FakeMessage(
            content=[FakeContentBlock(type="text", text="ok")],
            usage=FakeUsage(input_tokens=1, output_tokens=1),
        )
    )
    await provider.create(
        messages=[
            Message(role="user", content="call tools"),
            Message(role="tool", content="r1", tool_call_id="tc1", name="t1"),
            Message(role="tool", content="r2", tool_call_id="tc2", name="t2"),
            Message(role="user", content="next"),
        ],
        model="claude-3-haiku-20240307",
    )
    call = fake_client.messages.calls[0]
    assert call["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "call tools"}]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tc1", "content": "r1"},
                {"type": "tool_result", "tool_use_id": "tc2", "content": "r2"},
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": "next"}]},
    ]


@pytest.mark.asyncio
async def test_create_prefers_system_prompt_arg_over_messages(provider, fake_client):
    fake_client.messages.queue(
        FakeMessage(
            content=[FakeContentBlock(type="text", text="ok")],
            usage=FakeUsage(input_tokens=1, output_tokens=1),
        )
    )
    await provider.create(
        messages=[
            Message(role="system", content="from messages"),
            Message(role="user", content="hi"),
        ],
        model="claude-3-haiku-20240307",
        system_prompt="from arg",
    )
    call = fake_client.messages.calls[0]
    assert call["system"] == "from arg"


@pytest.mark.asyncio
async def test_create_forwards_thinking_and_output_config(provider, fake_client):
    fake_client.messages.queue(
        FakeMessage(
            content=[FakeContentBlock(type="text", text="ok")],
            usage=FakeUsage(input_tokens=1, output_tokens=1),
        )
    )
    await provider.create(
        messages=[Message(role="user", content="hi")],
        model="claude-sonnet-4-6",
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
    )
    call = fake_client.messages.calls[0]
    assert call["kwargs"] == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "medium"},
    }


def test_format_tool_result_image_data(provider):
    tc = ToolCall(id="tc1", name="snap", arguments={})
    tr = ToolResult(
        tool_call_id="tc1",
        blocks=[ImageBlock(data=b"hello", mime_type="image/jpeg")],
    )
    msg = provider.format_tool_result(tc, tr)
    assert msg.role == "tool"
    assert msg.tool_call_id == "tc1"
    assert msg.name == "snap"
    assert msg.content == [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "aGVsbG8=",
            },
        }
    ]


def test_format_tool_result_image_url(provider):
    tc = ToolCall(id="tc1", name="snap", arguments={})
    tr = ToolResult(
        tool_call_id="tc1",
        blocks=[ImageBlock(url="https://example.com/cat.png")],
    )
    msg = provider.format_tool_result(tc, tr)
    assert msg.content == [
        {
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://example.com/cat.png",
            },
        }
    ]


def test_format_tool_result_text_and_image(provider):
    tc = ToolCall(id="tc1", name="snap", arguments={})
    tr = ToolResult(
        tool_call_id="tc1",
        blocks=[
            TextBlock(text="a picture"),
            ImageBlock(data=b"hello", mime_type="image/png"),
        ],
    )
    msg = provider.format_tool_result(tc, tr)
    assert msg.content == [
        {"type": "text", "text": "a picture"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "aGVsbG8=",
            },
        },
    ]


def test_format_tool_result_document_data(provider):
    tc = ToolCall(id="tc1", name="read", arguments={})
    tr = ToolResult(
        tool_call_id="tc1",
        blocks=[DocumentBlock(data=b"PDFDATA", mime_type="application/pdf")],
    )
    msg = provider.format_tool_result(tc, tr)
    assert msg.content == [
        {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "application/pdf",
                "data": "UERGREFUQQ==",
            },
        }
    ]


def test_format_tool_result_document_url(provider):
    tc = ToolCall(id="tc1", name="read", arguments={})
    tr = ToolResult(
        tool_call_id="tc1",
        blocks=[DocumentBlock(url="https://example.com/doc.pdf", mime_type="application/pdf")],
    )
    msg = provider.format_tool_result(tc, tr)
    assert msg.content == [
        {
            "type": "document",
            "source": {
                "type": "url",
                "url": "https://example.com/doc.pdf",
            },
        }
    ]


def test_format_tool_result_video_data(provider):
    tc = ToolCall(id="tc1", name="vid", arguments={})
    tr = ToolResult(
        tool_call_id="tc1",
        blocks=[VideoBlock(data=b"hello", mime_type="video/mp4")],
    )
    msg = provider.format_tool_result(tc, tr)
    assert msg.content == [
        {
            "type": "video",
            "source": {
                "type": "base64",
                "media_type": "video/mp4",
                "data": "aGVsbG8=",
            },
        }
    ]


def test_format_tool_result_audio_data(provider):
    tc = ToolCall(id="tc1", name="aud", arguments={})
    tr = ToolResult(
        tool_call_id="tc1",
        blocks=[AudioBlock(data=b"hello", mime_type="audio/mpeg")],
    )
    msg = provider.format_tool_result(tc, tr)
    assert msg.content == [
        {
            "type": "audio",
            "source": {
                "type": "base64",
                "media_type": "audio/mpeg",
                "data": "aGVsbG8=",
            },
        }
    ]


@pytest.mark.asyncio
async def test_create_with_multimodal_tool_result(provider, fake_client):
    fake_client.messages.queue(
        FakeMessage(
            content=[FakeContentBlock(type="text", text="got it")],
            usage=FakeUsage(input_tokens=4, output_tokens=2),
        )
    )
    tc = ToolCall(id="tu_img", name="snap", arguments={})
    tr = ToolResult(
        tool_call_id="tu_img",
        blocks=[ImageBlock(data=b"hello", mime_type="image/jpeg")],
    )
    await provider.create(
        messages=[
            Message(role="user", content="take a photo"),
            Message(
                role="assistant",
                content="",
                tool_calls=[tc],
            ),
            provider.format_tool_result(tc, tr),
        ],
        model="claude-3-haiku-20240307",
    )
    call = fake_client.messages.calls[0]
    assert call["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "take a photo"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_img", "name": "snap", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_img",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "aGVsbG8=",
                            },
                        }
                    ],
                }
            ],
        },
    ]


@pytest.mark.asyncio
async def test_create_handles_none_content(provider, fake_client):
    """Anthropic can return `content=None` for extended-thinking-only responses."""
    fake_client.messages.queue(
        FakeMessage(
            content=None,
            usage=FakeUsage(input_tokens=5, output_tokens=0),
        )
    )
    result = await provider.create(
        messages=[Message(role="user", content="Hello")],
        model="claude-3-haiku-20240307",
    )
    assert result.content is None
    assert result.tool_calls == []
    assert result.usage.total_tokens == 5


@pytest.mark.asyncio
async def test_create_handles_empty_content(provider, fake_client):
    """Anthropic can return an empty content list (no text, no tool_use)."""
    fake_client.messages.queue(
        FakeMessage(
            content=[],
            usage=FakeUsage(input_tokens=5, output_tokens=0),
        )
    )
    result = await provider.create(
        messages=[Message(role="user", content="Hello")],
        model="claude-3-haiku-20240307",
    )
    assert result.content is None
    assert result.tool_calls == []


@pytest.mark.asyncio
async def test_stream_off_by_default_uses_create(fake_client):
    from coreouto.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(client=fake_client)
    fake_client.messages.queue(
        FakeMessage(
            content=[FakeContentBlock(type="text", text="ok")],
            usage=FakeUsage(input_tokens=1, output_tokens=1),
        )
    )
    await provider.create(messages=[Message(role="user", content="hi")], model="m")
    assert fake_client.messages.calls
    assert not fake_client.messages.stream_calls


@pytest.mark.asyncio
async def test_stream_true_routes_to_streaming_path(fake_client):
    from coreouto.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(client=fake_client, stream=True)
    fake_client.messages.queue(
        FakeMessage(
            content=[FakeContentBlock(type="text", text="streamed")],
            usage=FakeUsage(input_tokens=2, output_tokens=4),
        )
    )
    result = await provider.create(messages=[Message(role="user", content="hi")], model="m")

    assert fake_client.messages.stream_calls
    assert not fake_client.messages.calls
    assert result.content == "streamed"
    assert result.usage.total_tokens == 6
    assert result.raw is not None


@pytest.mark.asyncio
async def test_stream_per_call_override_enables_streaming(fake_client):
    from coreouto.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(client=fake_client, stream=False)
    fake_client.messages.queue(
        FakeMessage(
            content=[FakeContentBlock(type="text", text="via override")],
            usage=FakeUsage(input_tokens=1, output_tokens=1),
        )
    )
    await provider.create(messages=[Message(role="user", content="hi")], model="m", stream=True)
    assert fake_client.messages.stream_calls
    assert not fake_client.messages.calls
    assert "stream" not in fake_client.messages.stream_calls[0]


@pytest.mark.asyncio
async def test_stream_preserves_request_kwargs(fake_client):
    from coreouto.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(client=fake_client, stream=True)
    fake_client.messages.queue(
        FakeMessage(
            content=[FakeContentBlock(type="text", text="ok")],
            usage=FakeUsage(input_tokens=1, output_tokens=1),
        )
    )
    await provider.create(
        messages=[Message(role="user", content="hi")],
        model="claude-opus-4-8",
        max_tokens=4096,
    )
    call = fake_client.messages.stream_calls[0]
    assert call["model"] == "claude-opus-4-8"
    assert call["max_tokens"] == 4096
    assert call["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert call["system"] is None


@pytest.mark.asyncio
async def test_stream_invokes_text_callback(fake_client):
    from coreouto.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(client=fake_client, stream=True)
    fake_client.messages.queue(
        FakeMessage(
            content=[FakeContentBlock(type="text", text="Hello world")],
            usage=FakeUsage(input_tokens=1, output_tokens=1),
        )
    )
    fake_client.messages.stream_text_deltas = ["Hello ", "world"]
    received: list[str] = []

    async def cb(text: str) -> None:
        received.append(text)

    await provider.create(
        messages=[Message(role="user", content="hi")], model="m", _on_stream_text=cb
    )
    assert received == ["Hello ", "world"]


@pytest.mark.asyncio
async def test_create_extracts_thinking(provider, fake_client):
    fake_client.messages.queue(
        FakeMessage(
            content=[
                FakeContentBlock(type="thinking", thinking="Let me reason about this."),
                FakeContentBlock(type="text", text="The answer is 42."),
            ],
            usage=FakeUsage(input_tokens=10, output_tokens=20),
        )
    )
    result = await provider.create(
        messages=[Message(role="user", content="What is the answer?")],
        model="claude-sonnet-4-6",
    )
    assert result.content == "The answer is 42."
    assert result.thinking == "Let me reason about this."


@pytest.mark.asyncio
async def test_stream_invokes_thinking_callback(fake_client):
    from coreouto.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(client=fake_client, stream=True)
    fake_client.messages.queue(
        FakeMessage(
            content=[FakeContentBlock(type="text", text="ok")],
            usage=FakeUsage(input_tokens=1, output_tokens=1),
        )
    )
    fake_client.messages.stream_thinking_deltas = ["Reasoning ", "step by step"]
    received: list[str] = []

    async def cb(text: str) -> None:
        received.append(text)

    await provider.create(
        messages=[Message(role="user", content="hi")], model="m", _on_stream_thinking=cb
    )
    assert received == ["Reasoning ", "step by step"]
