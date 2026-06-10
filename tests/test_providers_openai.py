from __future__ import annotations

import json
import types
from typing import Any

import pytest

from coreouto._types import LLMResponse, Message, ToolCall, ToolResult, Usage
from coreouto.providers import openai as openai_provider
from coreouto.tools import Tool


class FakeCompletions:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


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


def test_import_error_when_openai_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openai_provider, "AsyncOpenAI", None)
    with pytest.raises(ImportError, match="pip install coreouto\\[openai\\]"):
        openai_provider.OpenAIProvider(api_key="fake")
