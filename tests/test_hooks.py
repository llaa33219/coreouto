from __future__ import annotations

import asyncio

import pytest

from coreouto.hooks import (
    AFTER_LLM_CALL,
    AFTER_TOOL_CALL,
    BEFORE_LLM_CALL,
    BEFORE_TOOL_CALL,
    ON_FINISH,
    ON_ITERATION,
    ON_STREAM_TEXT,
    ON_STREAM_THINKING,
    ON_THINKING,
    ON_USER_INJECTION,
    clear_hooks,
    get_hooks,
    register_hook,
    trigger,
)


def test_event_constants_exist_as_strings() -> None:
    assert isinstance(BEFORE_LLM_CALL, str)
    assert isinstance(AFTER_LLM_CALL, str)
    assert isinstance(BEFORE_TOOL_CALL, str)
    assert isinstance(AFTER_TOOL_CALL, str)
    assert isinstance(ON_ITERATION, str)
    assert isinstance(ON_FINISH, str)
    assert isinstance(ON_USER_INJECTION, str)
    assert isinstance(ON_STREAM_TEXT, str)
    assert isinstance(ON_STREAM_THINKING, str)
    assert isinstance(ON_THINKING, str)


def test_event_constants_unique() -> None:
    constants = [
        BEFORE_LLM_CALL,
        AFTER_LLM_CALL,
        BEFORE_TOOL_CALL,
        AFTER_TOOL_CALL,
        ON_ITERATION,
        ON_FINISH,
        ON_USER_INJECTION,
        ON_STREAM_TEXT,
        ON_STREAM_THINKING,
        ON_THINKING,
    ]
    assert len(constants) == len(set(constants))


def _reset() -> None:
    clear_hooks()


def test_register_hook_and_get() -> None:
    _reset()
    called = []

    def hook(**kwargs: object) -> None:
        called.append(kwargs)

    register_hook(BEFORE_LLM_CALL, hook)
    assert get_hooks(BEFORE_LLM_CALL) == [hook]


def test_get_hooks_returns_copy() -> None:
    _reset()

    def hook(**kwargs: object) -> None:
        pass

    register_hook(AFTER_LLM_CALL, hook)
    copy = get_hooks(AFTER_LLM_CALL)
    copy.append(lambda **kw: None)
    assert len(get_hooks(AFTER_LLM_CALL)) == 1


def test_get_hooks_unknown_event_returns_empty() -> None:
    _reset()
    assert get_hooks("nonexistent_event") == []


def test_clear_single_event() -> None:
    _reset()
    register_hook(BEFORE_LLM_CALL, lambda **kw: None)
    register_hook(AFTER_LLM_CALL, lambda **kw: None)

    clear_hooks(BEFORE_LLM_CALL)
    assert get_hooks(BEFORE_LLM_CALL) == []
    assert len(get_hooks(AFTER_LLM_CALL)) == 1


def test_clear_all() -> None:
    _reset()
    register_hook(BEFORE_LLM_CALL, lambda **kw: None)
    register_hook(AFTER_LLM_CALL, lambda **kw: None)
    register_hook(ON_FINISH, lambda **kw: None)

    clear_hooks()
    assert get_hooks(BEFORE_LLM_CALL) == []
    assert get_hooks(AFTER_LLM_CALL) == []
    assert get_hooks(ON_FINISH) == []


async def test_trigger_sync_hook() -> None:
    _reset()
    received: list[dict[str, object]] = []

    def hook(**kwargs: object) -> None:
        received.append(kwargs)

    register_hook(BEFORE_TOOL_CALL, hook)
    await trigger(BEFORE_TOOL_CALL, tool_name="search", query="hello")
    assert received == [{"tool_name": "search", "query": "hello"}]


async def test_trigger_async_hook() -> None:
    _reset()
    received: list[dict[str, object]] = []

    async def hook(**kwargs: object) -> None:
        await asyncio.sleep(0)
        received.append(kwargs)

    register_hook(AFTER_LLM_CALL, hook)
    await trigger(AFTER_LLM_CALL, model="gpt-4")
    assert received == [{"model": "gpt-4"}]


async def test_trigger_mixed_hooks_order() -> None:
    _reset()
    order: list[str] = []

    def sync_hook(**kwargs: object) -> None:
        order.append("sync")

    async def async_hook(**kwargs: object) -> None:
        await asyncio.sleep(0)
        order.append("async")

    register_hook(ON_ITERATION, sync_hook)
    register_hook(ON_ITERATION, async_hook)
    register_hook(ON_ITERATION, sync_hook)

    await trigger(ON_ITERATION, iteration=1)
    assert order == ["sync", "async", "sync"]


async def test_trigger_no_hooks() -> None:
    _reset()
    await trigger(ON_FINISH, result="done")


async def test_trigger_exception_propagates() -> None:
    _reset()

    def bad_hook(**kwargs: object) -> None:
        raise ValueError("boom")

    register_hook(BEFORE_LLM_CALL, bad_hook)
    with pytest.raises(ValueError, match="boom"):
        await trigger(BEFORE_LLM_CALL)


async def test_trigger_exception_stops_later_hooks() -> None:
    _reset()
    order: list[str] = []

    def first(**kwargs: object) -> None:
        order.append("first")
        raise RuntimeError("fail")

    def second(**kwargs: object) -> None:
        order.append("second")

    register_hook(AFTER_TOOL_CALL, first)
    register_hook(AFTER_TOOL_CALL, second)

    with pytest.raises(RuntimeError, match="fail"):
        await trigger(AFTER_TOOL_CALL)

    assert order == ["first"]


async def test_trigger_on_finish_hook() -> None:
    _reset()
    received: list[dict[str, object]] = []

    def hook(**kwargs: object) -> None:
        received.append(kwargs)

    register_hook(ON_FINISH, hook)
    await trigger(
        ON_FINISH,
        content="answer",
        messages=[],
        iterations=1,
    )
    assert received == [
        {
            "content": "answer",
            "messages": [],
            "iterations": 1,
        }
    ]


@pytest.mark.asyncio
async def test_on_stream_text_fires_through_agent() -> None:
    _reset()
    from coreouto import Agent, AgentConfig, register_provider
    from coreouto._types import LLMResponse, Message, Usage
    from coreouto.providers import clear_providers

    class StreamingProvider:
        _stream = True

        async def create(self, messages, *, model, tools=None, system_prompt=None, **kwargs):
            cb = kwargs.pop("_on_stream_text", None)
            if cb is not None:
                await cb("Hello ")
                await cb("world")
            return LLMResponse(
                content="Hello world",
                tool_calls=[],
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                stop_reason="end_turn",
            )

        def format_assistant_message(self, response):
            return Message(role="assistant", content=response.content or "")

        def format_tool_result(self, tool_call, result):
            return Message(role="tool", content="", tool_call_id=tool_call.id, name=tool_call.name)

    clear_providers()
    register_provider("test-stream", StreamingProvider())

    received: list[str] = []

    def hook(*, text: str, **_kwargs: object) -> None:
        received.append(text)

    register_hook(ON_STREAM_TEXT, hook)

    agent = Agent(AgentConfig(name="t", model="m", provider="test-stream"))
    response = await agent.call("hi")

    assert response.content == "Hello world"
    assert received == ["Hello ", "world"]

    clear_hooks()
    clear_providers()


@pytest.mark.asyncio
async def test_on_thinking_fires_when_response_has_thinking() -> None:
    _reset()
    from coreouto import Agent, AgentConfig, register_provider
    from coreouto._types import LLMResponse, Message, Usage
    from coreouto.providers import clear_providers

    class ThinkingProvider:
        async def create(self, messages, *, model, tools=None, system_prompt=None, **kwargs):
            kwargs.pop("_on_stream_text", None)
            kwargs.pop("_on_stream_thinking", None)
            return LLMResponse(
                content="Answer.",
                thinking="I reasoned about this.",
                tool_calls=[],
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                stop_reason="end_turn",
            )

        def format_assistant_message(self, response):
            return Message(role="assistant", content=response.content or "")

        def format_tool_result(self, tool_call, result):
            return Message(role="tool", content="", tool_call_id=tool_call.id, name=tool_call.name)

    clear_providers()
    register_provider("test-think", ThinkingProvider())

    received: list[str] = []

    def hook(*, thinking: str, **_kwargs: object) -> None:
        received.append(thinking)

    register_hook(ON_THINKING, hook)

    agent = Agent(AgentConfig(name="t", model="m", provider="test-think"))
    await agent.call("hi")

    assert received == ["I reasoned about this."]

    clear_hooks()
    clear_providers()
