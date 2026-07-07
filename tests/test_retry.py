from __future__ import annotations

import asyncio
from typing import Any

import pytest

from coreouto._types import AgentConfig
from coreouto.agent import Agent
from coreouto.hooks import ON_RETRY, clear_hooks, register_hook
from coreouto.providers import clear_providers, register_provider
from tests.conftest import MockLLMResponse, MockProvider


@pytest.fixture(autouse=True)
def _clean_state():
    clear_hooks()
    clear_providers()
    yield
    clear_hooks()
    clear_providers()


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Avoid actually sleeping during retry tests."""

    async def _instant_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)


class FlakyProvider:
    """Wraps a MockProvider and raises on the first N ``create`` calls.

    After the failure budget is exhausted, calls pass through to the wrapped
    provider so the canned LLMResponse sequence is consumed normally.
    """

    def __init__(self, inner: MockProvider, fail_times: int, exc: BaseException) -> None:
        self._inner = inner
        self._fail_times = fail_times
        self._exc = exc
        self.create_call_count = 0

    async def create(
        self,
        messages: list[Any],
        *,
        model: str,
        tools: list[Any] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> Any:
        self.create_call_count += 1
        if self.create_call_count <= self._fail_times:
            raise self._exc
        return await self._inner.create(
            messages,
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            **kwargs,
        )

    def format_assistant_message(self, response: Any) -> Any:
        return self._inner.format_assistant_message(response)

    def format_tool_result(self, tool_call: Any, result: Any) -> Any:
        return self._inner.format_tool_result(tool_call, result)


def _agent_config(retry_intervals: list[float] | None) -> AgentConfig:
    return AgentConfig(
        name="retry-test",
        model="m",
        provider="flaky",
        max_iterations=5,
        retry_intervals=retry_intervals,
    )


# ---------------------------------------------------------------------------
# Default behavior — no retry
# ---------------------------------------------------------------------------


async def test_retry_none_propagates_immediately():
    inner = MockProvider(provider_name="openai")
    inner.queue(MockLLMResponse(content="done"))
    flaky = FlakyProvider(inner, fail_times=1, exc=RuntimeError("boom"))
    register_provider("flaky", flaky)

    agent = Agent(_agent_config(retry_intervals=None))
    with pytest.raises(RuntimeError, match="boom"):
        await agent.call("hello")

    # No retry happened — exactly one create call.
    assert flaky.create_call_count == 1


async def test_retry_empty_list_propagates_immediately():
    inner = MockProvider(provider_name="openai")
    inner.queue(MockLLMResponse(content="done"))
    flaky = FlakyProvider(inner, fail_times=1, exc=RuntimeError("boom"))
    register_provider("flaky", flaky)

    agent = Agent(_agent_config(retry_intervals=[]))
    with pytest.raises(RuntimeError, match="boom"):
        await agent.call("hello")

    assert flaky.create_call_count == 1


# ---------------------------------------------------------------------------
# Retry succeeds
# ---------------------------------------------------------------------------


async def test_retry_succeeds_after_transient_failure():
    inner = MockProvider(provider_name="openai")
    inner.queue(MockLLMResponse(content="done"))
    # First create call fails; the retry succeeds.
    flaky = FlakyProvider(inner, fail_times=1, exc=RuntimeError("transient"))
    register_provider("flaky", flaky)

    agent = Agent(_agent_config(retry_intervals=[1, 3]))
    response = await agent.call("hello")

    assert response.content == "done"
    assert response.iterations == 1
    assert flaky.create_call_count == 2  # 1 initial failure + 1 retry success


async def test_retry_succeeds_after_multiple_failures():
    inner = MockProvider(provider_name="openai")
    inner.queue(MockLLMResponse(content="done"))
    # Two failures, then success on the third create call. With
    # retry_intervals=[1, 3, 5], that's the initial attempt + retries 1 and 2
    # failing, retry 3 succeeding.
    flaky = FlakyProvider(inner, fail_times=2, exc=ConnectionError("down"))
    register_provider("flaky", flaky)

    agent = Agent(_agent_config(retry_intervals=[1, 3, 5]))
    response = await agent.call("hello")

    assert response.content == "done"
    assert flaky.create_call_count == 3


async def test_retry_zero_interval_means_immediate():
    """An interval of 0 means retry immediately with no delay (still allowed)."""
    inner = MockProvider(provider_name="openai")
    inner.queue(MockLLMResponse(content="done"))
    flaky = FlakyProvider(inner, fail_times=1, exc=RuntimeError("blip"))
    register_provider("flaky", flaky)

    agent = Agent(_agent_config(retry_intervals=[0]))
    response = await agent.call("hello")

    assert response.content == "done"
    assert flaky.create_call_count == 2


# ---------------------------------------------------------------------------
# Retry exhausted
# ---------------------------------------------------------------------------


async def test_retry_exhausted_reraises_last_error():
    inner = MockProvider(provider_name="openai")
    inner.queue(MockLLMResponse(content="done"))
    # Always fails — more failures than retries.
    flaky = FlakyProvider(inner, fail_times=99, exc=RuntimeError("persistent"))
    register_provider("flaky", flaky)

    agent = Agent(_agent_config(retry_intervals=[1, 2, 3]))
    with pytest.raises(RuntimeError, match="persistent"):
        await agent.call("hello")

    # 1 initial attempt + 3 retries = 4 create calls.
    assert flaky.create_call_count == 4


async def test_retry_single_interval_exhausted_reraises():
    inner = MockProvider(provider_name="openai")
    inner.queue(MockLLMResponse(content="done"))
    flaky = FlakyProvider(inner, fail_times=99, exc=ValueError("bad"))
    register_provider("flaky", flaky)

    agent = Agent(_agent_config(retry_intervals=[5]))
    with pytest.raises(ValueError, match="bad"):
        await agent.call("hello")

    assert flaky.create_call_count == 2  # 1 initial + 1 retry


# ---------------------------------------------------------------------------
# ON_RETRY hook
# ---------------------------------------------------------------------------


async def test_on_retry_hook_fires_with_attempt_and_interval():
    inner = MockProvider(provider_name="openai")
    inner.queue(MockLLMResponse(content="done"))
    flaky = FlakyProvider(inner, fail_times=2, exc=RuntimeError("flake"))
    register_provider("flaky", flaky)

    events: list[dict[str, Any]] = []

    def on_retry(**kwargs: Any) -> None:
        events.append(kwargs)

    register_hook(ON_RETRY, on_retry)

    agent = Agent(_agent_config(retry_intervals=[1, 3]))
    response = await agent.call("hello")

    assert response.content == "done"
    # Two failures -> two ON_RETRY fires (one before each retry sleep).
    assert len(events) == 2
    assert events[0]["attempt"] == 1
    assert events[0]["interval"] == 1
    assert isinstance(events[0]["error"], RuntimeError)
    assert events[1]["attempt"] == 2
    assert events[1]["interval"] == 3
    # messages and model are surfaced for observability.
    assert "messages" in events[0]
    assert events[0]["model"] == "m"


async def test_on_retry_hook_not_fired_when_no_retry():
    inner = MockProvider(provider_name="openai")
    inner.queue(MockLLMResponse(content="done"))
    flaky = FlakyProvider(inner, fail_times=1, exc=RuntimeError("boom"))
    register_provider("flaky", flaky)

    events: list[dict[str, Any]] = []
    register_hook(ON_RETRY, lambda **kw: events.append(kw))

    agent = Agent(_agent_config(retry_intervals=None))
    with pytest.raises(RuntimeError):
        await agent.call("hello")

    assert events == []


# ---------------------------------------------------------------------------
# Retry applies on every iteration, not just the first
# ---------------------------------------------------------------------------


async def test_retry_applies_on_every_iteration():
    """A multi-iteration loop retries the LLM call on each iteration."""
    inner = MockProvider(provider_name="openai")
    inner.queue(MockLLMResponse(content="done"))
    flaky = FlakyProvider(inner, fail_times=99, exc=RuntimeError("always"))
    register_provider("flaky", flaky)

    agent = Agent(_agent_config(retry_intervals=[1]))
    with pytest.raises(RuntimeError):
        await agent.call("hello")

    # Only the first iteration's create is attempted (it always fails), and
    # 1 retry on that same iteration exhausts the budget.
    assert flaky.create_call_count == 2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_agent_config_rejects_negative_interval():
    with pytest.raises(ValueError, match="non-negative"):
        AgentConfig(
            name="bad",
            model="m",
            provider="mock",
            retry_intervals=[1, -1, 3],
        )


def test_agent_config_allows_zero_interval():
    cfg = AgentConfig(
        name="ok",
        model="m",
        provider="mock",
        retry_intervals=[0, 0, 1],
    )
    assert cfg.retry_intervals == [0, 0, 1]


def test_agent_config_retry_intervals_default_none():
    cfg = AgentConfig(name="x", model="m", provider="mock")
    assert cfg.retry_intervals is None


# ---------------------------------------------------------------------------
# Preset propagation
# ---------------------------------------------------------------------------


def test_preset_retry_intervals_propagated_to_config():
    from coreouto.presets import (
        clear_agent_presets,
        register_agent_preset,
    )

    clear_agent_presets()
    try:
        preset = register_agent_preset(
            "retry-agent",
            model="m",
            provider="mock",
            retry_intervals=[1, 5, 10],
        )
        assert preset.retry_intervals == [1, 5, 10]
        config = preset.to_config()
        assert config.retry_intervals == [1, 5, 10]
    finally:
        clear_agent_presets()


def test_preset_retry_intervals_default_none():
    from coreouto.presets import (
        clear_agent_presets,
        register_agent_preset,
    )

    clear_agent_presets()
    try:
        preset = register_agent_preset("plain", model="m", provider="mock")
        assert preset.retry_intervals is None
        assert preset.to_config().retry_intervals is None
    finally:
        clear_agent_presets()


def test_preset_to_config_makes_copy_of_intervals():
    """to_config must not share the list instance with the preset."""
    from coreouto.presets import (
        clear_agent_presets,
        register_agent_preset,
    )

    clear_agent_presets()
    try:
        original = [1, 2, 3]
        preset = register_agent_preset("copy", model="m", provider="mock", retry_intervals=original)
        config = preset.to_config()
        config.retry_intervals.append(99)
        assert preset.retry_intervals == [1, 2, 3]
    finally:
        clear_agent_presets()
