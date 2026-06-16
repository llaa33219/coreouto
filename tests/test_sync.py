"""Tests for the synchronous call wrapper."""

from __future__ import annotations

import pytest

from coreouto._types import AgentConfig
from coreouto.agent import Agent, MaxIterationsError
from coreouto.hooks import clear_hooks
from coreouto.providers import clear_providers, register_provider
from coreouto.tools import clear_tools
from tests.conftest import MockLLMResponse, MockProvider


@pytest.fixture(autouse=True)
def _clean_state():
    clear_tools()
    clear_hooks()
    clear_providers()
    yield
    clear_tools()
    clear_hooks()
    clear_providers()


def test_call_sync_returns_same_response_as_async():
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "finish_1", "name": "finish", "arguments": {"content": "done"}}]
        )
    )
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", max_iterations=10))
    response = agent.call_sync("hello")

    assert response.content == "done"
    assert response.iterations == 1
    assert response.finish_called is True
    assert len(response.messages) == 3
    assert response.messages[0].role == "system"
    assert response.messages[1].role == "user"
    assert response.messages[1].content == "hello"
    assert response.messages[2].role == "assistant"


async def test_call_sync_inside_running_loop_raises():
    provider = MockProvider()
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", max_iterations=10))
    with pytest.raises(
        RuntimeError,
        match=r"call_sync\(\) cannot be used inside a running event loop",
    ):
        agent.call_sync("hello")


def test_call_sync_with_override():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="a"))
    provider.queue(MockLLMResponse(content="b"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", max_iterations=5))
    with pytest.raises(MaxIterationsError, match=r"max_iterations \(2\) reached"):
        agent.call_sync(
            "hello",
            override=AgentConfig(name="ovr", model="m", provider="mock", max_iterations=2),
        )


def test_call_sync_raises_max_iterations_error():
    provider = MockProvider()
    for _ in range(3):
        provider.queue(MockLLMResponse(content="thinking..."))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", max_iterations=2))
    with pytest.raises(MaxIterationsError, match=r"max_iterations \(2\) reached"):
        agent.call_sync("hello")


def test_call_sync_propagates_keyerror_for_missing_tool():
    register_provider("mock", MockProvider())
    agent = Agent(AgentConfig(name="test", model="m", provider="mock", tools=["nonexistent"]))
    with pytest.raises(KeyError, match=r"tool not registered: 'nonexistent'"):
        agent.call_sync("hello")


def test_call_sync_propagates_keyerror_for_missing_provider():
    agent = Agent(AgentConfig(name="test", model="m", provider="nonexistent"))
    with pytest.raises(KeyError, match="provider not registered"):
        agent.call_sync("hello")


def test_call_sync_multiple_sequential_calls():
    provider = MockProvider()
    for i in range(3):
        provider.queue(
            MockLLMResponse(
                tool_calls=[
                    {
                        "id": "finish_1",
                        "name": "finish",
                        "arguments": {"content": f"done{i}"},
                    }
                ]
            )
        )
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", max_iterations=10))

    response1 = agent.call_sync("hello")
    assert response1.content == "done0"

    response2 = agent.call_sync("hello again")
    assert response2.content == "done1"

    response3 = agent.call_sync("hello once more")
    assert response3.content == "done2"

    assert len(provider.calls) == 3
