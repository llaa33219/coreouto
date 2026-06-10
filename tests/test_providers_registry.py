from __future__ import annotations

import pytest

from coreouto.providers import (
    available_providers,
    clear_providers,
    get_provider,
    register_provider,
)
from tests.conftest import MockLLMResponse, MockProvider


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_providers()
    yield
    clear_providers()


def test_register_and_get():
    provider = MockProvider()
    register_provider("mock", provider)
    assert get_provider("mock") is provider


def test_re_register_overwrites():
    first = MockProvider()
    second = MockProvider()
    register_provider("mock", first)
    register_provider("mock", second)
    assert get_provider("mock") is second


def test_get_missing_raises_keyerror():
    with pytest.raises(KeyError, match="provider not registered"):
        get_provider("nonexistent")


def test_get_missing_includes_available_in_message():
    register_provider("alpha", MockProvider())
    with pytest.raises(KeyError, match=r"\['alpha'\]"):
        get_provider("beta")


def test_available_providers_sorted():
    register_provider("zebra", MockProvider())
    register_provider("alpha", MockProvider())
    register_provider("mid", MockProvider())
    assert available_providers() == ["alpha", "mid", "zebra"]


def test_available_providers_empty():
    assert available_providers() == []


def test_clear_providers():
    register_provider("a", MockProvider())
    register_provider("b", MockProvider())
    assert len(available_providers()) == 2
    clear_providers()
    assert available_providers() == []


def test_registered_provider_methods_callable():
    provider = MockProvider(
        responses=[MockLLMResponse(content="hello", prompt_tokens=10, completion_tokens=5)]
    )
    register_provider("mock", provider)
    retrieved = get_provider("mock")
    assert retrieved is provider
    assert hasattr(retrieved, "create")
    assert hasattr(retrieved, "format_assistant_message")
    assert hasattr(retrieved, "format_tool_result")


async def test_registered_provider_create_works():
    provider = MockProvider(
        responses=[MockLLMResponse(content="hi", prompt_tokens=3, completion_tokens=2)]
    )
    register_provider("mock", provider)
    retrieved = get_provider("mock")
    result = await retrieved.create(messages=[], model="test-model")
    assert result.content == "hi"
    assert result.usage.total_tokens == 5
