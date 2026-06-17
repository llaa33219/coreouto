from __future__ import annotations

import asyncio
import inspect
import typing

import pytest

from coreouto._types import Response
from coreouto.multi_agent import agent_as_tool, make_delegate_tool
from coreouto.presets import clear_agent_presets, register_agent_preset
from coreouto.providers import clear_providers, register_provider
from coreouto.tools import Tool, clear_tools
from tests.conftest import MockLLMResponse, MockProvider


@pytest.fixture(autouse=True)
def _clean_state() -> None:
    clear_agent_presets()
    clear_tools()
    clear_providers()
    yield
    clear_agent_presets()
    clear_tools()
    clear_providers()


def _register_simple_preset(name: str = "researcher") -> None:
    register_agent_preset(
        name,
        model="test-model",
        provider="mock",
        system_prompt="you research",
    )


def _register_mock_provider(name: str = "mock") -> MockProvider:
    provider = MockProvider()
    register_provider(name, provider)
    return provider


class TestAgentAsToolBasics:
    def test_returns_a_tool(self) -> None:
        _register_simple_preset()
        tool = agent_as_tool("researcher")
        assert isinstance(tool, Tool)

    def test_default_name_is_call_prefix(self) -> None:
        _register_simple_preset()
        tool = agent_as_tool("researcher")
        assert tool.name == "call_researcher"

    def test_default_description_mentions_preset_name(self) -> None:
        _register_simple_preset()
        tool = agent_as_tool("researcher")
        assert tool.description == (
            "Delegate a sub-task to the researcher agent. Input is the task description."
        )

    def test_custom_name_overrides_default(self) -> None:
        _register_simple_preset()
        tool = agent_as_tool("researcher", name="ask_researcher")
        assert tool.name == "ask_researcher"

    def test_custom_description_overrides_default(self) -> None:
        _register_simple_preset()
        tool = agent_as_tool(
            "researcher",
            description="Ask the researcher to dig into something.",
        )
        assert tool.description == "Ask the researcher to dig into something."

    def test_custom_name_and_description_together(self) -> None:
        _register_simple_preset()
        tool = agent_as_tool(
            "researcher",
            name="ask",
            description="ask it",
        )
        assert tool.name == "ask"
        assert tool.description == "ask it"


class TestAgentAsToolSchema:
    def test_schema_type_is_object(self) -> None:
        _register_simple_preset()
        tool = agent_as_tool("researcher")
        assert tool.parameters["type"] == "object"

    def test_schema_has_task_property_string(self) -> None:
        _register_simple_preset()
        tool = agent_as_tool("researcher")
        assert tool.parameters["properties"]["task"] == {
            "type": "string",
            "description": "The task description to pass to the sub-agent.",
        }

    def test_schema_requires_task(self) -> None:
        _register_simple_preset()
        tool = agent_as_tool("researcher")
        assert tool.parameters["required"] == ["task"]

    def test_schema_has_no_extra_properties(self) -> None:
        _register_simple_preset()
        tool = agent_as_tool("researcher")
        assert set(tool.parameters["properties"].keys()) == {"task"}


class TestAgentAsToolHandler:
    def test_handler_is_async_coroutine_function(self) -> None:
        _register_simple_preset()
        tool = agent_as_tool("researcher")
        assert inspect.iscoroutinefunction(tool.handler)

    def test_handler_signature_accepts_task_str_keyword(self) -> None:
        _register_simple_preset()
        tool = agent_as_tool("researcher")
        sig = inspect.signature(tool.handler)
        params = sig.parameters
        assert "task" in params
        hints = typing.get_type_hints(tool.handler)
        assert hints["task"] is str

    def test_handler_returns_str(self) -> None:
        _register_simple_preset()
        tool = agent_as_tool("researcher")
        hints = typing.get_type_hints(tool.handler)
        assert hints["return"] is str

    async def test_handler_returns_agent_final_content(self) -> None:
        _register_simple_preset()
        provider = _register_mock_provider()
        provider.queue(MockLLMResponse(content="research result"))

        tool = agent_as_tool("researcher")
        result = await tool.handler(task="find X")

        assert result == "research result"

    async def test_handler_passes_task_as_user_message(self) -> None:
        _register_simple_preset()
        provider = _register_mock_provider()
        provider.queue(MockLLMResponse(content="ok"))

        tool = agent_as_tool("researcher")
        await tool.handler(task="find fusion energy news")

        user_msgs = [m for m in provider.calls[0]["messages"] if getattr(m, "role", None) == "user"]
        assert user_msgs[0].content == "find fusion energy news"

    async def test_handler_uses_preset_provider(self) -> None:
        register_agent_preset(
            "writer",
            model="write-model",
            provider="mock",
        )
        provider = _register_mock_provider()
        provider.queue(MockLLMResponse(content="drafted"))

        tool = agent_as_tool("writer")
        await tool.handler(task="write a poem")

        assert provider.calls[0]["model"] == "write-model"

    async def test_handler_uses_preset_system_prompt(self) -> None:
        register_agent_preset(
            "researcher",
            model="m",
            provider="mock",
            system_prompt="be concise",
        )
        provider = _register_mock_provider()
        provider.queue(MockLLMResponse(content="ok"))

        tool = agent_as_tool("researcher")
        await tool.handler(task="hi")

        system_msgs = [
            m for m in provider.calls[0]["messages"] if getattr(m, "role", None) == "system"
        ]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "be concise"


class TestAgentAsToolErrors:
    def test_missing_preset_propagates_keyerror(self) -> None:
        with pytest.raises(KeyError):
            agent_as_tool("nonexistent_preset_xyz")

    def test_missing_preset_message_lists_available(self) -> None:
        _register_simple_preset("alpha_preset")
        _register_simple_preset("beta_preset")
        with pytest.raises(KeyError) as exc_info:
            agent_as_tool("missing_preset_zzz")
        msg = str(exc_info.value)
        assert "missing_preset_zzz" in msg
        assert "alpha_preset" in msg
        assert "beta_preset" in msg


class TestAgentAsToolIndependence:
    def test_each_call_returns_independent_agent(self) -> None:
        _register_simple_preset()
        tool1 = agent_as_tool("researcher")
        tool2 = agent_as_tool("researcher")
        assert tool1.handler is not tool2.handler
        assert tool1 is not tool2

    async def test_handlers_have_independent_state(self) -> None:
        _register_simple_preset()

        provider1 = MockProvider()
        provider1.queue(MockLLMResponse(content="from-a"))
        register_provider("mock", provider1)

        tool1 = agent_as_tool("researcher")
        result1 = await tool1.handler(task="a")

        provider2 = MockProvider()
        provider2.queue(MockLLMResponse(content="from-b"))
        register_provider("mock", provider2)

        tool2 = agent_as_tool("researcher")
        result2 = await tool2.handler(task="b")

        assert result1 == "from-a"
        assert result2 == "from-b"
        assert len(provider1.calls) == 1
        assert len(provider2.calls) == 1


class TestAgentAsToolConstruction:
    def test_handler_handler_does_not_register_globally(self) -> None:
        from coreouto.tools import get_tool

        _register_simple_preset()
        tool = agent_as_tool("researcher")
        assert get_tool(tool.name) is None

    def test_underlying_agent_is_agent_instance(self) -> None:
        _register_simple_preset()
        tool = agent_as_tool("researcher")
        assert asyncio.iscoroutinefunction(tool.handler)


class TestMakeDelegateTool:
    def test_returns_a_tool(self) -> None:
        tool = make_delegate_tool()
        assert isinstance(tool, Tool)

    def test_default_name_is_call_agent(self) -> None:
        tool = make_delegate_tool()
        assert tool.name == "call_agent"

    def test_custom_name_overrides_default(self) -> None:
        tool = make_delegate_tool(name="dispatcher")
        assert tool.name == "dispatcher"

    def test_default_description_is_set(self) -> None:
        tool = make_delegate_tool()
        assert tool.description == (
            "Call a registered agent by name and pass it a message. "
            "The agent's response is returned as a string."
        )

    def test_custom_description_overrides_default(self) -> None:
        tool = make_delegate_tool(description="delegate to a sub-agent")
        assert tool.description == "delegate to a sub-agent"

    def test_schema_type_is_object(self) -> None:
        tool = make_delegate_tool()
        assert tool.parameters["type"] == "object"

    def test_schema_has_agent_name_and_message_as_strings(self) -> None:
        tool = make_delegate_tool()
        props = tool.parameters["properties"]
        assert props["agent_name"]["type"] == "string"
        assert props["message"]["type"] == "string"

    def test_schema_has_no_extra_properties(self) -> None:
        tool = make_delegate_tool()
        assert set(tool.parameters["properties"].keys()) == {"agent_name", "message"}

    def test_schema_marks_both_required(self) -> None:
        tool = make_delegate_tool()
        assert set(tool.parameters["required"]) == {"agent_name", "message"}

    def test_handler_is_async_coroutine_function(self) -> None:
        tool = make_delegate_tool()
        assert inspect.iscoroutinefunction(tool.handler)

    def test_handler_signature_accepts_agent_name_and_message(self) -> None:
        tool = make_delegate_tool()
        sig = inspect.signature(tool.handler)
        params = sig.parameters
        assert "agent_name" in params
        assert "message" in params
        hints = typing.get_type_hints(tool.handler)
        assert hints["agent_name"] is str
        assert hints["message"] is str
        assert hints["return"] is str

    def test_does_not_register_globally(self) -> None:
        from coreouto.tools import get_tool

        tool = make_delegate_tool()
        assert get_tool(tool.name) is None

    async def test_dispatches_to_named_agent(self) -> None:
        register_agent_preset("alice", model="alice-model", provider="alice_prov")
        register_agent_preset("bob", model="bob-model", provider="bob_prov")

        alice_provider = MockProvider()
        alice_provider.queue(MockLLMResponse(content="alice says hi"))
        register_provider("alice_prov", alice_provider)

        bob_provider = MockProvider()
        bob_provider.queue(MockLLMResponse(content="bob says hi"))
        register_provider("bob_prov", bob_provider)

        tool = make_delegate_tool()
        result = await tool.handler(agent_name="alice", message="hi")

        assert result == "alice says hi"
        assert len(alice_provider.calls) == 1
        assert len(bob_provider.calls) == 0

    async def test_dispatches_to_different_agents_independently(self) -> None:
        register_agent_preset("alice", model="alice-model", provider="alice_prov")
        register_agent_preset("bob", model="bob-model", provider="bob_prov")

        alice_provider = MockProvider()
        alice_provider.queue(MockLLMResponse(content="alice reply"))
        register_provider("alice_prov", alice_provider)

        bob_provider = MockProvider()
        bob_provider.queue(MockLLMResponse(content="bob reply"))
        register_provider("bob_prov", bob_provider)

        tool = make_delegate_tool()
        result_alice = await tool.handler(agent_name="alice", message="hi alice")
        result_bob = await tool.handler(agent_name="bob", message="hi bob")

        assert result_alice == "alice reply"
        assert result_bob == "bob reply"
        assert alice_provider.calls[0]["model"] == "alice-model"
        assert bob_provider.calls[0]["model"] == "bob-model"
        alice_user_msgs = [
            m for m in alice_provider.calls[0]["messages"] if getattr(m, "role", None) == "user"
        ]
        bob_user_msgs = [
            m for m in bob_provider.calls[0]["messages"] if getattr(m, "role", None) == "user"
        ]
        assert alice_user_msgs[0].content == "hi alice"
        assert bob_user_msgs[0].content == "hi bob"

    async def test_missing_preset_raises_keyerror(self) -> None:
        tool = make_delegate_tool()
        with pytest.raises(KeyError):
            await tool.handler(agent_name="nonexistent_zzz", message="hi")

    async def test_returns_string_not_response(self) -> None:
        register_agent_preset("alice", model="m", provider="mock")
        provider = _register_mock_provider()
        provider.queue(MockLLMResponse(content="string reply"))

        tool = make_delegate_tool()
        result = await tool.handler(agent_name="alice", message="hi")

        assert isinstance(result, str)
        assert not isinstance(result, Response)
        assert result == "string reply"
