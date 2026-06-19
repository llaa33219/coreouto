from __future__ import annotations

import asyncio

import pytest

from coreouto._types import (
    AgentConfig,
    ImageBlock,
    Message,
    TextBlock,
    ToolCall,
    ToolResult,
)
from coreouto.agent import _DEFAULT_SYSTEM_PROMPT, Agent, MaxIterationsError
from coreouto.hooks import (
    AFTER_LLM_CALL,
    AFTER_TOOL_CALL,
    BEFORE_LLM_CALL,
    BEFORE_TOOL_CALL,
    ON_FINISH,
    ON_ITERATION,
    clear_hooks,
    register_hook,
)
from coreouto.providers import clear_providers, register_provider
from coreouto.tools import clear_tools, register_tool
from tests.conftest import HookRecorder, MockLLMResponse, MockProvider


@pytest.fixture(autouse=True)
def _clean_state():
    clear_tools()
    clear_hooks()
    clear_providers()
    yield
    clear_tools()
    clear_hooks()
    clear_providers()


def _terminate_response(content: str | None = None) -> MockLLMResponse:
    """Build a response that terminates the agent loop.

    The `MockProvider` auto-injects a `finish` tool call (since the
    `terminate` default is True) so the loop closes under coreouto's
    model-driven policy. The response's `stop_reason` is left as `None`
    so the conftest fixture can fill in the provider's natural end-of-turn
    value (`end_turn` for Anthropic, `stop` for OpenAI Chat, `completed`
    for OpenAI Responses, `STOP` for Gemini).
    """
    return MockLLMResponse(content=content or "")


def _continue_loop_tool_call(content: str, call_id: str = "cl_1") -> dict:
    return {"id": call_id, "name": "continue_loop", "arguments": {"content": content}}


async def test_finish_tool_terminates_loop_with_content_as_final_answer():
    provider = MockProvider()
    provider.queue(_terminate_response("done"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", max_iterations=10))
    response = await agent.call("hello")

    # The model calls `finish` to terminate; the `content` argument of the
    # `finish` call is the canonical final answer (NOT `response.content`).
    assert response.content == "done"
    assert response.iterations == 1
    assert response.stop_reason == "finish"
    assert len(response.messages) == 3
    assert response.messages[0].role == "system"
    assert response.messages[1].role == "user"
    assert response.messages[1].content == "hello"
    assert response.messages[2].role == "assistant"
    # The assistant message now carries the `finish` tool call.
    assert response.messages[2].tool_calls is not None
    finish_call = response.messages[2].tool_calls[0]
    assert finish_call.name == "finish"
    assert finish_call.arguments["content"] == "done"


async def test_multi_iteration_tool_then_text():
    side_effect: list[str] = []

    @register_tool("echo")
    def echo(msg: str) -> str:
        side_effect.append(msg)
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}])
    )
    provider.queue(_terminate_response("done"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", tools=["echo"]))
    response = await agent.call("hello")

    assert side_effect == ["hi"]
    assert response.content == "done"
    assert response.iterations == 2


async def test_multi_iteration_two_tools_in_one_response():
    calls: list[tuple[str, str]] = []

    @register_tool("tool_a")
    def tool_a(x: str) -> str:
        calls.append(("a", x))
        return f"a-{x}"

    @register_tool("tool_b")
    def tool_b(y: str) -> str:
        calls.append(("b", y))
        return f"b-{y}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[
                {"id": "tc1", "name": "tool_a", "arguments": {"x": "1"}},
                {"id": "tc2", "name": "tool_b", "arguments": {"y": "2"}},
            ]
        )
    )
    provider.queue(_terminate_response("done"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", tools=["tool_a", "tool_b"]))
    response = await agent.call("hello")

    assert calls == [("a", "1"), ("b", "2")]
    assert response.iterations == 2


async def test_max_iterations_error():
    @register_tool("think")
    def think() -> str:
        return "thought"

    provider = MockProvider()
    for _ in range(3):
        provider.queue(MockLLMResponse(tool_calls=[{"id": "t", "name": "think", "arguments": {}}]))
    register_provider("mock", provider)

    agent = Agent(
        AgentConfig(name="test", model="m", provider="mock", max_iterations=2, tools=["think"])
    )
    with pytest.raises(
        MaxIterationsError,
        match=r"max_iterations \(2\) reached without terminating the loop",
    ):
        await agent.call("hello")


async def test_system_prompt_injected_as_first_message():
    provider = MockProvider()
    provider.queue(_terminate_response("done"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", system_prompt="you are X"))
    response = await agent.call("hello")

    assert response.content == "done"
    first_call = provider.calls[0]
    messages = first_call["messages"]
    assert messages[0].role == "system"
    assert messages[0].content == "you are X"
    assert messages[1].role == "user"
    assert messages[1].content == "hello"


async def test_hook_firing_order():
    recorder = HookRecorder()
    factory = recorder.make()

    for event in [
        BEFORE_LLM_CALL,
        AFTER_LLM_CALL,
        ON_ITERATION,
        BEFORE_TOOL_CALL,
        AFTER_TOOL_CALL,
        ON_FINISH,
    ]:
        register_hook(event, factory(event))

    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}])
    )
    provider.queue(_terminate_response("done"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", tools=["echo"]))
    await agent.call("hello")

    events = [e[0] for e in recorder.events]
    expected = [
        BEFORE_LLM_CALL,
        AFTER_LLM_CALL,
        ON_ITERATION,
        BEFORE_TOOL_CALL,
        AFTER_TOOL_CALL,
        BEFORE_LLM_CALL,
        AFTER_LLM_CALL,
        ON_ITERATION,
        ON_FINISH,
    ]
    assert events == expected


async def test_override_at_call_time():
    @register_tool("think")
    def think() -> str:
        return "thought"

    provider = MockProvider()
    provider.queue(MockLLMResponse(tool_calls=[{"id": "t", "name": "think", "arguments": {}}]))
    provider.queue(MockLLMResponse(tool_calls=[{"id": "t", "name": "think", "arguments": {}}]))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", max_iterations=5))
    with pytest.raises(
        MaxIterationsError,
        match=r"max_iterations \(2\) reached without terminating the loop",
    ):
        await agent.call(
            "hello",
            override=AgentConfig(
                name="ovr", model="m", provider="mock", max_iterations=2, tools=["think"]
            ),
        )


async def test_tool_error_surfaces_as_is_error():
    @register_tool("boom")
    def boom() -> str:
        raise RuntimeError("exploded")

    provider = MockProvider()
    provider.queue(MockLLMResponse(tool_calls=[{"id": "tc1", "name": "boom", "arguments": {}}]))
    provider.queue(_terminate_response("done"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", tools=["boom"]))
    await agent.call("hello")

    _tc, result = provider.formatted_tool_results[0]
    assert isinstance(result, ToolResult)
    assert result.is_error is True
    assert "RuntimeError: exploded" in result.content


async def test_missing_tool_raises_keyerror():
    register_provider("mock", MockProvider())
    agent = Agent(AgentConfig(name="test", model="m", provider="mock", tools=["nonexistent"]))
    with pytest.raises(KeyError, match=r"tool not registered: 'nonexistent'"):
        await agent.call("hello")


async def test_missing_provider_raises_keyerror():
    agent = Agent(AgentConfig(name="test", model="m", provider="nonexistent"))
    with pytest.raises(KeyError, match="provider not registered"):
        await agent.call("hello")


async def test_async_tool_handler():
    @register_tool("async_echo")
    async def async_echo(msg: str) -> str:
        return f"async-{msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "async_echo", "arguments": {"msg": "hi"}}]
        )
    )
    provider.queue(_terminate_response("done"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", tools=["async_echo"]))
    await agent.call("hello")

    _tc, result = provider.formatted_tool_results[0]
    assert result.content == "async-hi"


async def test_usage_tracking():
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            content="done",
            prompt_tokens=10,
            completion_tokens=5,
        )
    )
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    response = await agent.call("hello")

    assert len(response.usage) == 1
    assert response.usage[0].prompt_tokens == 10
    assert response.usage[0].completion_tokens == 5
    assert response.usage[0].total_tokens == 15


async def test_usage_tracking_multiple_iterations():
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "noop", "arguments": {}}],
            prompt_tokens=5,
            completion_tokens=3,
        )
    )
    provider.queue(
        MockLLMResponse(
            content="done",
            prompt_tokens=8,
            completion_tokens=4,
        )
    )
    register_provider("mock", provider)

    @register_tool("noop")
    def noop() -> str:
        return "ok"

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", tools=["noop"]))
    response = await agent.call("hello")

    assert len(response.usage) == 2
    assert response.usage[0].prompt_tokens == 5
    assert response.usage[0].completion_tokens == 3
    assert response.usage[1].prompt_tokens == 8
    assert response.usage[1].completion_tokens == 4


async def test_agent_store_config_and_provider_name():
    cfg = AgentConfig(name="test", model="m", provider="mock")
    agent = Agent(cfg)
    assert agent.config is cfg
    assert agent.provider_name == "mock"


async def test_max_iterations_zero():
    @register_tool("think")
    def think() -> str:
        return "thought"

    provider = MockProvider()
    provider.queue(MockLLMResponse(tool_calls=[{"id": "t", "name": "think", "arguments": {}}]))
    register_provider("mock", provider)

    agent = Agent(
        AgentConfig(name="test", model="m", provider="mock", max_iterations=0, tools=["think"])
    )
    with pytest.raises(
        MaxIterationsError,
        match=r"max_iterations \(0\) reached without terminating the loop",
    ):
        await agent.call("hello")


async def test_provider_config_forwarded_to_provider():
    provider = MockProvider()
    provider.queue(_terminate_response("done"))
    register_provider("mock", provider)

    agent = Agent(
        AgentConfig(
            name="test",
            model="m",
            provider="mock",
            provider_config={"temperature": 0.5, "max_tokens": 100},
        )
    )
    await agent.call("hello")

    assert provider.calls[0]["kwargs"] == {"temperature": 0.5, "max_tokens": 100}


async def test_max_iterations_raises_when_loop_never_terminates():
    @register_tool("think")
    def think() -> str:
        return "thought"

    provider = MockProvider()
    provider.queue(MockLLMResponse(tool_calls=[{"id": "t", "name": "think", "arguments": {}}]))
    provider.queue(MockLLMResponse(tool_calls=[{"id": "t", "name": "think", "arguments": {}}]))
    provider.queue(MockLLMResponse(tool_calls=[{"id": "t", "name": "think", "arguments": {}}]))
    register_provider("mock", provider)

    agent = Agent(
        AgentConfig(name="test", model="m", provider="mock", max_iterations=2, tools=["think"])
    )
    with pytest.raises(
        MaxIterationsError,
        match=r"max_iterations \(2\) reached without terminating the loop",
    ):
        await agent.call("hello")


async def test_max_tokens_translates_per_provider():
    for pname, expected_key in (
        ("openai", "max_tokens"),
        ("openai-response", "max_output_tokens"),
        ("anthropic", "max_tokens"),
        ("google", "max_output_tokens"),
    ):
        provider = MockProvider()
        provider.queue(_terminate_response("done"))
        register_provider(pname, provider)

        agent = Agent(
            AgentConfig(
                name="test",
                model="m",
                provider=pname,
                provider_config={"max_tokens": 100},
            )
        )
        await agent.call("hello")

        assert provider.calls[0]["kwargs"] == {expected_key: 100}
        clear_providers()


async def test_reasoning_effort_forwards_per_provider():
    cases = [
        (
            "openai-response",
            {"reasoning_effort": "medium"},
            {"reasoning": {"effort": "medium"}},
        ),
        (
            "anthropic",
            {"reasoning_effort": "high"},
            {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
            },
        ),
    ]
    for pname, provider_config, expected_kwargs in cases:
        provider = MockProvider()
        provider.queue(_terminate_response("done"))
        register_provider(pname, provider)

        agent = Agent(
            AgentConfig(
                name="test",
                model="m",
                provider=pname,
                provider_config=provider_config,
            )
        )
        await agent.call("hello")

        assert provider.calls[0]["kwargs"] == expected_kwargs, pname
        clear_providers()


async def test_reasoning_effort_anthropic_none_drops_kwarg():
    provider = MockProvider()
    provider.queue(_terminate_response("done"))
    register_provider("anthropic", provider)

    agent = Agent(
        AgentConfig(
            name="test",
            model="m",
            provider="anthropic",
            provider_config={"reasoning_effort": "none"},
        )
    )
    await agent.call("hello")

    assert provider.calls[0]["kwargs"] == {}


async def test_provider_passthrough_merges_with_normalized():
    provider = MockProvider()
    provider.queue(_terminate_response("done"))
    register_provider("openai", provider)

    agent = Agent(
        AgentConfig(
            name="test",
            model="m",
            provider="openai",
            provider_config={"temperature": 0.5},
            provider_passthrough={"top_logprobs": 5},
        )
    )
    await agent.call("hello")

    assert provider.calls[0]["kwargs"] == {"temperature": 0.5, "top_logprobs": 5}


async def test_provider_passthrough_overrides_normalized_on_conflict():
    provider = MockProvider()
    provider.queue(_terminate_response("done"))
    register_provider("openai", provider)

    agent = Agent(
        AgentConfig(
            name="test",
            model="m",
            provider="openai",
            provider_config={"max_tokens": 100},
            provider_passthrough={"max_output_tokens": 200},
        )
    )
    await agent.call("hello")

    assert provider.calls[0]["kwargs"] == {"max_tokens": 100, "max_output_tokens": 200}


async def test_unknown_provider_config_key_raises():
    register_provider("openai", MockProvider())
    agent = Agent(
        AgentConfig(
            name="test",
            model="m",
            provider="openai",
            provider_config={"response_format": {"type": "json_object"}},
        )
    )
    with pytest.raises(ValueError, match=r"unknown provider_config key 'response_format'"):
        await agent.call("hello")


async def test_text_only_response_terminates_loop():
    """A response with only text (no tool call) terminates the loop. The text
    becomes the final answer. No `finish` tool call needed.
    """
    provider = MockProvider()
    provider.queue(_terminate_response("just text"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", max_iterations=2))
    response = await agent.call("hello")

    assert response.content == "just text"
    assert response.iterations == 1
    assert response.stop_reason == "finish"


async def test_empty_response_terminates_loop_with_empty_content():
    """An empty response (no content, no tool calls) terminates the loop with
    empty content.
    """
    provider = MockProvider()
    provider.queue(_terminate_response(""))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", max_iterations=2))
    response = await agent.call("hello")

    assert response.content == ""
    assert response.iterations == 1
    assert response.stop_reason == "finish"


async def test_history_prepended_to_messages():
    provider = MockProvider()
    provider.queue(_terminate_response("ok"))
    register_provider("mock", provider)

    history = [
        Message(role="user", content="earlier question"),
        Message(role="assistant", content="earlier answer"),
    ]
    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    await agent.call("new question", history=history)

    messages = provider.calls[0]["messages"]
    assert len(messages) == 4
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert messages[1].content == "earlier question"
    assert messages[2].role == "assistant"
    assert messages[2].content == "earlier answer"
    assert messages[3].role == "user"
    assert messages[3].content == "new question"


async def test_history_none_default_preserves_existing_behavior():
    provider = MockProvider()
    provider.queue(_terminate_response("ok"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    await agent.call("hello")

    messages = provider.calls[0]["messages"]
    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert messages[1].content == "hello"


async def test_history_empty_list_equals_none():
    provider = MockProvider()
    provider.queue(_terminate_response("ok"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    await agent.call("hello", history=[])

    messages = provider.calls[0]["messages"]
    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[1].content == "hello"


async def test_history_with_system_prompt_prepends_cfg_first():
    provider = MockProvider()
    provider.queue(_terminate_response("ok"))
    register_provider("mock", provider)

    history = [
        Message(role="user", content="earlier"),
        Message(role="assistant", content="answer"),
    ]
    agent = Agent(AgentConfig(name="test", model="m", provider="mock", system_prompt="you are X"))
    await agent.call("new", history=history)

    messages = provider.calls[0]["messages"]
    assert len(messages) == 4
    assert messages[0].role == "system"
    assert messages[0].content == "you are X"
    assert messages[1].role == "user"
    assert messages[1].content == "earlier"
    assert messages[2].role == "assistant"
    assert messages[2].content == "answer"
    assert messages[3].role == "user"
    assert messages[3].content == "new"


async def test_history_can_be_fabricated():
    provider = MockProvider()
    captured_messages: list = []

    original_create = provider.create

    async def capturing_create(messages, **kwargs):
        captured_messages.append(list(messages))
        return await original_create(messages, **kwargs)

    provider.create = capturing_create
    provider.queue(_terminate_response("ok"))
    register_provider("mock", provider)

    fake_history = [
        Message(role="user", content="fabricated question 1"),
        Message(role="assistant", content="fabricated answer 1"),
        Message(role="user", content="fabricated question 2"),
        Message(role="assistant", content="fabricated answer 2"),
    ]
    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    await agent.call("real question", history=fake_history)

    sent = captured_messages[0]
    assert sent[0].role == "system"
    non_system = [m for m in sent if m.role != "system"]
    assert [m.content for m in non_system] == [
        "fabricated question 1",
        "fabricated answer 1",
        "fabricated question 2",
        "fabricated answer 2",
        "real question",
    ]


async def test_history_preserves_assistant_tool_calls():
    provider = MockProvider()
    provider.queue(_terminate_response("ok"))
    register_provider("mock", provider)

    history = [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="hist_tc1", name="search", arguments={"q": "x"}),
            ],
        ),
        Message(role="tool", content="search result", tool_call_id="hist_tc1", name="search"),
    ]
    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    await agent.call("continue", history=history)

    messages = provider.calls[0]["messages"]
    assert messages[0].role == "system"
    assert messages[1].role == "assistant"
    assert messages[1].tool_calls is not None
    assert messages[1].tool_calls[0].name == "search"
    assert messages[2].role == "tool"
    assert messages[2].tool_call_id == "hist_tc1"


def test_call_sync_passes_history_through():
    from coreouto.sync import call_sync

    provider = MockProvider()
    provider.queue(_terminate_response("ok"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    history = [Message(role="user", content="prev"), Message(role="assistant", content="ans")]
    response = call_sync(agent, "new", history=history)
    assert response.content == "ok"
    sent = provider.calls[0]["messages"]
    assert [m.content for m in sent] == [_DEFAULT_SYSTEM_PROMPT, "prev", "ans", "new"]


async def test_history_works_with_override():
    provider = MockProvider()
    provider.queue(_terminate_response("ok"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", system_prompt="default"))
    history = [Message(role="user", content="q1"), Message(role="assistant", content="a1")]
    override = AgentConfig(
        name="ovr",
        model="m",
        provider="mock",
        system_prompt="override prompt",
    )
    await agent.call("q2", history=history, override=override)

    messages = provider.calls[0]["messages"]
    assert messages[0].role == "system"
    assert messages[0].content == "override prompt"
    assert messages[1].content == "q1"
    assert messages[2].content == "a1"
    assert messages[3].content == "q2"


async def test_inject_user_message_basic():
    @register_tool("noop")
    def noop() -> str:
        return "ok"

    provider = MockProvider()
    provider.queue(MockLLMResponse(tool_calls=[{"id": "t1", "name": "noop", "arguments": {}}]))
    provider.queue(_terminate_response("ok"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", tools=["noop"]))
    agent.inject_user_message("injected mid-loop")
    response = await agent.call("initial")

    sent = provider.calls[1]["messages"]
    contents = [m.content for m in sent if m.role == "user"]
    assert "initial" in contents
    assert "injected mid-loop" in contents
    assert response.content == "ok"


async def test_inject_user_message_fires_hook():
    @register_tool("noop")
    def noop() -> str:
        return "ok"

    provider = MockProvider()
    provider.queue(MockLLMResponse(tool_calls=[{"id": "t1", "name": "noop", "arguments": {}}]))
    provider.queue(_terminate_response("ok"))
    register_provider("mock", provider)

    injection_events: list[Message] = []

    def capture(message, **_):
        injection_events.append(message)

    register_hook("on_user_injection", capture)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", tools=["noop"]))
    agent.inject_user_message("hi from outside")
    await agent.call("hi from caller")

    assert len(injection_events) == 1
    assert injection_events[0].role == "user"
    assert injection_events[0].content == "hi from outside"


async def test_inject_multiple_messages_all_drained_in_one_iteration():
    @register_tool("noop")
    def noop() -> str:
        return "ok"

    provider = MockProvider()
    provider.queue(MockLLMResponse(tool_calls=[{"id": "t1", "name": "noop", "arguments": {}}]))
    provider.queue(_terminate_response("ok"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", tools=["noop"]))
    agent.inject_user_message("first")
    agent.inject_user_message("second")
    agent.inject_user_message("third")
    await agent.call("initial")

    sent = provider.calls[1]["messages"]
    user_contents = [m.content for m in sent if m.role == "user"]
    for expected in ["initial", "first", "second", "third"]:
        assert expected in user_contents, f"missing {expected!r} in {user_contents}"


async def test_inject_from_concurrent_task():
    import asyncio

    @register_tool("noop")
    def noop() -> str:
        return "ok"

    provider = MockProvider()
    provider.queue(MockLLMResponse(tool_calls=[{"id": "t1", "name": "noop", "arguments": {}}]))
    provider.queue(_terminate_response("ok"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", tools=["noop"]))

    async def inject_after_delay():
        await asyncio.sleep(0)
        agent.inject_user_message("from another task")

    task = asyncio.create_task(inject_after_delay())
    _ = task
    response = await agent.call("initial")

    sent = provider.calls[1]["messages"]
    user_contents = [m.content for m in sent if m.role == "user"]
    for expected in ["initial", "from another task"]:
        assert expected in user_contents, f"missing {expected!r} in {user_contents}"
    assert response.content == "ok"


async def test_no_injection_no_change():
    provider = MockProvider()
    provider.queue(_terminate_response("ok"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    await agent.call("only message")

    sent = provider.calls[0]["messages"]
    assert len(sent) == 2
    assert sent[0].role == "system"
    assert sent[1].content == "only message"


async def test_on_finish_hook_receives_extracted_content():
    received = []

    def hook(**kwargs):
        received.append(kwargs)

    register_hook(ON_FINISH, hook)

    provider = MockProvider()
    provider.queue(_terminate_response("my answer"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    response = await agent.call("hello")

    assert response.content == "my answer"
    assert len(received) == 1
    assert received[0]["content"] == "my answer"
    assert received[0]["iterations"] == 1
    assert "messages" in received[0]
    assert "tool_call_id" not in received[0]


async def test_tool_returning_list_of_content_blocks():
    @register_tool("show_image")
    async def show_image(label: str) -> list:
        return [
            TextBlock(text=f"Image for: {label}"),
            ImageBlock(data=b"\x89PNG\r\nfake-bytes", mime_type="image/png"),
        ]

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "show_image", "arguments": {"label": "cat"}}]
        )
    )
    provider.queue(_terminate_response("done"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    response = await agent.call("show me a cat")

    assert response.content == "done"
    assert len(provider.calls) == 2
    second_call_messages = provider.calls[1]["messages"]
    tool_msg = next(m for m in second_call_messages if m.role == "tool")
    assert tool_msg.tool_call_id == "tc1"
    assert isinstance(tool_msg.content, list)
    assert len(tool_msg.content) == 2
    assert isinstance(tool_msg.content[0], TextBlock)
    assert tool_msg.content[0].text == "Image for: cat"
    assert isinstance(tool_msg.content[1], ImageBlock)
    assert tool_msg.content[1].data == b"\x89PNG\r\nfake-bytes"
    assert tool_msg.content[1].mime_type == "image/png"


async def test_tool_returning_tool_result_with_blocks():
    @register_tool("fetch")
    async def fetch(q: str) -> ToolResult:
        return ToolResult(
            tool_call_id="",
            blocks=[ImageBlock(data=b"jpeg-bytes", mime_type="image/jpeg")],
        )

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(tool_calls=[{"id": "tc1", "name": "fetch", "arguments": {"q": "x"}}])
    )
    provider.queue(_terminate_response("ok"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    await agent.call("fetch")

    tool_msg = next(m for m in provider.calls[1]["messages"] if m.role == "tool")
    assert isinstance(tool_msg.content, list)
    assert tool_msg.content[0].data == b"jpeg-bytes"


async def test_tool_returning_plain_string_still_works():
    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"got: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}])
    )
    provider.queue(_terminate_response("done"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    await agent.call("test")

    tool_msg = next(m for m in provider.calls[1]["messages"] if m.role == "tool")
    assert tool_msg.content == "got: hi"


async def test_parallel_tool_calls_run_concurrently():
    """When parallel_tool_calls=True, two tool calls dispatched in the
    same LLM turn should run concurrently via asyncio.gather. We measure
    overlap with a shared barrier.
    """
    import time

    barrier = asyncio.Event()
    start: list[float] = []
    end: list[float] = []

    @register_tool("slow")
    async def slow() -> str:
        start.append(time.perf_counter())
        await barrier.wait()
        end.append(time.perf_counter())
        return "slow"

    @register_tool("fast")
    async def fast() -> str:
        start.append(time.perf_counter())
        barrier.set()
        end.append(time.perf_counter())
        return "fast"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[
                {"id": "tc_slow", "name": "slow", "arguments": {}},
                {"id": "tc_fast", "name": "fast", "arguments": {}},
            ]
        )
    )
    provider.queue(_terminate_response("done"))
    register_provider("mock", provider)

    agent = Agent(
        AgentConfig(
            name="test",
            model="m",
            provider="mock",
            tools=["slow", "fast"],
            parallel_tool_calls=True,
        )
    )
    await agent.call("go")

    assert len(start) == 2
    assert len(end) == 2
    # If parallel: both tools started before either ended. If sequential:
    # slow starts, then ends, then fast starts. We assert both started
    # before slow ended.
    assert end[0] - start[0] < 0.01 or end[1] - start[1] < 0.01, (
        "tools ran sequentially (no concurrency overlap)"
    )


async def test_sync_tools_dont_block_event_loop_when_parallel():
    """Two long-running sync tools should run in parallel via the thread
    pool. If they ran serially on the event loop, total time would be
    ~2x one tool's sleep. With to_thread offload, total time is ~1x.
    """
    import time

    @register_tool("sleep_a")
    def sleep_a() -> str:
        time.sleep(0.2)
        return "a"

    @register_tool("sleep_b")
    def sleep_b() -> str:
        time.sleep(0.2)
        return "b"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[
                {"id": "tc_a", "name": "sleep_a", "arguments": {}},
                {"id": "tc_b", "name": "sleep_b", "arguments": {}},
            ]
        )
    )
    provider.queue(_terminate_response("done"))
    register_provider("mock", provider)

    agent = Agent(
        AgentConfig(
            name="test",
            model="m",
            provider="mock",
            tools=["sleep_a", "sleep_b"],
            parallel_tool_calls=True,
        )
    )
    t0 = time.perf_counter()
    await agent.call("go")
    elapsed = time.perf_counter() - t0

    # Serial: ~0.4s. Parallel via to_thread: ~0.2s. Allow generous bound.
    assert elapsed < 0.35, f"sync tools ran serially: {elapsed:.3f}s"


async def test_non_parallelizable_tool_forces_serial_dispatch():
    """A single non-parallelizable tool in a turn forces the whole turn
    to run serially, even when parallel_tool_calls=True.
    """

    @register_tool("parallel_ok")
    async def parallel_ok() -> str:
        return "ok"

    @register_tool("must_be_alone", parallelizable=False)
    async def must_be_alone() -> str:
        return "alone"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[
                {"id": "tc1", "name": "parallel_ok", "arguments": {}},
                {"id": "tc2", "name": "must_be_alone", "arguments": {}},
            ]
        )
    )
    provider.queue(_terminate_response("done"))
    register_provider("mock", provider)

    agent = Agent(
        AgentConfig(
            name="test",
            model="m",
            provider="mock",
            tools=["parallel_ok", "must_be_alone"],
            parallel_tool_calls=True,
        )
    )
    response = await agent.call("go")

    # Both tools still ran (in serial order), and the loop terminated.
    assert response.content == "done"


async def test_parallel_tool_results_in_history_preserve_order():
    """When two tools run in parallel, their tool result messages in the
    conversation history should be in the same order as the model's
    tool_calls (not whatever order asyncio.gather happened to finish).
    """

    invocations: list[str] = []

    @register_tool("a")
    async def a() -> str:
        invocations.append("a-start")
        await asyncio.sleep(0.05)
        invocations.append("a-end")
        return "a-result"

    @register_tool("b")
    async def b() -> str:
        invocations.append("b-start")
        await asyncio.sleep(0.0)
        invocations.append("b-end")
        return "b-result"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[
                {"id": "tc_a", "name": "a", "arguments": {}},
                {"id": "tc_b", "name": "b", "arguments": {}},
            ]
        )
    )
    provider.queue(_terminate_response("done"))
    register_provider("mock", provider)

    agent = Agent(
        AgentConfig(
            name="test",
            model="m",
            provider="mock",
            tools=["a", "b"],
            parallel_tool_calls=True,
        )
    )
    await agent.call("go")

    # The second LLM call should have tool messages in input order (a, b)
    second_call = provider.calls[1]
    tool_messages = [m for m in second_call["messages"] if m.role == "tool"]
    assert len(tool_messages) == 2
    assert tool_messages[0].content == "a-result"
    assert tool_messages[1].content == "b-result"


async def test_parallel_default_off_preserves_legacy_sequential():
    """Default parallel_tool_calls=False keeps the legacy sequential
    behavior. Two tools run one after the other.
    """
    invocations: list[str] = []

    @register_tool("one")
    async def one() -> str:
        invocations.append("one")
        return "1"

    @register_tool("two")
    async def two() -> str:
        invocations.append("two")
        return "2"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[
                {"id": "tc1", "name": "one", "arguments": {}},
                {"id": "tc2", "name": "two", "arguments": {}},
            ]
        )
    )
    provider.queue(_terminate_response("done"))
    register_provider("mock", provider)

    agent = Agent(
        AgentConfig(
            name="test",
            model="m",
            provider="mock",
            tools=["one", "two"],
        )
    )
    await agent.call("go")

    assert invocations == ["one", "two"]


async def test_default_max_iterations_is_unlimited():
    """By default max_iterations is None, which means the loop never
    raises MaxIterationsError on iteration count alone. A text-only
    response terminates the loop normally.
    """
    provider = MockProvider()
    provider.queue(_terminate_response("ok"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    response = await agent.call("hi")
    assert response.content == "ok"
    assert response.stop_reason == "finish"


async def test_max_iterations_explicit_none_is_unlimited():
    """Setting max_iterations=None explicitly is the same as the default."""
    provider = MockProvider()
    provider.queue(_terminate_response("ok"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", max_iterations=None))
    response = await agent.call("hi")
    assert response.content == "ok"
    assert response.stop_reason == "finish"


async def test_max_iterations_finite_still_raises():
    """When max_iterations is a positive int, the loop still raises
    MaxIterationsError after that many iterations when the model keeps
    producing tool calls without terminating.
    """

    @register_tool("noop")
    def noop() -> str:
        return "ok"

    provider = MockProvider()
    for _ in range(5):
        provider.queue(MockLLMResponse(tool_calls=[{"id": "t", "name": "noop", "arguments": {}}]))
    register_provider("mock", provider)

    agent = Agent(
        AgentConfig(name="test", model="m", provider="mock", max_iterations=3, tools=["noop"])
    )
    with pytest.raises(
        MaxIterationsError,
        match=r"max_iterations \(3\) reached without terminating the loop",
    ):
        await agent.call("hi")


async def test_continue_loop_tool_call_with_other_tools_executes_all_then_continues():
    side_effect: list[str] = []

    @register_tool("echo")
    def echo(msg: str) -> str:
        side_effect.append(msg)
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[
                {"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}},
                _continue_loop_tool_call("working on it"),
            ]
        )
    )
    provider.queue(_terminate_response("echo: hi"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", tools=["echo"]))
    response = await agent.call("hello")

    assert side_effect == ["hi"]
    assert response.content == "echo: hi"
    assert response.iterations == 2
    second_call_messages = provider.calls[1]["messages"]
    tool_msgs = [m for m in second_call_messages if m.role == "tool"]
    tool_msg_names = [m.name for m in tool_msgs]
    assert "continue_loop" in tool_msg_names
    assert "echo" in tool_msg_names


async def test_continue_loop_tool_call_with_no_other_tools_continues_loop():
    provider = MockProvider()
    provider.queue(MockLLMResponse(tool_calls=[_continue_loop_tool_call("still working")]))
    provider.queue(_terminate_response("final answer"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    response = await agent.call("hello")

    assert response.iterations == 2
    assert response.content == "final answer"


async def test_continue_loop_result_appears_in_messages():
    provider = MockProvider()
    provider.queue(MockLLMResponse(tool_calls=[_continue_loop_tool_call("status update")]))
    provider.queue(_terminate_response("done"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    response = await agent.call("hello")

    tool_msgs = [m for m in response.messages if m.role == "tool" and m.name == "continue_loop"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == "status update"
    assert tool_msgs[0].tool_call_id == "cl_1"


async def test_response_stop_reason_default_is_finish():
    provider = MockProvider()
    provider.queue(_terminate_response("done"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    response = await agent.call("hello")

    assert response.stop_reason == "finish"


def test_agent_config_with_continue_loop_tool_raises():
    with pytest.raises(ValueError, match=r"reserved tool name"):
        AgentConfig(name="test", model="m", provider="mock", tools=["continue_loop"])


def test_agent_config_with_finish_tool_raises():
    with pytest.raises(ValueError, match=r"reserved tool name"):
        AgentConfig(name="test", model="m", provider="mock", tools=["finish"])


# ---------------------------------------------------------------------------
# Provider-driven loop termination
# ---------------------------------------------------------------------------
# The loop no longer ends purely on "no tool calls". Each provider has a
# native end-of-turn signal (stop_reason / finish_reason / status). The agent
# loop classifies that signal per provider; tool_calls is consulted only for
# providers whose API does not surface "I just called a tool" in stop_reason
# (Google Gemini, OpenAI Responses).


async def test_anthropic_tool_use_stop_reason_continues_loop():
    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}],
            stop_reason="tool_use",
            terminate=False,
        )
    )
    provider.queue(_terminate_response("done"))
    register_provider("anthropic", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="anthropic", tools=["echo"]))
    response = await agent.call("hello")

    assert response.iterations == 2
    assert response.content == "done"
    assert response.stop_reason == "finish"


async def test_anthropic_max_tokens_stop_reason_terminates_and_surfaces_stop_reason():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="partial", stop_reason="max_tokens"))
    register_provider("anthropic", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="anthropic"))
    response = await agent.call("hello")

    assert response.iterations == 1
    assert response.stop_reason == "max_tokens"


async def test_anthropic_refusal_stop_reason_terminates_and_surfaces_stop_reason():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="I cannot help.", stop_reason="refusal"))
    register_provider("anthropic", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="anthropic"))
    response = await agent.call("hello")

    assert response.stop_reason == "refusal"


async def test_anthropic_end_turn_with_continue_loop_tool_call_still_continues():
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[_continue_loop_tool_call("mid-task update")],
            stop_reason="tool_use",
        )
    )
    provider.queue(MockLLMResponse(content="done", stop_reason="end_turn"))
    register_provider("anthropic", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="anthropic"))
    response = await agent.call("hello")

    assert response.iterations == 2
    assert response.content == "done"


async def test_openai_chat_tool_calls_finish_reason_continues_loop():
    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}],
            stop_reason="tool_calls",
        )
    )
    provider.queue(MockLLMResponse(content="done", stop_reason="stop"))
    register_provider("openai", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="openai", tools=["echo"]))
    response = await agent.call("hello")

    assert response.iterations == 2
    assert response.content == "done"


async def test_openai_chat_length_finish_reason_terminates_and_surfaces_stop_reason():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="partial", stop_reason="length"))
    register_provider("openai", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="openai"))
    response = await agent.call("hello")

    assert response.stop_reason == "length"


async def test_openai_chat_content_filter_finish_reason_terminates_and_surfaces_stop_reason():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="filtered", stop_reason="content_filter"))
    register_provider("openai", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="openai"))
    response = await agent.call("hello")

    assert response.stop_reason == "content_filter"


async def test_openai_responses_completed_with_tool_calls_continues_loop():
    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}],
            stop_reason="completed",
        )
    )
    provider.queue(MockLLMResponse(content="done", stop_reason="completed"))
    register_provider("openai-response", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="openai-response", tools=["echo"]))
    response = await agent.call("hello")

    assert response.iterations == 2
    assert response.content == "done"


async def test_openai_responses_completed_without_tool_calls_terminates():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="done", stop_reason="completed"))
    register_provider("openai-response", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="openai-response"))
    response = await agent.call("hello")

    assert response.iterations == 1
    assert response.stop_reason == "finish"


async def test_openai_responses_incomplete_terminates_and_surfaces_stop_reason():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="partial", stop_reason="incomplete:max_output_tokens"))
    register_provider("openai-response", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="openai-response"))
    response = await agent.call("hello")

    assert response.stop_reason == "incomplete"


async def test_google_stop_finish_reason_with_tool_calls_continues_loop():
    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}],
            stop_reason="STOP",
        )
    )
    provider.queue(MockLLMResponse(content="done", stop_reason="STOP"))
    register_provider("google", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="google", tools=["echo"]))
    response = await agent.call("hello")

    assert response.iterations == 2
    assert response.content == "done"


async def test_google_stop_without_tool_calls_terminates():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="done", stop_reason="STOP"))
    register_provider("google", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="google"))
    response = await agent.call("hello")

    assert response.iterations == 1
    assert response.stop_reason == "finish"


async def test_google_safety_finish_reason_terminates_even_with_tool_calls():
    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}],
            stop_reason="SAFETY",
        )
    )
    register_provider("google", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="google", tools=["echo"]))
    response = await agent.call("hello")

    # SAFETY must override any tool_calls — refuse to execute unsafe tool calls.
    assert response.iterations == 1
    assert response.stop_reason == "finish"


async def test_google_max_tokens_finish_reason_terminates_with_max_tokens_literal():
    # Gemini's MAX_TOKENS finish_reason is surfaced on Response.stop_reason
    # as the "max_tokens" literal (mirroring Anthropic) so callers can
    # distinguish a length cap from a clean finish.
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="partial", stop_reason="MAX_TOKENS"))
    register_provider("google", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="google"))
    response = await agent.call("hello")

    assert response.stop_reason == "max_tokens"


async def test_unknown_provider_falls_back_to_no_tool_calls_rule():
    @register_tool("think")
    def think() -> str:
        return "ok"

    provider = MockProvider()
    provider.queue(MockLLMResponse(tool_calls=[{"id": "t", "name": "think", "arguments": {}}]))
    provider.queue(MockLLMResponse(content="done"))
    register_provider("custom", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="custom", tools=["think"]))
    response = await agent.call("hello")

    assert response.iterations == 2
    assert response.content == "done"


# ---------------------------------------------------------------------------
# Strict "if it's not END, keep going" policy
# ---------------------------------------------------------------------------
# The default rule for all four built-in providers: trust the provider's
# documented end-of-turn vocabulary. If the value is not in the END set, the
# loop continues regardless of any other signal. These tests exercise the
# edge cases where the response has unusual but valid stop_reason values.


async def test_anthropic_pause_turn_continues_loop():
    # `pause_turn` is a continuation signal: the server-side sampling loop
    # hit its iteration limit (default 10) while running server tools like
    # web search / web fetch. The official guidance is to send the
    # assistant response back as-is in a new request so the loop can
    # continue — see the Anthropic SDK docstring and SDK issue #1170.
    # coreouto treats it the same as `tool_use`: the loop should re-send
    # the response on the next iteration.
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="mid-turn", stop_reason="pause_turn", terminate=False))
    provider.queue(_terminate_response("done"))
    register_provider("anthropic", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="anthropic"))
    response = await agent.call("hello")

    assert response.iterations == 2
    assert response.content == "done"


async def test_anthropic_unknown_stop_reason_continues_loop():
    # Any unrecognized stop_reason on Anthropic is treated as CONTINUE
    # rather than risking a silent termination on a future API addition.
    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}],
            stop_reason="future_anthropic_value",
        )
    )
    provider.queue(MockLLMResponse(content="done"))
    register_provider("anthropic", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="anthropic", tools=["echo"]))
    response = await agent.call("hello")

    assert response.iterations == 2
    assert response.content == "done"


async def test_anthropic_none_stop_reason_with_tool_calls_continues():
    # A response with no stop_reason at all (e.g. legacy SDK version) and
    # tool calls present must continue the loop.
    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}],
            stop_reason=None,
        )
    )
    provider.queue(MockLLMResponse(content="done"))
    register_provider("anthropic", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="anthropic", tools=["echo"]))
    response = await agent.call("hello")

    assert response.iterations == 2
    assert response.content == "done"


async def test_openai_chat_function_call_legacy_continues_loop():
    # The deprecated `function_call` finish_reason is the singular
    # predecessor of the modern `tool_calls` field — it is a "I called a
    # function" signal, not a finish signal. coreouto treats it the same
    # as `tool_calls`: the loop continues so the function call can be
    # executed. Note that the legacy function_call is a singular
    # field on the message, not a list; a real OpenAI Chat provider
    # adapter would parse it in `format_assistant_message`. Here we
    # verify the loop policy: the finish_reason alone must not end the
    # loop when a tool call is present.
    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}],
            stop_reason="function_call",
        )
    )
    provider.queue(MockLLMResponse(content="done", stop_reason="stop"))
    register_provider("openai", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="openai", tools=["echo"]))
    response = await agent.call("hello")

    assert response.iterations == 2
    assert response.content == "done"


async def test_openai_chat_function_call_without_tool_calls_terminates():
    # `function_call` finish_reason with no actual tool call content is a
    # degenerate legacy response — nothing to execute, so the loop ends.
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="done", stop_reason="function_call"))
    register_provider("openai", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="openai"))
    response = await agent.call("hello")

    assert response.iterations == 1
    assert response.content == "done"


async def test_openai_chat_unknown_finish_reason_continues_loop():
    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}],
            stop_reason="future_openai_value",
        )
    )
    provider.queue(MockLLMResponse(content="done"))
    register_provider("openai", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="openai", tools=["echo"]))
    response = await agent.call("hello")

    assert response.iterations == 2
    assert response.content == "done"


async def test_openai_responses_cancelled_terminates_with_stop_reason():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="", stop_reason="cancelled"))
    register_provider("openai-response", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="openai-response"))
    response = await agent.call("hello")

    assert response.stop_reason == "cancelled"


async def test_openai_responses_failed_terminates_with_stop_reason():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="error", stop_reason="failed"))
    register_provider("openai-response", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="openai-response"))
    response = await agent.call("hello")

    assert response.stop_reason == "failed"


async def test_google_stop_without_tool_calls_terminates_with_text():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="hello", stop_reason="STOP"))
    register_provider("google", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="google"))
    response = await agent.call("hello")

    assert response.iterations == 1
    assert response.content == "hello"
    assert response.stop_reason == "finish"


async def test_google_unspecified_with_tool_calls_continues_loop():
    # `FINISH_REASON_UNSPECIFIED` is a placeholder value (the SDK
    # couldn't classify the finish reason). When a valid function_call
    # part is present the model did call a tool — coreouto should still
    # execute it rather than dropping it on the floor.
    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}],
            stop_reason="FINISH_REASON_UNSPECIFIED",
        )
    )
    provider.queue(MockLLMResponse(content="done", stop_reason="STOP"))
    register_provider("google", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="google", tools=["echo"]))
    response = await agent.call("hello")

    assert response.iterations == 2
    assert response.content == "done"


async def test_google_unspecified_without_tool_calls_terminates():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="partial", stop_reason="FINISH_REASON_UNSPECIFIED"))
    register_provider("google", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="google"))
    response = await agent.call("hello")

    assert response.iterations == 1
    assert response.stop_reason == "finish"


# ---------------------------------------------------------------------------
# Mid-loop truncation / recoverable-incomplete scenarios
# ---------------------------------------------------------------------------
# These exercise the "recoverable" branch of the END rule: the provider's
# stop field carries a value that USED to unconditionally end the loop, but
# coreouto now checks has_tool_calls first and executes the calls. The
# original bug (v0.6.0-v0.6.2) dropped tool calls on the floor when a
# response was truncated mid-tool-emission.


async def test_openai_responses_incomplete_with_tool_calls_executes_them():
    # OpenAI Responses can return status="incomplete:max_output_tokens"
    # after the model emitted a function_call item but the run was cut
    # off by the output token cap. The tool call should still be
    # executed instead of being silently dropped.
    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}],
            stop_reason="incomplete:max_output_tokens",
        )
    )
    provider.queue(MockLLMResponse(content="done", stop_reason="completed"))
    register_provider("openai-response", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="openai-response", tools=["echo"]))
    response = await agent.call("hello")

    assert response.iterations == 2
    assert response.content == "done"


async def test_openai_responses_bare_incomplete_with_tool_calls_continues():
    # `status="incomplete"` (no reason) with a tool call present must
    # also continue. Some SDK versions emit this for partial streaming
    # responses.
    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}],
            stop_reason="incomplete",
        )
    )
    provider.queue(MockLLMResponse(content="done", stop_reason="completed"))
    register_provider("openai-response", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="openai-response", tools=["echo"]))
    response = await agent.call("hello")

    assert response.iterations == 2


async def test_openai_responses_incomplete_content_filter_still_terminates():
    # `incomplete:content_filter` is an unrecoverable safety block —
    # even with tool calls present the loop ends so the caller can
    # surface the issue rather than executing a tool that the provider
    # flagged as potentially unsafe.
    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}],
            stop_reason="incomplete:content_filter",
        )
    )
    register_provider("openai-response", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="openai-response", tools=["echo"]))
    response = await agent.call("hello")

    assert response.iterations == 1
    assert response.stop_reason == "incomplete"


async def test_google_unexpected_tool_call_with_tool_calls_continues():
    # `UNEXPECTED_TOOL_CALL` is normally a parse-failure signal, but if
    # the SDK still surfaced the function call to us we should run it.
    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}],
            stop_reason="UNEXPECTED_TOOL_CALL",
        )
    )
    provider.queue(MockLLMResponse(content="done", stop_reason="STOP"))
    register_provider("google", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="google", tools=["echo"]))
    response = await agent.call("hello")

    assert response.iterations == 2
    assert response.content == "done"


async def test_google_malformed_function_call_with_tool_calls_continues():
    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}],
            stop_reason="MALFORMED_FUNCTION_CALL",
        )
    )
    provider.queue(MockLLMResponse(content="done", stop_reason="STOP"))
    register_provider("google", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="google", tools=["echo"]))
    response = await agent.call("hello")

    assert response.iterations == 2


async def test_google_safety_still_terminates_even_with_tool_calls():
    # `SAFETY` is a real unrecoverable block — even with tool calls the
    # loop should end so the caller can surface the safety issue rather
    # than executing a tool the provider refused to deliver.
    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}],
            stop_reason="SAFETY",
        )
    )
    register_provider("google", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="google", tools=["echo"]))
    response = await agent.call("hello")

    assert response.iterations == 1
    assert response.stop_reason == "finish"


async def test_anthropic_pause_turn_with_tool_calls_continues():
    # Real-world pause_turn responses can also contain tool_use blocks
    # (the server-side sampling loop paused mid-tool-emission). The loop
    # must continue and execute those tool calls.
    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}],
            stop_reason="pause_turn",
            terminate=False,
        )
    )
    provider.queue(_terminate_response("done"))
    register_provider("anthropic", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="anthropic", tools=["echo"]))
    response = await agent.call("hello")

    assert response.iterations == 2
    assert response.content == "done"


# ---------------------------------------------------------------------------
# Model-driven termination: the `finish` tool
# ---------------------------------------------------------------------------
# The agent loop ends when the model calls the `finish` tool. Natural
# end-of-turn signals from the provider (end_turn, stop, completed,
# STOP) are NOT enough on their own — the loop re-prompts the model so
# it can call more tools or call `finish` explicitly. This implements
# the "explicitness" guarantee from coreouto's philosophy.


async def test_finish_tool_call_ends_loop_regardless_of_provider_signal():
    # A response that has only a `finish` tool call (no other tools)
    # terminates the loop immediately, even if the provider's stop
    # signal would otherwise be CONTINUE.
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "f1", "name": "finish", "arguments": {"content": "all done"}}],
            stop_reason="tool_use",
            terminate=False,  # already has finish; don't auto-inject another
        )
    )
    register_provider("anthropic", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="anthropic"))
    response = await agent.call("hello")

    assert response.iterations == 1
    assert response.content == "all done"
    assert response.stop_reason == "finish"


async def test_finish_overrides_unrecoverable_provider_termination():
    # If the model calls `finish` and the provider ALSO reports an
    # unrecoverable termination (e.g. max_tokens was hit on the same
    # turn), `finish` wins — the model declared its intent and the
    # final answer is the `finish` content, not the empty response.
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[
                {"id": "f1", "name": "finish", "arguments": {"content": "truncated but done"}}
            ],
            stop_reason="max_tokens",
            terminate=False,
        )
    )
    register_provider("anthropic", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="anthropic"))
    response = await agent.call("hello")

    assert response.iterations == 1
    assert response.content == "truncated but done"
    assert response.stop_reason == "max_tokens"  # the underlying signal is preserved


async def test_provider_end_turn_without_finish_keeps_looping():
    # Under the model-driven policy, a text-only response (no `finish`
    # call) on Anthropic's natural end_turn does NOT terminate the
    # loop. The provider's natural end-of-turn signal is no longer
    # sufficient on its own — the model must call `finish` to close
    # the loop. Without `finish`, the loop just keeps calling the
    # provider and hits `max_iterations`.
    provider = MockProvider()
    for _ in range(5):
        provider.queue(
            MockLLMResponse(
                content="thinking...",
                stop_reason="end_turn",
                terminate=False,  # do NOT auto-inject finish
            )
        )
    register_provider("anthropic", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="anthropic", max_iterations=3))
    with pytest.raises(MaxIterationsError):
        await agent.call("hello")


async def test_openai_responses_completed_without_finish_keeps_looping():
    # OpenAI Responses' natural "completed" status is no longer
    # sufficient to terminate the loop. The model must call `finish`.
    provider = MockProvider()
    for _ in range(5):
        provider.queue(
            MockLLMResponse(
                content="thinking...",
                stop_reason="completed",
                terminate=False,
            )
        )
    register_provider("openai-response", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="openai-response", max_iterations=2))
    with pytest.raises(MaxIterationsError):
        await agent.call("hello")


async def test_google_stop_without_finish_keeps_looping():
    provider = MockProvider()
    for _ in range(5):
        provider.queue(
            MockLLMResponse(
                content="thinking...",
                stop_reason="STOP",
                terminate=False,
            )
        )
    register_provider("google", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="google", max_iterations=2))
    with pytest.raises(MaxIterationsError):
        await agent.call("hello")


async def test_openai_chat_stop_without_finish_keeps_looping():
    provider = MockProvider()
    for _ in range(5):
        provider.queue(
            MockLLMResponse(
                content="thinking...",
                stop_reason="stop",
                terminate=False,
            )
        )
    register_provider("openai", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="openai", max_iterations=2))
    with pytest.raises(MaxIterationsError):
        await agent.call("hello")


async def test_finish_with_other_tools_still_terminates_and_takes_finish_content():
    # The model can call `finish` alongside other tool calls in the
    # same turn (uncommon but valid). The loop ends, and the final
    # answer is the `finish` content (NOT the response's text).
    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            content="some text",
            tool_calls=[
                {"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}},
                {"id": "f1", "name": "finish", "arguments": {"content": "the answer"}},
            ],
            stop_reason="tool_use",
            terminate=False,
        )
    )
    register_provider("anthropic", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="anthropic", tools=["echo"]))
    response = await agent.call("hello")

    assert response.iterations == 1
    assert response.content == "the answer"


async def test_continue_loop_then_finish_terminates_with_finish_content():
    # A typical agent flow: model calls continue_loop to share progress,
    # then calls finish. The progress text from continue_loop is
    # recorded in the messages; the final answer is the finish content.
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[
                {
                    "id": "cl_1",
                    "name": "continue_loop",
                    "arguments": {"content": "still working..."},
                }
            ],
            stop_reason="tool_use",
            terminate=False,
        )
    )
    provider.queue(_terminate_response("final answer"))
    register_provider("anthropic", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="anthropic"))
    response = await agent.call("hello")

    assert response.iterations == 2
    assert response.content == "final answer"
    # The progress message is preserved in the conversation history.
    tool_msgs = [m for m in response.messages if m.role == "tool" and m.name == "continue_loop"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == "still working..."


# ---------------------------------------------------------------------------
# Two-step termination: detect attempt → confirm → finish
# ---------------------------------------------------------------------------
# Under the two-step policy, the loop detects a termination attempt
# (provider's natural end-of-turn signal without `finish`), then
# re-prompts the model with a confirmation user message. The model's
# next turn either calls `finish` (terminate) or calls more tools
# (continue). This is the explicit confirmation step the user
# requested — it guards against models that "think they're done" and
# emit a natural end-of-turn signal without explicitly calling
# `finish`.


async def test_natural_end_without_finish_injects_confirmation_and_continues():
    # A response that emits Anthropic's natural end_turn without a
    # `finish` call is a tentative termination attempt. The loop
    # injects a confirmation user message and re-prompts the model.
    # The model's next turn (this test queues it) calls `finish`,
    # which terminates the loop.
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            content="I think I'm done.",
            stop_reason="end_turn",
            terminate=False,  # no finish — tentative termination
        )
    )
    provider.queue(_terminate_response("the final answer"))
    register_provider("anthropic", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="anthropic"))
    response = await agent.call("hello")

    # The loop ran the tentative turn (iteration 1) plus the
    # confirmation turn (iteration 2), and the confirmation turn
    # called `finish`.
    assert response.iterations == 2
    assert response.content == "the final answer"
    # The confirmation user message is in the conversation history
    # (between the tentative assistant turn and the final
    # `finish`-calling turn).
    user_messages = [m for m in response.messages if m.role == "user"]
    confirmation_msgs = [m for m in user_messages if "finish" in (m.content or "")]
    assert len(confirmation_msgs) == 1


async def test_confirmation_lets_model_keep_working_instead_of_finishing():
    # The model "thinks it's done" but on the confirmation turn
    # decides to keep working — it calls another tool instead of
    # `finish`. The loop continues to run that tool.
    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"echo: {msg}"

    provider = MockProvider()
    # Tentative turn: natural end_turn, no finish.
    provider.queue(
        MockLLMResponse(
            content="wait, I want to do more.",
            stop_reason="end_turn",
            terminate=False,
        )
    )
    # Confirmation turn: model calls echo instead of finish.
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}],
            stop_reason="tool_use",
            terminate=False,
        )
    )
    # After the echo, the model calls finish.
    provider.queue(_terminate_response("done"))
    register_provider("anthropic", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="anthropic", tools=["echo"]))
    response = await agent.call("hello")

    assert response.iterations == 3
    assert response.content == "done"


async def test_unrecoverable_termination_terminates_immediately_without_finish():
    # An unrecoverable provider termination (SAFETY, refusal, content_filter,
    # etc.) ends the loop immediately even without a `finish` call —
    # the provider refused to deliver, the run was cancelled, etc. The
    # user shouldn't have to wait for a confirmation turn that will
    # never produce a useful answer.
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            content="[filtered]",
            stop_reason="content_filter",
            terminate=False,  # no finish; not auto-injected because terminate=False
        )
    )
    register_provider("openai", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="openai"))
    response = await agent.call("hello")

    assert response.iterations == 1
    assert response.stop_reason == "content_filter"


async def test_natural_end_with_finish_skips_confirmation_terminates_immediately():
    # If the model already called `finish` in the same turn as the
    # provider's natural end-of-turn signal, the loop terminates
    # immediately — no confirmation needed. (The two-step policy is
    # only invoked on the *tentative* path where `finish` is missing.)
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "f1", "name": "finish", "arguments": {"content": "all done"}}],
            stop_reason="end_turn",
            terminate=False,
        )
    )
    register_provider("anthropic", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="anthropic"))
    response = await agent.call("hello")

    assert response.iterations == 1
    assert response.content == "all done"


async def test_confirmation_step_keeps_looping_until_max_iterations_if_model_never_finishes():
    # If the model never calls `finish` and never calls other tools,
    # the loop keeps injecting confirmations until `max_iterations`
    # is hit. This is the safety net for the two-step policy: a model
    # that never declares its intent to end will eventually raise
    # `MaxIterationsError`.
    provider = MockProvider()
    for _ in range(5):
        provider.queue(
            MockLLMResponse(
                content="still thinking...",
                stop_reason="end_turn",
                terminate=False,
            )
        )
    register_provider("anthropic", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="anthropic", max_iterations=3))
    with pytest.raises(MaxIterationsError):
        await agent.call("hello")


# ---------------------------------------------------------------------------
# CONTINUE values are NOT tentative terminations
# ---------------------------------------------------------------------------
# `tool_use`, `tool_calls`, `function_call`, and `pause_turn` are
# CONTINUE values — the model is mid-task or the server says keep
# going. They must NOT trigger the two-step confirmation flow; the
# loop should just continue (executing any tool calls and re-prompting
# the model on the next iteration). See the Anthropic and OpenAI SDK
# docstrings: tool_use/tool_calls mean "the model called a tool",
# pause_turn means "re-send the response to let the model continue".


async def test_anthropic_pause_turn_without_finish_continues_without_confirmation():
    # `pause_turn` is a CONTINUE signal — the server-side sampling
    # loop hit its iteration limit. The official guidance is to
    # send the response back so the loop can continue. coreouto
    # should NOT inject a confirmation message (that would be
    # incorrect — the model didn't "think it's done", the server
    # paused the loop). It should just re-prompt with the existing
    # messages and let the model continue naturally.
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            content="mid-thought...",
            stop_reason="pause_turn",
            terminate=False,
        )
    )
    provider.queue(_terminate_response("done"))
    register_provider("anthropic", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="anthropic"))
    response = await agent.call("hello")

    # Two iterations: pause_turn (no confirmation, just re-prompt)
    # → finish.
    assert response.iterations == 2
    assert response.content == "done"
    # Crucially, no confirmation user message was injected.
    user_messages = [m for m in response.messages if m.role == "user"]
    confirmation_msgs = [m for m in user_messages if "finish" in (m.content or "")]
    assert len(confirmation_msgs) == 0


async def test_openai_chat_tool_calls_continue_reason_continues_without_confirmation():
    # `tool_calls` finish_reason with no actual tool calls (degenerate
    # but possible) should continue without injecting a confirmation.
    # In practice this is rare; the loop should just re-prompt.
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            content="...",
            stop_reason="tool_calls",
            terminate=False,
        )
    )
    provider.queue(_terminate_response("done"))
    register_provider("openai", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="openai"))
    response = await agent.call("hello")

    assert response.iterations == 2
    assert response.content == "done"
    user_messages = [m for m in response.messages if m.role == "user"]
    confirmation_msgs = [m for m in user_messages if "finish" in (m.content or "")]
    assert len(confirmation_msgs) == 0


async def test_openai_chat_legacy_function_call_continues_without_confirmation():
    # The legacy `function_call` finish_reason is the deprecated
    # predecessor of `tool_calls` — same meaning, same CONTINUE
    # classification.
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            content="...",
            stop_reason="function_call",
            terminate=False,
        )
    )
    provider.queue(_terminate_response("done"))
    register_provider("openai", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="openai"))
    response = await agent.call("hello")

    assert response.iterations == 2
    assert response.content == "done"
    user_messages = [m for m in response.messages if m.role == "user"]
    confirmation_msgs = [m for m in user_messages if "finish" in (m.content or "")]
    assert len(confirmation_msgs) == 0


async def test_anthropic_tool_use_stop_reason_with_tool_calls_executes_them():
    # The classic case: `tool_use` stop_reason with actual tool
    # calls. The loop must execute the tools, NOT inject a
    # confirmation. The continuation is mechanical — the model
    # called a tool, the loop runs it, the model gets the result.
    @register_tool("echo")
    def echo(msg: str) -> str:
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}],
            stop_reason="tool_use",
            terminate=False,
        )
    )
    provider.queue(_terminate_response("done"))
    register_provider("anthropic", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="anthropic", tools=["echo"]))
    response = await agent.call("hello")

    assert response.iterations == 2
    assert response.content == "done"
    # The tool was actually executed (not dropped).
    tool_msgs = [m for m in response.messages if m.role == "tool" and m.name == "echo"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == "echo: hi"
    # No confirmation was injected.
    user_messages = [m for m in response.messages if m.role == "user"]
    confirmation_msgs = [m for m in user_messages if "finish" in (m.content or "")]
    assert len(confirmation_msgs) == 0
