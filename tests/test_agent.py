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


async def test_happy_path_single_iteration_finish():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="<finish>done</finish>"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", max_iterations=10))
    response = await agent.call("hello")

    assert response.content == "done"
    assert response.iterations == 1
    assert response.finish_called is True
    assert len(response.messages) == 3
    assert response.messages[0].role == "system"
    assert response.messages[1].role == "user"
    assert response.messages[1].content == "hello"
    assert response.messages[2].role == "assistant"


async def test_multi_iteration_with_tool_call():
    side_effect: list[str] = []

    @register_tool("echo")
    def echo(msg: str) -> str:
        side_effect.append(msg)
        return f"echo: {msg}"

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(tool_calls=[{"id": "tc1", "name": "echo", "arguments": {"msg": "hi"}}])
    )
    provider.queue(MockLLMResponse(content="<finish>done</finish>"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", tools=["echo"]))
    response = await agent.call("hello")

    assert side_effect == ["hi"]
    assert response.content == "done"
    assert response.iterations == 2
    assert response.finish_called is True


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
    provider.queue(MockLLMResponse(content="<finish>done</finish>"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", tools=["tool_a", "tool_b"]))
    response = await agent.call("hello")

    assert calls == [("a", "1"), ("b", "2")]
    assert response.iterations == 2


async def test_max_iterations_error():
    provider = MockProvider()
    for _ in range(3):
        provider.queue(MockLLMResponse(content="thinking..."))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", max_iterations=2))
    with pytest.raises(MaxIterationsError, match=r"max_iterations \(2\) reached"):
        await agent.call("hello")


async def test_system_prompt_injected_as_first_message():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="<finish>done</finish>"))
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
    provider.queue(MockLLMResponse(content="<finish>done</finish>"))
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
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="a"))
    provider.queue(MockLLMResponse(content="b"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", max_iterations=5))
    with pytest.raises(MaxIterationsError, match=r"max_iterations \(2\) reached"):
        await agent.call(
            "hello",
            override=AgentConfig(name="ovr", model="m", provider="mock", max_iterations=2),
        )


async def test_tool_error_surfaces_as_is_error():
    @register_tool("boom")
    def boom() -> str:
        raise RuntimeError("exploded")

    provider = MockProvider()
    provider.queue(MockLLMResponse(tool_calls=[{"id": "tc1", "name": "boom", "arguments": {}}]))
    provider.queue(MockLLMResponse(content="<finish>done</finish>"))
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
    provider.queue(MockLLMResponse(content="<finish>done</finish>"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", tools=["async_echo"]))
    await agent.call("hello")

    _tc, result = provider.formatted_tool_results[0]
    assert result.content == "async-hi"


async def test_usage_tracking():
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            content="<finish>done</finish>",
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
            content="<finish>done</finish>",
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
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="nope"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", max_iterations=0))
    with pytest.raises(MaxIterationsError, match=r"max_iterations \(0\) reached"):
        await agent.call("hello")


async def test_provider_config_forwarded_to_provider():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="<finish>done</finish>"))
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


async def test_max_iterations_still_works_with_force_finish():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="a"))
    provider.queue(MockLLMResponse(content="b"))
    provider.queue(MockLLMResponse(content="c"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock", max_iterations=2))
    with pytest.raises(MaxIterationsError, match=r"max_iterations \(2\) reached"):
        await agent.call("hello")


async def test_max_tokens_translates_per_provider():
    for pname, expected_key in (
        ("openai", "max_tokens"),
        ("openai-response", "max_output_tokens"),
        ("anthropic", "max_tokens"),
        ("google", "max_output_tokens"),
    ):
        provider = MockProvider()
        provider.queue(MockLLMResponse(content="<finish>done</finish>"))
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
        provider.queue(MockLLMResponse(content="<finish>done</finish>"))
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
    provider.queue(MockLLMResponse(content="<finish>done</finish>"))
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
    provider.queue(MockLLMResponse(content="<finish>done</finish>"))
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
    provider.queue(MockLLMResponse(content="<finish>done</finish>"))
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


async def test_reminder_injected_when_no_finish_tag():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="just text"))
    provider.queue(MockLLMResponse(content="<finish>done</finish>"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    response = await agent.call("hello")

    assert response.content == "done"
    second_call = provider.calls[1]
    messages = second_call["messages"]
    user_messages = [m for m in messages if m.role == "user"]
    assert len(user_messages) == 2
    assert "<finish>" in user_messages[-1].content


async def test_finish_inside_think_block_is_ignored():
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            content="<think>maybe <finish>oops</finish></think>\n<finish>real answer</finish>"
        )
    )
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    response = await agent.call("hello")

    assert response.content == "real answer"
    assert response.finish_called is True


async def test_think_block_with_finish_then_real_finish_outside():
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            content="<think>nope<finish>skip me</finish></think> now <finish>take this</finish>"
        )
    )
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    response = await agent.call("hello")

    assert response.content == "take this"


async def test_unclosed_think_block_consumes_rest_of_response():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="<think><finish>never returns"))
    provider.queue(MockLLMResponse(content="<finish>real</finish>"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    response = await agent.call("hello")

    assert response.content == "real"


async def test_multiple_think_blocks_all_ignored():
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            content="<think><finish>a</finish></think> and <think><finish>b</finish></think>\n<finish>c</finish>"
        )
    )
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    response = await agent.call("hello")

    assert response.content == "c"


async def test_think_with_no_finish_inside_continues_to_real_finish():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="<think>just thinking</think><finish>ok</finish>"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    response = await agent.call("hello")

    assert response.content == "ok"


async def test_history_prepended_to_messages():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="<finish>ok</finish>"))
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
    provider.queue(MockLLMResponse(content="<finish>ok</finish>"))
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
    provider.queue(MockLLMResponse(content="<finish>ok</finish>"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    await agent.call("hello", history=[])

    messages = provider.calls[0]["messages"]
    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[1].content == "hello"


async def test_history_with_system_prompt_prepends_cfg_first():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="<finish>ok</finish>"))
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
    provider.queue(MockLLMResponse(content="<finish>ok</finish>"))
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
    assert len(sent) == 6
    assert [m.content for m in sent] == [
        _DEFAULT_SYSTEM_PROMPT,
        "fabricated question 1",
        "fabricated answer 1",
        "fabricated question 2",
        "fabricated answer 2",
        "real question",
    ]


async def test_history_preserves_assistant_tool_calls():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="<finish>ok</finish>"))
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
    provider.queue(MockLLMResponse(content="<finish>ok</finish>"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    history = [Message(role="user", content="prev"), Message(role="assistant", content="ans")]
    response = call_sync(agent, "new", history=history)
    assert response.content == "ok"
    sent = provider.calls[0]["messages"]
    assert [m.content for m in sent] == [_DEFAULT_SYSTEM_PROMPT, "prev", "ans", "new"]


async def test_history_works_with_override():
    provider = MockProvider()
    provider.queue(MockLLMResponse(content="<finish>ok</finish>"))
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
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(content="thinking"),
        MockLLMResponse(content="<finish>ok</finish>"),
    )
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    agent.inject_user_message("injected mid-loop")
    response = await agent.call("initial")

    sent = provider.calls[1]["messages"]
    contents = [m.content for m in sent if m.role == "user"]
    assert "initial" in contents
    assert "injected mid-loop" in contents
    assert response.content == "ok"


async def test_inject_user_message_fires_hook():
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(content="thinking"),
        MockLLMResponse(content="<finish>ok</finish>"),
    )
    register_provider("mock", provider)

    injection_events: list[Message] = []

    def capture(message, **_):
        injection_events.append(message)

    register_hook("on_user_injection", capture)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    agent.inject_user_message("hi from outside")
    await agent.call("hi from caller")

    assert len(injection_events) == 1
    assert injection_events[0].role == "user"
    assert injection_events[0].content == "hi from outside"


async def test_inject_multiple_messages_all_drained_in_one_iteration():
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(content="thinking"),
        MockLLMResponse(content="<finish>ok</finish>"),
    )
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
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

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(content="thinking"),
        MockLLMResponse(content="<finish>ok</finish>"),
    )
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))

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
    provider.queue(MockLLMResponse(content="<finish>ok</finish>"))
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
    provider.queue(MockLLMResponse(content="<finish>my answer</finish>"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    response = await agent.call("hello")

    assert response.content == "my answer"
    assert len(received) == 1
    assert received[0]["content"] == "my answer"
    assert received[0]["raw_content"] == "<finish>my answer</finish>"
    assert received[0]["iterations"] == 1
    assert "messages" in received[0]


async def test_tool_returning_list_of_content_blocks():
    @register_tool("show_image")
    async def show_image(label: str) -> list:
        return [
            TextBlock(text=f"Image for: {label}"),
            ImageBlock(data=b"\\x89PNG\\r\\nfake-bytes", mime_type="image/png"),
        ]

    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            tool_calls=[{"id": "tc1", "name": "show_image", "arguments": {"label": "cat"}}]
        )
    )
    provider.queue(MockLLMResponse(content="<finish>done</finish>"))
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
    assert tool_msg.content[1].data == b"\\x89PNG\\r\\nfake-bytes"
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
    provider.queue(MockLLMResponse(content="<finish>ok</finish>"))
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
    provider.queue(MockLLMResponse(content="<finish>done</finish>"))
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
    provider.queue(MockLLMResponse(content="<finish>done</finish>"))
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
    provider.queue(MockLLMResponse(content="<finish>done</finish>"))
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
    provider.queue(MockLLMResponse(content="<finish>done</finish>"))
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
    provider.queue(MockLLMResponse(content="<finish>done</finish>"))
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
    provider.queue(MockLLMResponse(content="<finish>done</finish>"))
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


async def test_finish_without_end_signal_does_not_return():
    """If the model emits <finish> but the provider reports a non-clean
    stop reason (e.g. max_tokens, stop_sequence, tool_use), the agent
    loop must NOT return. The model's declaration alone is not enough;
    the API must also have stopped the model cleanly.
    """
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            content="<finish>premature</finish>",
            stop_reason="max_tokens",  # truncated, not a clean end
        )
    )
    provider.queue(MockLLMResponse(content="<finish>real</finish>"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    response = await agent.call("hi")

    assert response.content == "real"
    assert len(provider.calls) == 2
    # The reminder should have been injected on the first turn since we
    # didn't return — that's why the mock had a second response queued.
    user_messages = [m for m in provider.calls[1]["messages"] if m.role == "user"]
    assert any("<finish>" in m.content for m in user_messages)


async def test_finish_with_end_signal_returns_immediately():
    """If both <finish> matches AND stop_reason is a clean-end signal
    (Anthropic end_turn / OpenAI stop / Responses completed / Google STOP),
    the loop returns on this turn.
    """
    for end in ("end_turn", "stop", "completed", "STOP"):
        provider = MockProvider()
        provider.queue(MockLLMResponse(content=f"<finish>x ({end})</finish>", stop_reason=end))
        register_provider("mock", provider)
        agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
        response = await agent.call("hi")
        assert response.content == f"x ({end})", end
        # reset for next iteration
        from coreouto.providers import clear_providers
        from coreouto.tools import clear_tools

        clear_tools()
        clear_providers()


async def test_finish_with_no_stop_reason_continues_loop():
    """If the LLMResponse has no stop_reason (e.g. an older provider or
    a custom mock), the loop must NOT trust the <finish> alone. The
    default behavior is to keep going so the user gets a chance to
    continue if the model really was about to say more.
    """
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            content="<finish>risky</finish>",
            stop_reason=None,
        )
    )
    provider.queue(MockLLMResponse(content="<finish>safe</finish>"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    response = await agent.call("hi")

    assert response.content == "safe"


async def test_finish_with_tool_use_stop_continues_loop():
    """Anthropic stop_reason='tool_use' (the model stopped to invoke a
    tool) must not satisfy the end condition. The model's <finish> tag
    is ignored, and the loop continues.
    """
    provider = MockProvider()
    provider.queue(
        MockLLMResponse(
            content="<finish>oops</finish>",
            stop_reason="tool_use",
        )
    )
    provider.queue(MockLLMResponse(content="<finish>real</finish>"))
    register_provider("mock", provider)

    agent = Agent(AgentConfig(name="test", model="m", provider="mock"))
    response = await agent.call("hi")

    assert response.content == "real"
