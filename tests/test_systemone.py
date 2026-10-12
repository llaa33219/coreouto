"""Tests for the System One model abstraction and the SystemOneProvider bridge."""

from __future__ import annotations

import base64
import json
from typing import Any, Literal

import pytest

from coreouto import Agent, AgentConfig, Response, agent_as_tool, register_agent_preset
from coreouto.agent import MaxIterationsError
from coreouto.providers import clear_providers, register_provider
from coreouto.systemone import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneResult,
    available_systemone_models,
    clear_systemone_models,
    get_systemone_model,
    register_systemone_model,
)
from coreouto.systemone.bridge import GOAL_COMPLETE_KEY, NEXT_TOOL_KEY, SystemOneProvider
from coreouto.tools import clear_tools, register_tool
from tests.conftest import MockLLMResponse, MockProvider, MockSystemOneModel


@pytest.fixture(autouse=True)
def _clean_state():
    clear_tools()
    clear_providers()
    clear_systemone_models()
    yield
    clear_tools()
    clear_providers()
    clear_systemone_models()


def _noul(value: float) -> NoulAnswer:
    return NoulAnswer(noul=value)


def _choice(value: str) -> ChoiceAnswer:
    return ChoiceAnswer(choice=value)


def _iteration_result(complete: float, tool: str, **args: ChoiceAnswer) -> SystemOneResult:
    return SystemOneResult(
        answers={
            GOAL_COMPLETE_KEY: _noul(complete),
            NEXT_TOOL_KEY: _choice(tool),
            **args,
        }
    )


def _make_agent(
    dm: MockSystemOneModel,
    *,
    tools: list[str] | None = None,
    provider_name: str = "jev",
    model: str = "jev-latest",
) -> Agent:
    register_provider(provider_name, SystemOneProvider(dm))
    return Agent(
        AgentConfig(
            name="test",
            model=model,
            provider=provider_name,
            tools=tools or [],
        )
    )


# --- registry ---------------------------------------------------------------


def test_systemone_model_registry_roundtrip():
    dm = MockSystemOneModel()
    register_systemone_model("mock", dm)
    assert get_systemone_model("mock") is dm
    assert available_systemone_models() == ["mock"]


def test_systemone_model_registry_missing_names_available():
    with pytest.raises(KeyError, match="System One model not registered: 'nope'"):
        get_systemone_model("nope")


def test_systemone_provider_accepts_registry_name():
    dm = MockSystemOneModel()
    register_systemone_model("mock", dm)
    provider = SystemOneProvider("mock")
    assert provider._systemone_model is dm


# --- types ------------------------------------------------------------------


def test_question_types_are_discriminated():
    assert NoulQuestion(instructions="?").type == "noul"
    assert ChoiceQuestion(instructions="?", criteria={"a": "A"}).type == "choice"
    assert ScoreQuestion(instructions="?", criteria=["lo", "hi"]).type == "score"


def test_systemone_result_answers_json():
    result = SystemOneResult(
        answers={
            "v": ChoiceAnswer(choice="x", probabilities={"x": 1.0}, confidence=0.9),
            "n": NoulAnswer(noul=0.3),
            "s": ScoreAnswer(score=1.5),
        }
    )
    parsed = json.loads(result.answers_json())
    assert parsed["v"]["choice"] == "x"
    assert parsed["n"]["noul"] == 0.3
    assert parsed["s"]["score"] == 1.5


# --- bridge: the agent loop driven by typed selections ------------------------------


async def test_loop_runs_tool_then_finishes_empty_without_answer_format():
    @register_tool("get_status")
    def get_status() -> str:
        """Get the deployment status."""
        return "all green"

    dm = MockSystemOneModel(
        [
            _iteration_result(0.1, "get_status"),
            _iteration_result(0.95, "get_status"),
        ]
    )
    agent = _make_agent(dm, tools=["get_status"])
    response = await agent.call("Is the deploy healthy?")

    assert isinstance(response, Response)
    assert response.iterations == 2
    assert response.stop_reason == "finish"
    assert response.content == ""
    assert response.answer is None

    tool_messages = [m for m in response.messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "all green"


async def test_state_carries_goal_history_and_current_situation():
    @register_tool("get_status")
    def get_status() -> str:
        """Get the deployment status."""
        return "all green"

    dm = MockSystemOneModel(
        [
            _iteration_result(0.1, "get_status"),
            _iteration_result(0.99, "get_status"),
        ]
    )
    agent = _make_agent(dm, tools=["get_status"])
    await agent.call("Is the deploy healthy?")

    first_state = dm.calls[0]["state"]
    assert "Is the deploy healthy?" in first_state["goal"]
    assert first_state["history"] == []

    second_state = dm.calls[1]["state"]
    assert second_state["history"][0] == {
        "action": "call_tool",
        "tool": "get_status",
        "arguments": {},
    }
    assert second_state["current"] == [{"tool": "get_status", "result": "all green"}]


async def test_iteration_questions_cover_goal_tool_and_enum_args():
    @register_tool("scale")
    def scale(direction: Literal["up", "down"]) -> str:
        """Scale the deployment."""
        return f"scaled {direction}"

    dm = MockSystemOneModel([_iteration_result(0.9, "scale")])
    agent = _make_agent(dm, tools=["scale"])
    await agent.call("scale?")

    questions = dm.calls[0]["questions"]
    assert isinstance(questions[GOAL_COMPLETE_KEY], NoulQuestion)
    assert isinstance(questions[NEXT_TOOL_KEY], ChoiceQuestion)
    assert set(questions[NEXT_TOOL_KEY].criteria) == {"scale"}
    arg_question = questions["arg__scale__direction"]
    assert set(arg_question.criteria) == {"up", "down"}


async def test_enum_argument_is_selected_and_passed_to_the_tool():
    @register_tool("scale")
    def scale(direction: Literal["up", "down"]) -> str:
        """Scale the deployment."""
        return f"scaled {direction}"

    dm = MockSystemOneModel(
        [
            _iteration_result(0.1, "scale", **{"arg__scale__direction": _choice("up")}),
            _iteration_result(0.99, "scale"),
        ]
    )
    agent = _make_agent(dm, tools=["scale"])
    response = await agent.call("scale as needed")

    tool_messages = [m for m in response.messages if m.role == "tool"]
    assert tool_messages[0].content == "scaled up"


async def test_goal_complete_at_threshold_boundary():
    @register_tool("x")
    def x() -> str:
        """No-op."""
        return "x"

    dm = MockSystemOneModel([_iteration_result(0.5, "x")])
    agent = _make_agent(dm, tools=["x"])
    response = await agent.call("done?")
    assert response.stop_reason == "finish"
    assert response.iterations == 1


async def test_loop_is_bounded_by_max_iterations():
    @register_tool("ping")
    def ping() -> str:
        """Ping."""
        return "pong"

    dm = MockSystemOneModel([_iteration_result(0.0, "ping")] * 5)
    register_provider("jev", SystemOneProvider(dm))
    agent = Agent(
        AgentConfig(name="t", model="m", provider="jev", tools=["ping"], max_iterations=3)
    )
    with pytest.raises(MaxIterationsError):
        await agent.call("loop forever")


async def test_unknown_tool_selection_raises_clear_error():
    @register_tool("real_tool")
    def real_tool() -> str:
        """A real tool."""
        return "ok"

    dm = MockSystemOneModel([_iteration_result(0.1, "hallucinated_tool")])
    agent = _make_agent(dm, tools=["real_tool"])
    with pytest.raises(ValueError, match="unknown tool 'hallucinated_tool'"):
        await agent.call("go")


# --- bridge: answer_format ----------------------------------------------------


async def test_answer_format_runs_answer_selection_after_goal_complete():
    @register_tool("get_status")
    def get_status() -> str:
        """Get the deployment status."""
        return "all green"

    dm = MockSystemOneModel(
        [
            _iteration_result(0.99, "get_status"),
            SystemOneResult(answers={"verdict": _choice("healthy")}),
        ]
    )
    agent = _make_agent(dm, tools=["get_status"])
    response = await agent.call(
        "verdict?",
        answer_format={
            "verdict": ChoiceQuestion(
                instructions="final verdict", criteria={"healthy": "h", "broken": "b"}
            )
        },
    )

    assert response.content == ""
    assert response.answer is not None
    assert response.answer.answers["verdict"].choice == "healthy"
    assert set(dm.calls[1]["questions"]) == {"verdict"}
    assert dm.calls[1]["state"]["goal"]  # the answer phase sees the full state


async def test_single_answer_mode_skips_goal_detection():
    dm = MockSystemOneModel([SystemOneResult(answers={"v": _choice("yes")})])
    agent = _make_agent(dm)
    response = await agent.call(
        "quick?",
        answer_format={"v": ChoiceQuestion(instructions="?", criteria={"yes": "y", "no": "n"})},
    )

    assert response.iterations == 1
    assert response.answer.answers["v"].choice == "yes"
    questions = dm.calls[0]["questions"]
    assert GOAL_COMPLETE_KEY not in questions
    assert set(questions) == {"v"}


async def test_no_tools_without_answer_format_raises_clear_error():
    dm = MockSystemOneModel()
    agent = _make_agent(dm)
    with pytest.raises(ValueError, match="answer_format"):
        await agent.call("nothing to decide")


async def test_chat_provider_rejects_answer_format():
    register_provider("mock", MockProvider([MockLLMResponse(content="hi")]))
    agent = Agent(AgentConfig(name="t", model="m", provider="mock"))
    with pytest.raises(ValueError, match="does not support answer_format"):
        await agent.call("hi", answer_format={"x": NoulQuestion(instructions="?")})


# --- bridge: tool compatibility ----------------------------------------------


async def test_free_form_required_parameter_is_rejected():
    @register_tool("search")
    def search(query: str) -> str:
        """Search."""
        return query

    dm = MockSystemOneModel()
    agent = _make_agent(dm, tools=["search"])
    with pytest.raises(ValueError, match=r"'search'.*'query'.*free-form input"):
        await agent.call("find something")
    assert dm.calls == []


async def test_optional_non_enum_parameter_is_omitted():
    @register_tool("ping")
    def ping(verbose: bool = False) -> str:
        """Ping."""
        return f"pong verbose={verbose}"

    dm = MockSystemOneModel([_iteration_result(0.9, "ping")])
    agent = _make_agent(dm, tools=["ping"])
    response = await agent.call("ping?")
    assert response.stop_reason == "finish"


# --- composition: agent_as_tool with answer_format ----------------------------


async def test_agent_as_tool_returns_selected_answers_as_tool_result():
    dm = MockSystemOneModel([SystemOneResult(answers={"verdict": _choice("billing")})])
    register_provider("jev", SystemOneProvider(dm))
    register_agent_preset("triage", model="jev-latest", provider="jev")

    tool = agent_as_tool(
        "triage",
        answer_format={
            "verdict": ChoiceQuestion(
                instructions="Which team?", criteria={"billing": "b", "tech": "t"}
            )
        },
    )
    result = await tool.handler(task="user wants a refund")

    parsed = json.loads(result)
    assert parsed["verdict"]["choice"] == "billing"


async def test_agent_as_tool_without_answer_format_returns_content():
    register_provider("mock", MockProvider([MockLLMResponse(content="sub answer")]))
    register_agent_preset("sub", model="m", provider="mock")

    tool = agent_as_tool("sub")
    assert await tool.handler(task="do it") == "sub answer"


# --- bridge: per-turn multimodal input (turn_input) ---------------------------


async def test_turn_input_supplies_media_every_iteration():
    from coreouto._types import ImageBlock

    @register_tool("click_button")
    def click_button() -> str:
        """Click the button."""
        return "clicked"

    captures = []

    def screenshot(messages: list) -> list:
        captures.append(len(messages))
        return [ImageBlock(data=b"png-bytes", mime_type="image/png")]

    dm = MockSystemOneModel(
        [
            _iteration_result(0.1, "click_button"),
            _iteration_result(0.99, "click_button"),
            SystemOneResult(answers={"done": _choice("yes")}),
        ]
    )
    register_provider("jev", SystemOneProvider(dm, turn_input=screenshot))
    agent = Agent(AgentConfig(name="t", model="clef", provider="jev", tools=["click_button"]))
    response = await agent.call(
        "finish the task on screen",
        answer_format={"done": ChoiceQuestion(instructions="done?", criteria={"yes": "y"})},
    )

    assert response.answer.answers["done"].choice == "yes"
    # media reached every decide() call: 2 loop iterations + 1 answer phase
    assert len(dm.calls) == 3
    for call in dm.calls:
        media = call["kwargs"]["media"]
        assert len(media) == 1 and media[0].data == b"png-bytes"
    # the callback saw a growing transcript
    assert captures[1] > captures[0]


async def test_turn_input_may_be_async_and_may_return_none():
    async def no_capture(messages: list) -> None:
        return None

    dm = MockSystemOneModel([SystemOneResult(answers={"v": _choice("x")})])
    register_provider("jev", SystemOneProvider(dm, turn_input=no_capture))
    agent = Agent(AgentConfig(name="t", model="clef", provider="jev"))
    response = await agent.call(
        "q", answer_format={"v": ChoiceQuestion(instructions="?", criteria={"x": "x"})}
    )
    assert response.answer.answers["v"].choice == "x"
    assert dm.calls[0]["kwargs"]["media"] is None


async def test_without_turn_input_media_is_none():
    dm = MockSystemOneModel([SystemOneResult(answers={"v": _choice("x")})])
    agent = _make_agent(dm)
    await agent.call(
        "q", answer_format={"v": ChoiceQuestion(instructions="?", criteria={"x": "x"})}
    )
    assert dm.calls[0]["kwargs"]["media"] is None


async def test_turn_input_text_blocks_land_in_state_context():
    from coreouto._types import TextBlock

    @register_tool("click_button")
    def click_button() -> str:
        """Click the button."""
        return "clicked"

    def observe(messages: list) -> list:
        return [TextBlock(text="window: Settings / dialog open")]

    dm = MockSystemOneModel([_iteration_result(0.9, "click_button")])
    register_provider("jev", SystemOneProvider(dm, turn_input=observe))
    agent = Agent(AgentConfig(name="t", model="clef", provider="jev", tools=["click_button"]))
    await agent.call("handle the dialog")

    assert dm.calls[0]["state"]["context"] == "window: Settings / dialog open"
    assert dm.calls[0]["kwargs"]["media"] is None


async def test_turn_input_routes_text_and_media_to_separate_channels():
    from coreouto._types import ImageBlock, TextBlock

    def observe(messages: list) -> list:
        return [
            TextBlock(text="time: 12:00"),
            ImageBlock(data=b"png", mime_type="image/png"),
            TextBlock(text="battery: 80%"),
        ]

    dm = MockSystemOneModel([SystemOneResult(answers={"v": _choice("x")})])
    register_provider("jev", SystemOneProvider(dm, turn_input=observe))
    agent = Agent(AgentConfig(name="t", model="clef", provider="jev"))
    await agent.call(
        "q", answer_format={"v": ChoiceQuestion(instructions="?", criteria={"x": "x"})}
    )

    call = dm.calls[0]
    assert call["state"]["context"] == "time: 12:00\nbattery: 80%"
    assert len(call["kwargs"]["media"]) == 1


async def test_turn_input_rejects_document_blocks():
    from coreouto._types import DocumentBlock

    def observe(messages: list) -> list:
        return [DocumentBlock(data=b"pdf", mime_type="application/pdf")]

    dm = MockSystemOneModel([SystemOneResult(answers={"v": _choice("x")})])
    register_provider("jev", SystemOneProvider(dm, turn_input=observe))
    agent = Agent(AgentConfig(name="t", model="clef", provider="jev"))
    with pytest.raises(ValueError, match="turn_input cannot return DocumentBlock"):
        await agent.call(
            "q", answer_format={"v": ChoiceQuestion(instructions="?", criteria={"x": "x"})}
        )


# --- bridge: System One hooks -----------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_hooks():
    from coreouto.hooks import clear_hooks

    clear_hooks()
    yield
    clear_hooks()


async def test_after_llm_call_carries_each_turns_systemone_result():
    """The per-turn System One result needs no new hook: it rides AFTER_LLM_CALL
    as response.raw, including the answer-phase result as response.answer."""
    from coreouto.hooks import AFTER_LLM_CALL, register_hook

    @register_tool("ping")
    def ping() -> str:
        """Ping."""
        return "pong"

    seen = []
    register_hook(AFTER_LLM_CALL, lambda response, **kw: seen.append(response))

    dm = MockSystemOneModel(
        [
            _iteration_result(0.2, "ping"),
            _iteration_result(0.99, "ping"),
            SystemOneResult(answers={"v": _choice("x")}),
        ]
    )
    agent = _make_agent(dm, tools=["ping"])
    response = await agent.call(
        "go", answer_format={"v": ChoiceQuestion(instructions="?", criteria={"x": "x"})}
    )

    assert len(seen) == 2
    assert isinstance(seen[0].raw, SystemOneResult)
    assert seen[0].raw.answers[GOAL_COMPLETE_KEY].noul == 0.2
    assert seen[1].answer.answers["v"].choice == "x"
    assert response.answer.answers["v"].choice == "x"


async def test_on_goal_complete_fires_before_answer_selection():
    from coreouto.hooks import ON_GOAL_COMPLETE, register_hook

    @register_tool("ping")
    def ping() -> str:
        """Ping."""
        return "pong"

    events = []

    def on_goal_complete(state, result, model, **kw):
        events.append((result, len(dm.calls)))

    register_hook(ON_GOAL_COMPLETE, on_goal_complete)

    dm = MockSystemOneModel(
        [
            _iteration_result(0.99, "ping"),
            SystemOneResult(answers={"v": _choice("x")}),
        ]
    )
    agent = _make_agent(dm, tools=["ping"])
    await agent.call(
        "go", answer_format={"v": ChoiceQuestion(instructions="?", criteria={"x": "x"})}
    )

    # the hook observed the goal-check result...
    assert len(events) == 1
    assert events[0][0].answers[GOAL_COMPLETE_KEY].noul == 0.99
    # ...before the answer-selection decide() ran (only 1 call so far)
    assert events[0][1] == 1
    assert len(dm.calls) == 2


async def test_on_goal_complete_fires_without_answer_format():
    from coreouto.hooks import ON_GOAL_COMPLETE, register_hook

    @register_tool("ping")
    def ping() -> str:
        """Ping."""
        return "pong"

    events = []
    register_hook(ON_GOAL_COMPLETE, lambda **kw: events.append(kw))

    dm = MockSystemOneModel([_iteration_result(0.99, "ping")])
    agent = _make_agent(dm, tools=["ping"])
    response = await agent.call("go")

    assert len(events) == 1
    assert events[0]["state"]["goal"]
    assert response.answer is None


async def test_before_turn_input_observes_and_on_turn_input_mutates():
    from coreouto._types import ImageBlock, TextBlock
    from coreouto.hooks import BEFORE_TURN_INPUT, ON_TURN_INPUT, register_hook

    observed = []

    def before(messages, **kw):
        observed.append(len(messages))

    def redact(blocks, **kw):
        # in-place mutation: drop the screenshot, note the redaction
        blocks.clear()
        blocks.append(TextBlock(text="screenshot redacted"))

    register_hook(BEFORE_TURN_INPUT, before)
    register_hook(ON_TURN_INPUT, redact)

    def observe(messages: list) -> list:
        return [
            TextBlock(text="window: Settings"),
            ImageBlock(data=b"png", mime_type="image/png"),
        ]

    dm = MockSystemOneModel([SystemOneResult(answers={"v": _choice("x")})])
    register_provider("jev", SystemOneProvider(dm, turn_input=observe))
    agent = Agent(AgentConfig(name="t", model="clef", provider="jev"))
    await agent.call(
        "q", answer_format={"v": ChoiceQuestion(instructions="?", criteria={"x": "x"})}
    )

    assert observed == [2]  # system + user message were visible before the call
    call = dm.calls[0]
    assert call["state"]["context"] == "screenshot redacted"
    assert call["kwargs"]["media"] is None


async def test_systemone_collection_hook_collects_results():
    from coreouto.contrib.hooks import systemone_collection_hook
    from coreouto.hooks import AFTER_LLM_CALL, register_hook

    @register_tool("ping")
    def ping() -> str:
        """Ping."""
        return "pong"

    hook, sink = systemone_collection_hook()
    register_hook(AFTER_LLM_CALL, hook)

    dm = MockSystemOneModel([_iteration_result(0.3, "ping"), _iteration_result(0.9, "ping")])
    agent = _make_agent(dm, tools=["ping"])
    await agent.call("go")

    assert len(sink) == 2
    assert sink[0].answers[GOAL_COMPLETE_KEY].noul == 0.3
    assert sink[1].answers[GOAL_COMPLETE_KEY].noul == 0.9


async def test_systemone_collection_hook_skips_chat_providers():
    from coreouto.contrib.hooks import systemone_collection_hook
    from coreouto.hooks import AFTER_LLM_CALL, register_hook

    hook, sink = systemone_collection_hook()
    register_hook(AFTER_LLM_CALL, hook)

    register_provider("mock", MockProvider([MockLLMResponse(content="hi")]))
    agent = Agent(AgentConfig(name="t", model="m", provider="mock"))
    await agent.call("hi")

    assert sink == []


async def test_typesafe_media_becomes_data_uri_extra_body():
    ts = pytest.importorskip("typesafe_sdk")
    from coreouto._types import AudioBlock, ImageBlock
    from coreouto.systemone.typesafe import TypeSafeSystemOneModel

    class FakeClient:
        def __init__(self) -> None:
            self.seen: dict[str, Any] = {}

        async def system_one(self, state, questions, *, model=None, **kwargs):
            self.seen = kwargs
            return ts.SystemOneResponse(
                model="clef",
                usage=ts.Usage(input_tokens=100, output_tokens=None),
                answers={"n": ts.NoulAnswer(type="noul", noul=0.5)},
            )

    client = FakeClient()
    dm = TypeSafeSystemOneModel(client=client)
    await dm.decide(
        "state",
        questions={"n": NoulQuestion(instructions="?")},
        model="clef",
        media=[
            ImageBlock(data=b"\x89PNG", mime_type="image/png"),
            AudioBlock(data=b"RIFF", mime_type="audio/mpeg"),
        ],
    )

    extra_body = client.seen["extra_body"]
    expected_png = "data:image/png;base64," + base64.b64encode(b"\x89PNG").decode()
    assert extra_body["images"] == [expected_png]
    assert extra_body["audio"] == ["data:audio/mpeg;base64,UklGRg=="]


async def test_typesafe_media_rejects_remote_urls():
    pytest.importorskip("typesafe_sdk")
    from coreouto._types import ImageBlock
    from coreouto.systemone.typesafe import TypeSafeSystemOneModel

    dm = TypeSafeSystemOneModel(client=object())
    with pytest.raises(ValueError, match="do not accept remote media URLs"):
        await dm.decide(
            "state",
            questions={"n": NoulQuestion(instructions="?")},
            model="clef",
            media=[ImageBlock(url="https://example.com/shot.png")],
        )


# --- typesafe implementation (SDK mapping, no real API calls) -----------------


async def test_typesafe_systemone_model_maps_questions_and_answers():
    ts = pytest.importorskip("typesafe_sdk")
    from coreouto.systemone.typesafe import TypeSafeSystemOneModel

    class FakeClient:
        def __init__(self) -> None:
            self.seen: dict[str, Any] = {}

        async def system_one(self, state, questions, *, model=None, **kwargs):
            self.seen = {"state": state, "questions": questions, "model": model}
            return ts.SystemOneResponse(
                model="jev-1.13.0",
                usage=ts.Usage(input_tokens=10, output_tokens=5),
                answers={
                    "n": ts.NoulAnswer(type="noul", noul=0.7),
                    "c": ts.ChoiceAnswer(
                        type="choice", choice="x", confidence=0.8, probabilities={"x": 0.8}
                    ),
                    "s": ts.ScoreAnswer(
                        type="score",
                        score=1.0,
                        confidence=0.5,
                        legend={0: "lo", 1: "hi"},
                        probabilities={0: 0.4, 1: 0.6},
                    ),
                },
            )

    client = FakeClient()
    dm = TypeSafeSystemOneModel(client=client)
    result = await dm.decide(
        {"some": "state"},
        questions={
            "n": NoulQuestion(instructions="yes?"),
            "c": ChoiceQuestion(instructions="pick", criteria={"x": "ex"}),
            "s": ScoreQuestion(instructions="rate", criteria=["lo", "hi"]),
        },
        model="jev-latest",
    )

    assert isinstance(client.seen["questions"]["n"], ts.Noul)
    assert isinstance(client.seen["questions"]["c"], ts.Choice)
    assert isinstance(client.seen["questions"]["s"], ts.Score)

    assert result.answers["n"].noul == 0.7
    assert result.answers["c"].choice == "x"
    assert result.answers["s"].probabilities == {"0": 0.4, "1": 0.6}
    assert result.usage is not None
    assert result.usage.total_tokens == 15
    assert result.model == "jev-1.13.0"


def test_typesafe_import_error_without_sdk(monkeypatch):
    import coreouto.systemone.typesafe as typesafe_module

    monkeypatch.setattr(typesafe_module, "AsyncTypeSafeClient", None)
    with pytest.raises(ImportError, match=r"coreouto\[typesafe\]"):
        typesafe_module.TypeSafeSystemOneModel()
