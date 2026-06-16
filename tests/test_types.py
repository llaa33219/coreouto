from __future__ import annotations

import pytest
from pydantic import ValidationError

from coreouto._types import (
    AgentConfig,
    LLMResponse,
    Message,
    Response,
    ToolCall,
    ToolResult,
    Usage,
)


class TestToolCall:
    def test_required_fields_only(self) -> None:
        tc = ToolCall(id="call_1", name="search", arguments={})
        assert tc.id == "call_1"
        assert tc.name == "search"
        assert tc.arguments == {}

    def test_all_fields(self) -> None:
        tc = ToolCall(id="call_1", name="search", arguments={"query": "fusion"})
        assert tc.arguments == {"query": "fusion"}

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            ToolCall(id="call_1", arguments={})  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            ToolCall(name="search", arguments={})  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            ToolCall()  # type: ignore[call-arg]

    def test_round_trip_json(self) -> None:
        tc = ToolCall(id="call_1", name="search", arguments={"q": "x", "n": 2})
        rt = ToolCall.model_validate_json(tc.model_dump_json())
        assert rt == tc

    def test_arguments_accepts_arbitrary_dict(self) -> None:
        tc = ToolCall(
            id="c",
            name="n",
            arguments={"a": 1, "b": "two", "c": [1, 2], "d": {"nested": True}},
        )
        assert tc.arguments["d"]["nested"] is True


class TestToolResult:
    def test_required_fields(self) -> None:
        tr = ToolResult(tool_call_id="call_1", content="ok")
        assert tr.tool_call_id == "call_1"
        assert tr.content == "ok"
        assert tr.is_error is False

    def test_is_error_true(self) -> None:
        tr = ToolResult(tool_call_id="call_1", content="boom", is_error=True)
        assert tr.is_error is True

    def test_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            ToolResult(tool_call_id="call_1")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            ToolResult(content="ok")  # type: ignore[call-arg]

    def test_round_trip_json(self) -> None:
        tr = ToolResult(tool_call_id="c1", content="result", is_error=True)
        rt = ToolResult.model_validate_json(tr.model_dump_json())
        assert rt == tr


class TestMessage:
    def test_user_message_minimal(self) -> None:
        m = Message(role="user", content="hi")
        assert m.role == "user"
        assert m.content == "hi"
        assert m.tool_calls is None
        assert m.tool_call_id is None
        assert m.name is None

    def test_system_message(self) -> None:
        m = Message(role="system", content="You are a helper.")
        assert m.role == "system"

    def test_assistant_with_tool_calls(self) -> None:
        tc = ToolCall(id="c1", name="search", arguments={"q": "x"})
        m = Message(role="assistant", content="", tool_calls=[tc])
        assert m.role == "assistant"
        assert m.tool_calls is not None
        assert len(m.tool_calls) == 1
        assert m.tool_calls[0].name == "search"

    def test_tool_message(self) -> None:
        m = Message(
            role="tool",
            content="result text",
            tool_call_id="c1",
            name="search",
        )
        assert m.role == "tool"
        assert m.tool_call_id == "c1"
        assert m.name == "search"
        assert m.tool_calls is None

    def test_all_roles_accepted(self) -> None:
        for role in ("system", "user", "assistant", "tool"):
            m = Message(role=role, content="x")  # type: ignore[arg-type]
            assert m.role == role

    def test_invalid_role_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Message(role="manager", content="x")  # type: ignore[arg-type]

    def test_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            Message(content="x")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            Message(role="user")  # type: ignore[call-arg]

    def test_round_trip_json(self) -> None:
        tc = ToolCall(id="c1", name="search", arguments={"q": "x"})
        m = Message(
            role="assistant",
            content="text",
            tool_calls=[tc],
        )
        rt = Message.model_validate_json(m.model_dump_json())
        assert rt == m

    def test_tool_round_trip_json(self) -> None:
        m = Message(
            role="tool",
            content="res",
            tool_call_id="c1",
            name="search",
        )
        rt = Message.model_validate_json(m.model_dump_json())
        assert rt == m


class TestUsage:
    def test_valid_construction(self) -> None:
        u = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert u.prompt_tokens == 10
        assert u.completion_tokens == 20
        assert u.total_tokens == 30

    def test_zero_tokens(self) -> None:
        u = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        assert u.total_tokens == 0

    def test_validator_passes_when_total_matches(self) -> None:
        Usage(prompt_tokens=5, completion_tokens=7, total_tokens=12)

    def test_validator_fails_when_total_mismatch(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Usage(prompt_tokens=5, completion_tokens=7, total_tokens=999)
        assert "total_tokens" in str(exc_info.value)

    def test_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            Usage(prompt_tokens=1, completion_tokens=2)  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            Usage(prompt_tokens=1, total_tokens=1)  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            Usage(completion_tokens=1, total_tokens=1)  # type: ignore[call-arg]

    def test_round_trip_json(self) -> None:
        u = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        rt = Usage.model_validate_json(u.model_dump_json())
        assert rt == u


class TestLLMResponse:
    def test_minimal(self) -> None:
        r = LLMResponse()
        assert r.content is None
        assert r.tool_calls == []
        assert r.usage is None
        assert r.raw is None

    def test_with_content(self) -> None:
        r = LLMResponse(content="hello")
        assert r.content == "hello"
        assert r.tool_calls == []

    def test_with_tool_calls(self) -> None:
        tc = ToolCall(id="c1", name="search", arguments={"q": "x"})
        r = LLMResponse(content="", tool_calls=[tc])
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "search"

    def test_with_usage(self) -> None:
        u = Usage(prompt_tokens=5, completion_tokens=7, total_tokens=12)
        r = LLMResponse(content="hi", usage=u)
        assert r.usage == u

    def test_with_raw_arbitrary_object(self) -> None:
        sentinel = object()
        r = LLMResponse(content="hi", raw=sentinel)
        assert r.raw is sentinel

    def test_with_raw_dict(self) -> None:
        raw = {"id": "x", "choices": [{"finish_reason": "stop"}]}
        r = LLMResponse(content="hi", raw=raw)
        assert r.raw == raw

    def test_full_construction(self) -> None:
        tc = ToolCall(id="c1", name="search", arguments={})
        u = Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        r = LLMResponse(
            content="text",
            tool_calls=[tc],
            usage=u,
            raw={"k": "v"},
        )
        assert r.content == "text"
        assert r.tool_calls[0] == tc
        assert r.usage == u
        assert r.raw == {"k": "v"}

    def test_round_trip_json(self) -> None:
        tc = ToolCall(id="c1", name="search", arguments={"q": "x"})
        u = Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        r = LLMResponse(content="hi", tool_calls=[tc], usage=u)
        rt = LLMResponse.model_validate_json(r.model_dump_json())
        assert rt.content == r.content
        assert rt.tool_calls == r.tool_calls
        assert rt.usage == r.usage


class TestAgentConfig:
    def test_required_fields(self) -> None:
        cfg = AgentConfig(name="a", model="gpt-4", provider="openai")
        assert cfg.name == "a"
        assert cfg.model == "gpt-4"
        assert cfg.provider == "openai"
        assert cfg.system_prompt is None
        assert cfg.tools == []
        assert cfg.max_iterations is None

    def test_all_fields(self) -> None:
        cfg = AgentConfig(
            name="researcher",
            model="gpt-4",
            provider="openai",
            system_prompt="You are a researcher.",
            tools=["search", "fetch"],
            max_iterations=10,
        )
        assert cfg.system_prompt == "You are a researcher."
        assert cfg.tools == ["search", "fetch"]
        assert cfg.max_iterations == 10

    def test_provider_config_defaults_to_empty_dict(self) -> None:
        cfg = AgentConfig(name="a", model="m", provider="p")
        assert cfg.provider_config == {}

    def test_provider_config_accepts_arbitrary_keys(self) -> None:
        cfg = AgentConfig(
            name="a",
            model="m",
            provider="p",
            provider_config={"temperature": 0.5, "max_tokens": 100, "top_p": 0.9},
        )
        assert cfg.provider_config["temperature"] == 0.5
        assert cfg.provider_config["max_tokens"] == 100
        assert cfg.provider_config["top_p"] == 0.9

    def test_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(model="m", provider="p")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            AgentConfig(name="a", provider="p")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            AgentConfig(name="a", model="m")  # type: ignore[call-arg]

    def test_round_trip_json(self) -> None:
        cfg = AgentConfig(
            name="a",
            model="m",
            provider="p",
            system_prompt="s",
            tools=["t1", "t2"],
            max_iterations=5,
            provider_config={"temperature": 0.7},
        )
        rt = AgentConfig.model_validate_json(cfg.model_dump_json())
        assert rt == cfg


class TestResponse:
    def test_required_fields(self) -> None:
        m1 = Message(role="user", content="hi")
        m2 = Message(role="assistant", content="hello")
        r = Response(content="hello", messages=[m1, m2], iterations=1)
        assert r.content == "hello"
        assert len(r.messages) == 2
        assert r.iterations == 1
        assert r.usage == []
        assert r.finish_called is True
        assert r.warnings == []

    def test_all_fields(self) -> None:
        m1 = Message(role="user", content="hi")
        u = Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        r = Response(
            content="done",
            messages=[m1],
            iterations=3,
            usage=[u, u],
            finish_called=False,
            warnings=["warn1", "warn2"],
        )
        assert r.iterations == 3
        assert len(r.usage) == 2
        assert r.finish_called is False
        assert r.warnings == ["warn1", "warn2"]

    def test_warnings_defaults_to_empty_list(self) -> None:
        r = Response(content="x", messages=[], iterations=0)
        assert r.warnings == []

    def test_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            Response(messages=[], iterations=1)  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            Response(content="x", iterations=1)  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            Response(content="x", messages=[])  # type: ignore[call-arg]

    def test_round_trip_json(self) -> None:
        m = Message(role="user", content="hi")
        u = Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        r = Response(
            content="done",
            messages=[m],
            iterations=2,
            usage=[u],
            finish_called=True,
            warnings=["w1"],
        )
        rt = Response.model_validate_json(r.model_dump_json())
        assert rt == r

    def test_empty_messages_allowed(self) -> None:
        r = Response(content="", messages=[], iterations=0)
        assert r.messages == []
        assert r.iterations == 0
