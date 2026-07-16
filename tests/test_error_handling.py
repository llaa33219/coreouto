from __future__ import annotations

from typing import Any

import pytest

from coreouto._types import AgentConfig, ErrorRule, Message
from coreouto.agent import Agent, _match_error_rule
from coreouto.contrib.error_presets import COMMON_HTTP_ERRORS, INVALID_TOOL_ERRORS
from coreouto.hooks import ON_PROVIDER_ERROR, clear_hooks, register_hook
from coreouto.providers import clear_providers, register_provider
from tests.conftest import MockLLMResponse, MockProvider


@pytest.fixture(autouse=True)
def _clean_state():
    clear_hooks()
    clear_providers()
    yield
    clear_hooks()
    clear_providers()


class FakeStatusError(Exception):
    def __init__(self, status_code: int, message: str = "") -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(message or f"HTTP {status_code}")


class FakeCodeError(Exception):
    def __init__(self, code: int, message: str = "") -> None:
        self.code = code
        self.message = message
        super().__init__(message or f"code={code}")


class RaisingProvider:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.error_handling: list[ErrorRule] | None = None
        self.calls: list[dict[str, Any]] = []

    async def create(self, messages, *, model, tools=None, system_prompt=None, **kwargs):
        self.calls.append({"messages": list(messages), "model": model})
        raise self._exc

    def format_assistant_message(self, response):
        from coreouto._types import Message

        return Message(role="assistant", content=response.content or "")

    def format_tool_result(self, tool_call, result):
        return Message(
            role="tool",
            content=result.content,
            tool_call_id=tool_call.id,
            name=tool_call.name,
        )


class TransientProvider:
    """Fails the first N calls, then delegates to inner mock."""

    def __init__(self, inner: MockProvider, fail_times: int, exc: BaseException) -> None:
        self._inner = inner
        self._fail_times = fail_times
        self._exc = exc
        self.error_handling: list[ErrorRule] | None = None
        self.create_call_count = 0

    async def create(self, messages, *, model, tools=None, system_prompt=None, **kwargs):
        self.create_call_count += 1
        if self.create_call_count <= self._fail_times:
            raise self._exc
        return await self._inner.create(
            messages, model=model, tools=tools, system_prompt=system_prompt, **kwargs
        )

    def format_assistant_message(self, response):
        return self._inner.format_assistant_message(response)

    def format_tool_result(self, tool_call, result):
        return self._inner.format_tool_result(tool_call, result)


def _config(provider: str = "mock") -> AgentConfig:
    return AgentConfig(name="test", model="m", provider=provider)


# ===========================================================================
# _match_error_rule
# ===========================================================================


class TestMatchErrorRule:
    def test_match_by_status_code(self):
        rules = [ErrorRule(status_code=429, reaction="user_message", message="slow")]
        assert _match_error_rule(FakeStatusError(429), rules) is rules[0]

    def test_no_match_different_status(self):
        rules = [ErrorRule(status_code=429, reaction="user_message", message="slow")]
        assert _match_error_rule(FakeStatusError(500), rules) is None

    def test_match_by_content_contains(self):
        rules = [
            ErrorRule(
                status_code=400,
                content_contains="context_length",
                reaction="terminate",
                message="too long",
            )
        ]
        exc = FakeStatusError(400, "context_length_exceeded")
        assert _match_error_rule(exc, rules) is rules[0]

    def test_no_match_content_not_found(self):
        rules = [
            ErrorRule(
                status_code=400,
                content_contains="context_length",
                reaction="terminate",
                message="too long",
            )
        ]
        exc = FakeStatusError(400, "invalid_tool_name")
        assert _match_error_rule(exc, rules) is None

    def test_match_both_status_and_content(self):
        rules = [
            ErrorRule(
                status_code=400,
                content_contains="tool",
                reaction="tool_result",
                message="bad tool",
            )
        ]
        assert _match_error_rule(FakeStatusError(400, "invalid tool args"), rules) is rules[0]
        assert _match_error_rule(FakeStatusError(500, "invalid tool args"), rules) is None
        assert _match_error_rule(FakeStatusError(400, "context too long"), rules) is None

    def test_first_matching_rule_wins(self):
        rules = [
            ErrorRule(status_code=400, reaction="terminate", message="first"),
            ErrorRule(status_code=400, reaction="user_message", message="second"),
        ]
        assert _match_error_rule(FakeStatusError(400), rules).message == "first"

    def test_match_by_code_attribute_google_style(self):
        rules = [ErrorRule(status_code=429, reaction="user_message", message="slow")]
        assert _match_error_rule(FakeCodeError(429), rules) is rules[0]

    def test_empty_rules_returns_none(self):
        assert _match_error_rule(FakeStatusError(500), []) is None


# ===========================================================================
# Reaction: terminate
# ===========================================================================


class TestTerminateReaction:
    @pytest.mark.asyncio
    async def test_terminate_returns_response_with_message(self):
        provider = RaisingProvider(FakeStatusError(401, "bad key"))
        provider.error_handling = [
            ErrorRule(status_code=401, reaction="terminate", message="auth failed"),
        ]
        register_provider("mock", provider)

        agent = Agent(_config())
        response = await agent.call("hello")

        assert response.content == "auth failed"
        assert response.stop_reason == "failed"


# ===========================================================================
# Reaction: user_message
# ===========================================================================


class TestUserMessageReaction:
    @pytest.mark.asyncio
    async def test_user_message_injects_and_continues(self):
        mock = MockProvider(
            [MockLLMResponse(content="recovered", prompt_tokens=5, completion_tokens=5)],
            provider_name="mock",
        )
        provider = RaisingProvider(FakeStatusError(429, "slow down"))
        provider.error_handling = [
            ErrorRule(status_code=429, reaction="user_message", message="rate limited"),
        ]

        class ChainProvider:
            def __init__(self):
                self.error_handling = None
                self._call = 0

            async def create(self, messages, *, model, tools=None, system_prompt=None, **kwargs):
                self._call += 1
                if self._call == 1:
                    raise FakeStatusError(429, "slow down")
                return await mock.create(
                    messages, model=model, tools=tools, system_prompt=system_prompt, **kwargs
                )

            def format_assistant_message(self, response):
                return mock.format_assistant_message(response)

            def format_tool_result(self, tool_call, result):
                return mock.format_tool_result(tool_call, result)

        chain = ChainProvider()
        chain.error_handling = provider.error_handling
        register_provider("mock", chain)

        agent = Agent(_config())
        response = await agent.call("hello")

        assert response.content == "recovered"
        # The injected user message should be in the second call's messages
        second_messages = mock.calls[0]["messages"]
        injected = [
            m
            for m in second_messages
            if getattr(m, "role", None) == "user"
            and "rate limited" in str(getattr(m, "content", ""))
        ]
        assert len(injected) == 1


# ===========================================================================
# Reaction: tool_result
# ===========================================================================


class TestToolResultReaction:
    @pytest.mark.asyncio
    async def test_tool_result_appends_error_for_last_tool_calls(self):
        mock = MockProvider(
            [
                MockLLMResponse(
                    content=None,
                    tool_calls=[{"id": "tc1", "name": "search", "arguments": {"q": "test"}}],
                    prompt_tokens=5,
                    completion_tokens=5,
                ),
                MockLLMResponse(content="fixed", prompt_tokens=5, completion_tokens=5),
            ],
            provider_name="mock",
        )
        register_provider("mock", mock)

        from coreouto.tools import clear_tools, register_tool

        @register_tool("search")
        def search(q: str) -> str:
            return f"results for {q}"

        try:
            agent = Agent(_config())
            # First iteration: model calls "search" tool
            # Tool executes, result appended, loop continues
            # Second iteration: we'll simulate a 400 error

            # Actually, for this test we need the error to happen during provider.create
            # and then have tool_result reaction kick in.
            # Let's use a chain provider that fails on the second call.
            class FailingChain:
                def __init__(self):
                    self.error_handling = [
                        ErrorRule(
                            status_code=400,
                            content_contains="tool",
                            reaction="tool_result",
                            message="tool rejected",
                        )
                    ]
                    self._call = 0

                async def create(
                    self, messages, *, model, tools=None, system_prompt=None, **kwargs
                ):
                    self._call += 1
                    if self._call == 2:
                        raise FakeStatusError(400, "invalid tool arguments")
                    return await mock.create(
                        messages, model=model, tools=tools, system_prompt=system_prompt, **kwargs
                    )

                def format_assistant_message(self, response):
                    return mock.format_assistant_message(response)

                def format_tool_result(self, tool_call, result):
                    return mock.format_tool_result(tool_call, result)

            chain = FailingChain()
            register_provider("mock", chain)
            agent = Agent(_config())
            response = await agent.call("search for test")

            assert response.content == "fixed"
        finally:
            clear_tools()


# ===========================================================================
# Hook: ON_PROVIDER_ERROR
# ===========================================================================


class TestOnProviderErrorHook:
    @pytest.mark.asyncio
    async def test_hook_carries_error_and_reaction_info(self):
        provider = RaisingProvider(FakeStatusError(401, "bad key"))
        provider.error_handling = [
            ErrorRule(status_code=401, reaction="terminate", message="auth failed"),
        ]
        register_provider("mock", provider)

        events: list[dict[str, Any]] = []

        def on_error(**kwargs):
            events.append(kwargs)

        register_hook(ON_PROVIDER_ERROR, on_error)

        agent = Agent(_config("mock"))
        response = await agent.call("hello")

        assert response.content == "auth failed"
        assert len(events) == 1
        evt = events[0]
        assert evt["status_code"] == 401
        assert "bad key" in evt["error_message"]
        assert evt["reaction"] == "terminate"
        assert evt["reaction_message"] == "auth failed"

    @pytest.mark.asyncio
    async def test_hook_not_fired_when_no_rules(self):
        provider = RaisingProvider(FakeStatusError(500))
        provider.error_handling = None
        register_provider("mock", provider)

        events: list[dict[str, Any]] = []

        def on_error(**kwargs):
            events.append(kwargs)

        register_hook(ON_PROVIDER_ERROR, on_error)

        agent = Agent(_config("mock"))
        with pytest.raises(FakeStatusError):
            await agent.call("hello")

        assert len(events) == 0


# ===========================================================================
# Reaction: retry
# ===========================================================================


class TestRetryReaction:
    @pytest.mark.asyncio
    async def test_retry_succeeds_after_transient_failure(self):
        mock = MockProvider(
            [MockLLMResponse(content="ok", prompt_tokens=5, completion_tokens=5)],
            provider_name="mock",
        )
        provider = TransientProvider(mock, fail_times=1, exc=FakeStatusError(429))
        provider.error_handling = [
            ErrorRule(
                status_code=429,
                reaction="retry",
                message="rate limited",
                retry_after=0,
                retry_max=3,
            ),
        ]
        register_provider("mock", provider)

        agent = Agent(_config())
        response = await agent.call("hello")

        assert response.content == "ok"
        assert provider.create_call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted_propagates(self):
        provider = TransientProvider(
            MockProvider([], provider_name="mock"),
            fail_times=99,
            exc=FakeStatusError(429),
        )
        provider.error_handling = [
            ErrorRule(
                status_code=429,
                reaction="retry",
                message="rate limited",
                retry_after=0,
                retry_max=2,
            ),
        ]
        register_provider("mock", provider)

        agent = Agent(_config())
        with pytest.raises(FakeStatusError):
            await agent.call("hello")

        assert provider.create_call_count == 3

    @pytest.mark.asyncio
    async def test_retry_fires_hook_each_attempt(self):
        mock = MockProvider(
            [MockLLMResponse(content="ok", prompt_tokens=5, completion_tokens=5)],
            provider_name="mock",
        )
        provider = TransientProvider(mock, fail_times=2, exc=FakeStatusError(429))
        provider.error_handling = [
            ErrorRule(
                status_code=429,
                reaction="retry",
                message="rate limited",
                retry_after=0,
                retry_max=5,
            ),
        ]
        register_provider("mock", provider)

        events: list[dict[str, Any]] = []
        register_hook(ON_PROVIDER_ERROR, lambda **kw: events.append(kw))

        agent = Agent(_config())
        await agent.call("hello")

        assert len(events) == 2
        assert all(e["reaction"] == "retry" for e in events)

    @pytest.mark.asyncio
    async def test_retry_different_error_propagates(self):
        mock = MockProvider(
            [MockLLMResponse(content="ok", prompt_tokens=5, completion_tokens=5)],
            provider_name="mock",
        )

        class SwitchingProvider:
            def __init__(self):
                self.error_handling = [
                    ErrorRule(
                        status_code=429,
                        reaction="retry",
                        message="rate",
                        retry_after=0,
                        retry_max=5,
                    ),
                ]
                self._call = 0

            async def create(self, messages, *, model, tools=None, system_prompt=None, **kwargs):
                self._call += 1
                if self._call == 1:
                    raise FakeStatusError(429)
                raise FakeStatusError(401, "auth failed")

            def format_assistant_message(self, response):
                return mock.format_assistant_message(response)

            def format_tool_result(self, tool_call, result):
                return mock.format_tool_result(tool_call, result)

        register_provider("mock", SwitchingProvider())

        agent = Agent(_config())
        with pytest.raises(FakeStatusError) as exc_info:
            await agent.call("hello")
        assert exc_info.value.status_code == 401


# ===========================================================================
# Contrib presets
# ===========================================================================


class TestContribPresets:
    def test_common_http_errors_has_retry_for_rate_limit(self):
        rate_limit_rule = next(r for r in COMMON_HTTP_ERRORS if r.status_code == 429)
        assert rate_limit_rule.reaction == "retry"
        assert rate_limit_rule.retry_max >= 1
        assert rate_limit_rule.retry_backoff >= 1.0

    def test_common_http_errors_has_terminate_for_auth(self):
        auth_rule = next(r for r in COMMON_HTTP_ERRORS if r.status_code == 401)
        assert auth_rule.reaction == "terminate"

    def test_invalid_tool_errors_has_tool_result_reactions(self):
        assert all(r.reaction == "tool_result" for r in INVALID_TOOL_ERRORS)

    def test_presets_are_composable(self):
        combined = COMMON_HTTP_ERRORS + INVALID_TOOL_ERRORS
        assert len(combined) == len(COMMON_HTTP_ERRORS) + len(INVALID_TOOL_ERRORS)

    def test_presets_are_mutable_lists(self):
        extended = list(COMMON_HTTP_ERRORS)
        extended.append(
            ErrorRule(
                status_code=400,
                content_contains="context",
                reaction="terminate",
                message="too long",
            )
        )
        assert len(extended) == len(COMMON_HTTP_ERRORS) + 1
        assert len(COMMON_HTTP_ERRORS) < len(extended)
