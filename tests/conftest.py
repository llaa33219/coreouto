"""Shared test fixtures and the canonical MockProvider.

MockProvider is the single seam the entire test suite uses to avoid real
API calls. Each test constructs a `MockProvider` with a list of canned
`LLMResponse` objects (in order) and inspects `provider.calls` afterward.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

# Per-provider natural end-of-turn values. coreouto maps each provider's
# native field (Anthropic `stop_reason`, OpenAI Chat `finish_reason`,
# OpenAI Responses `status`, Gemini `finishReason`) into a single string
# exposed on `LLMResponse.stop_reason`. Tests use this table so a response
# without an explicit `stop_reason` still terminates the loop when run
# against any of the supported providers.
_PROVIDER_END_REASONS: dict[str, str] = {
    "anthropic": "end_turn",
    "openai": "stop",
    "openai-response": "completed",
    "google": "STOP",
}

# Sentinel for `MockLLMResponse.stop_reason`: distinguishes "use the
# provider's default end-of-turn value" from "the value is literally `None`".
_UNSET: Any = object()


@dataclass
class MockLLMResponse:
    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Default (`_UNSET`) means "fill in the provider's natural end-of-turn
    # value when the response is consumed". Tests that want a non-terminating
    # or provider-specific stop_reason (e.g. `"tool_use"`, `"SAFETY"`, or
    # `None` to simulate a legacy SDK) set it explicitly.
    stop_reason: Any = _UNSET


class MockProvider:
    """Provider that yields scripted LLM responses, recording every call."""

    def __init__(
        self,
        responses: list[MockLLMResponse] | None = None,
        *,
        provider_name: str | None = None,
    ) -> None:
        self._responses: list[MockLLMResponse] = list(responses or [])
        self._provider_name = provider_name
        self.calls: list[dict[str, Any]] = []
        self._index = 0
        self.formatted_assistants: list[Any] = []
        self.formatted_tool_results: list[Any] = []

    def queue(self, *responses: MockLLMResponse) -> None:
        self._responses.extend(responses)

    def _next(self, provider: str) -> MockLLMResponse:
        if self._index >= len(self._responses):
            raise AssertionError(
                f"MockProvider exhausted: requested response #{self._index + 1} "
                f"but only {len(self._responses)} responses were queued. "
                f"Recorded {len(self.calls)} calls so far."
            )
        r = self._responses[self._index]
        self._index += 1
        if r.stop_reason is _UNSET:
            r.stop_reason = _PROVIDER_END_REASONS.get(provider)
        return r

    async def create(
        self,
        messages: list[Any],
        *,
        model: str,
        tools: list[Any] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> Any:
        from coreouto._types import LLMResponse, Usage

        self.calls.append(
            {
                "messages": list(messages),
                "model": model,
                "tools": list(tools) if tools else [],
                "system_prompt": system_prompt,
                "kwargs": kwargs,
            }
        )
        r = self._next(self._provider_name or "")
        return LLMResponse(
            content=r.content,
            tool_calls=list(r.tool_calls),
            usage=Usage(
                prompt_tokens=r.prompt_tokens,
                completion_tokens=r.completion_tokens,
                total_tokens=r.prompt_tokens + r.completion_tokens,
            ),
            stop_reason=r.stop_reason,
            raw=r,
        )

    def format_assistant_message(self, response: Any) -> Any:
        from coreouto._types import Message

        self.formatted_assistants.append(response)
        return Message(
            role="assistant",
            content=response.content or "",
            tool_calls=response.tool_calls if response.tool_calls else None,
        )

    def format_tool_result(self, tool_call: Any, result: Any) -> Any:
        from coreouto._types import Message

        self.formatted_tool_results.append((tool_call, result))
        if hasattr(result, "blocks") and result.blocks is not None:
            content: Any = list(result.blocks)
        elif hasattr(result, "content") and isinstance(result.content, str):
            content = result.content
        else:
            content = str(getattr(result, "content", result))
        return Message(
            role="tool",
            content=content,
            tool_call_id=tool_call.id if hasattr(tool_call, "id") else tool_call["id"],
            name=tool_call.name if hasattr(tool_call, "name") else tool_call["name"],
        )


class HookRecorder:
    """Records every hook invocation in order."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def make(self) -> Callable[[str], Callable]:
        def factory(event: str) -> Callable:
            def hook(*args: Any, **kwargs: Any) -> None:
                self.events.append((event, {"args": args, "kwargs": kwargs}))

            return hook

        return factory


@pytest.fixture
def hook_recorder() -> HookRecorder:
    return HookRecorder()


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()
