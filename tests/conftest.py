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


@dataclass
class MockLLMResponse:
    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class MockProvider:
    """Provider that yields scripted LLM responses, recording every call."""

    def __init__(self, responses: list[MockLLMResponse] | None = None) -> None:
        self._responses: list[MockLLMResponse] = list(responses or [])
        self.calls: list[dict[str, Any]] = []
        self._index = 0
        self.formatted_assistants: list[Any] = []
        self.formatted_tool_results: list[Any] = []

    def queue(self, *responses: MockLLMResponse) -> None:
        self._responses.extend(responses)

    def _next(self) -> MockLLMResponse:
        if self._index >= len(self._responses):
            raise AssertionError(
                f"MockProvider exhausted: requested response #{self._index + 1} "
                f"but only {len(self._responses)} responses were queued. "
                f"Recorded {len(self.calls)} calls so far."
            )
        r = self._responses[self._index]
        self._index += 1
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
        r = self._next()
        return LLMResponse(
            content=r.content,
            tool_calls=list(r.tool_calls),
            usage=Usage(
                prompt_tokens=r.prompt_tokens,
                completion_tokens=r.completion_tokens,
                total_tokens=r.prompt_tokens + r.completion_tokens,
            ),
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
        return Message(
            role="tool",
            content=str(result.content) if hasattr(result, "content") else str(result),
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
