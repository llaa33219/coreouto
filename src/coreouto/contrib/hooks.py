"""Opt-in hook recipes for coreouto.

Each factory returns a hook callable ready to be passed to
``coreouto.hooks.register_hook``. Recipes that need to share state with the
caller expose that state by returning it alongside the hook as a tuple
``(hook, state)``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

from coreouto._types import LLMResponse, Message, ToolResult, Usage
from coreouto.hooks import (
    AFTER_LLM_CALL,
    AFTER_TOOL_CALL,
    BEFORE_LLM_CALL,
    BEFORE_TOOL_CALL,
    ON_FINISH,
)


def token_collection_hook(
    *, sink: list[Usage] | None = None
) -> tuple[Callable[..., None], list[Usage]]:
    if sink is None:
        sink = []

    def hook(response: LLMResponse, **_kwargs: Any) -> None:
        if response.usage is not None:
            sink.append(response.usage)

    return hook, sink


def auto_summarize_hook(
    *, threshold: int, summarize_fn: Callable[[list[Message]], list[Message]]
) -> Callable[..., None]:
    total: list[int] = [0]

    def hook(
        *, iteration: int, messages: list[Message], response: LLMResponse, **_kwargs: Any
    ) -> None:
        if response.usage is None:
            return
        total[0] += response.usage.total_tokens
        if total[0] >= threshold:
            summarized = summarize_fn(messages)
            messages.clear()
            messages.extend(summarized)

    return hook


def token_limit_warning_hook(
    *, limit: int, callback: Callable[[Usage], Any] | None = None
) -> Callable[..., None]:
    if callback is None:

        def callback(usage: Usage) -> None:
            print(f"WARNING: token limit {limit} exceeded, current {usage.total_tokens}")

    def hook(response: LLMResponse, **_kwargs: Any) -> None:
        if response.usage is not None and response.usage.total_tokens > limit:
            callback(response.usage)

    return hook


def iteration_notification_hook(
    *, every: int = 10, callback: Callable[[int], Any] | None = None
) -> Callable[..., None]:
    if callback is None:

        def callback(iteration: int) -> None:
            print(f"INFO: reached iteration {iteration}")

    def hook(*, iteration: int, **_kwargs: Any) -> None:
        if iteration % every == 0:
            callback(iteration)

    return hook


def tool_usage_collection_hook(
    *, sink: list[tuple[str, str, bool]] | None = None
) -> tuple[Callable[..., None], list[tuple[str, str, bool]]]:
    if sink is None:
        sink = []

    def hook(*, name: str, result: ToolResult, **_kwargs: Any) -> None:
        sink.append((name, result.content, result.is_error))

    return hook, sink


def thinking_printer_hook(*, end: str = "", flush: bool = True) -> Callable[..., None]:
    def hook(*, text: str, **_kwargs: Any) -> None:
        print(text, end=end, flush=flush)

    return hook


def stream_printer_hook(*, end: str = "", flush: bool = True) -> Callable[..., None]:
    def hook(*, text: str, **_kwargs: Any) -> None:
        print(text, end=end, flush=flush)

    return hook


class ActivityState:
    """Shared state returned by activity_tracker_hook()."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self.last_activity_at: float = clock()

    def seconds_since_last_activity(self) -> float:
        return self._clock() - self.last_activity_at


def activity_tracker_hook(
    *, clock: Callable[[], float] = time.monotonic
) -> tuple[Callable[..., None], ActivityState]:
    """Track how long the loop has been silent.

    Register the returned hook on every event you count as activity, e.g.
    AFTER_LLM_CALL, AFTER_TOOL_CALL, ON_ITERATION, ON_STREAM_TEXT,
    ON_PROVIDER_ERROR, ON_USER_INJECTION — each firing resets the clock.
    state.seconds_since_last_activity() then gives the seconds elapsed
    since the loop last did anything (tool result, API call, ...).
    """
    state = ActivityState(clock)

    def hook(**_kwargs: Any) -> None:
        state.last_activity_at = clock()

    return hook, state


class ApiCallState:
    """Shared state returned by api_call_tracker_hook()."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self.last_request_at: float | None = None
        self.last_response_at: float | None = None
        self.last_duration: float | None = None
        self.in_flight: bool = False

    def seconds_since_request(self) -> float | None:
        if self.last_request_at is None:
            return None
        return self._clock() - self.last_request_at

    def seconds_since_response(self) -> float | None:
        if self.last_response_at is None:
            return None
        return self._clock() - self.last_response_at


def api_call_tracker_hook(
    *, clock: Callable[[], float] = time.monotonic
) -> tuple[dict[str, Callable[..., None]], ApiCallState]:
    """Track provider API call timing.

    Returns (hooks, state). Register every hooks entry:

        for event, fn in hooks.items():
            register_hook(event, fn)

    state.in_flight is True between BEFORE_LLM_CALL and AFTER_LLM_CALL;
    state.seconds_since_request() measures from the moment the request
    started. AFTER_LLM_CALL does not fire on provider errors, so in_flight
    stays True through error-rule handling (retries are still one in-flight
    attempt) — register hooks[AFTER_LLM_CALL] on ON_PROVIDER_ERROR too if
    you want errors to close the window.
    """
    state = ApiCallState(clock)

    def before(**_kwargs: Any) -> None:
        state.last_request_at = clock()
        state.in_flight = True

    def after(**_kwargs: Any) -> None:
        now = clock()
        state.last_response_at = now
        state.in_flight = False
        if state.last_request_at is not None:
            state.last_duration = now - state.last_request_at

    return {BEFORE_LLM_CALL: before, AFTER_LLM_CALL: after}, state


class LoopProgressState:
    """Shared state returned by loop_progress_hook()."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self.phase: Literal["llm_call", "tool_call"] | None = None
        self.last_event_at: float = clock()

    def seconds_since_event(self) -> float:
        return self._clock() - self.last_event_at

    def is_stalled(self, after_seconds: float) -> bool:
        return self.seconds_since_event() > after_seconds


def loop_progress_hook(
    *, clock: Callable[[], float] = time.monotonic
) -> tuple[dict[str, Callable[..., None]], LoopProgressState]:
    """Track whether the loop is actively working on something.

    Returns (hooks, state); register every hooks entry as with
    api_call_tracker_hook(). state.phase is "llm_call" or "tool_call"
    while that operation is running and None otherwise (also cleared on
    ON_FINISH). state.is_stalled(after_seconds) is True when no tracked
    event fired for that long — the loop is then either hung inside an
    operation (state.phase tells you which) or wedged between them.
    """
    state = LoopProgressState(clock)

    def _mark(phase: Literal["llm_call", "tool_call"] | None) -> Callable[..., None]:
        def hook(**_kwargs: Any) -> None:
            state.phase = phase
            state.last_event_at = clock()

        return hook

    hooks = {
        BEFORE_LLM_CALL: _mark("llm_call"),
        AFTER_LLM_CALL: _mark(None),
        BEFORE_TOOL_CALL: _mark("tool_call"),
        AFTER_TOOL_CALL: _mark(None),
        ON_FINISH: _mark(None),
    }
    return hooks, state


__all__ = [
    "ActivityState",
    "ApiCallState",
    "LoopProgressState",
    "activity_tracker_hook",
    "api_call_tracker_hook",
    "auto_summarize_hook",
    "iteration_notification_hook",
    "loop_progress_hook",
    "stream_printer_hook",
    "thinking_printer_hook",
    "token_collection_hook",
    "token_limit_warning_hook",
    "tool_usage_collection_hook",
]
