"""Opt-in hook recipes for coreouto.

Each factory returns a hook callable ready to be passed to
``coreouto.hooks.register_hook``. Recipes that need to share state with the
caller expose that state by returning it alongside the hook as a tuple
``(hook, state)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from coreouto._types import LLMResponse, Message, ToolResult, Usage


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


__all__ = [
    "auto_summarize_hook",
    "iteration_notification_hook",
    "token_collection_hook",
    "token_limit_warning_hook",
    "tool_usage_collection_hook",
]
