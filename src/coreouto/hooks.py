from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

BEFORE_LLM_CALL = "before_llm_call"
AFTER_LLM_CALL = "after_llm_call"
BEFORE_TOOL_CALL = "before_tool_call"
AFTER_TOOL_CALL = "after_tool_call"
ON_ITERATION = "on_iteration"
ON_FINISH = "on_finish"
ON_USER_INJECTION = "on_user_injection"
ON_RETRY = "on_retry"
ON_STREAM_TEXT = "on_stream_text"
ON_STREAM_THINKING = "on_stream_thinking"
ON_THINKING = "on_thinking"

_HOOKS: dict[str, list[Callable[..., Any]]] = {}


def register_hook(event: str, fn: Callable[..., Any]) -> None:
    if event not in _HOOKS:
        _HOOKS[event] = []
    _HOOKS[event].append(fn)


def get_hooks(event: str) -> list[Callable[..., Any]]:
    return list(_HOOKS.get(event, []))


def clear_hooks(event: str | None = None) -> None:
    if event is None:
        _HOOKS.clear()
    else:
        _HOOKS.pop(event, None)


async def trigger(event: str, **kwargs: Any) -> None:
    for fn in _HOOKS.get(event, []):
        if inspect.iscoroutinefunction(fn):
            await fn(**kwargs)
        else:
            fn(**kwargs)
