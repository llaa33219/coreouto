"""Synchronous wrapper for the async agent call."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coreouto._types import Message, Response
    from coreouto.agent import Agent


def call_sync(
    agent: Agent,
    user_message: str,
    *,
    override=None,
    history: list[Message] | None = None,
) -> Response:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(agent.call(user_message, override=override, history=history))
    raise RuntimeError(
        "call_sync() cannot be used inside a running event loop. "
        "Use 'await agent.call(...)' instead."
    )
