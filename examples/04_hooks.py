"""Example 04: Hooks.

Three ways to attach hooks:
  1. A custom sync hook on BEFORE_LLM_CALL, defined inline.
  2. The contrib `token_collection_hook` on AFTER_LLM_CALL, which
     returns `(hook, sink)` — the sink is a list we can inspect later.
  3. The contrib `iteration_notification_hook(every=...)` on ON_ITERATION,
     which prints every Nth iteration.

The hook event names are the string constants exported from coreouto.

Run with:
    export GOOGLE_API_KEY=...
    python examples/04_hooks.py
"""

from __future__ import annotations

import asyncio
import os

import coreouto as co
from coreouto._types import AgentConfig
from coreouto.contrib.hooks import (
    iteration_notification_hook,
    token_collection_hook,
)


def print_message_count(*, messages, **_kwargs) -> None:
    print(f"[before_llm_call] {len(messages)} message(s) queued")


async def main() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for this example")

    co.providers.google.register(api_key=api_key)

    co.register_hook(co.BEFORE_LLM_CALL, print_message_count)

    tokens_hook, tokens = token_collection_hook()
    co.register_hook(co.AFTER_LLM_CALL, tokens_hook)

    co.register_hook(co.ON_ITERATION, iteration_notification_hook(every=1))

    config = AgentConfig(
        name="hooked",
        model="gemini-2.5-pro",
        provider="google",
        system_prompt="You are a helpful assistant.",
    )
    response = await co.Agent(config).call("Say hello in exactly one word.")
    print("Response:", response.content)
    print("Collected token snapshots:", tokens)


if __name__ == "__main__":
    asyncio.run(main())
