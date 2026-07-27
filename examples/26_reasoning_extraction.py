"""Example 26: Extracting the model's reasoning process.

Two complementary ways to get at the reasoning ("thinking") a model
produces on the way to its answer:

* ``on_stream_thinking`` + the contrib ``thinking_printer_hook`` — fires for
  each reasoning *fragment* during a streaming call. The contrib recipe just
  prints fragments as they arrive; register your own callable to route them
  anywhere (UI, log stream, ...).

* ``on_thinking`` — fires once per LLM call that produced reasoning, with the
  *complete* thinking text for that turn in the ``thinking`` kwarg. This is
  the extraction point: collect turns into a sink for auditing, evals, or
  debugging. Works with and without streaming.

The full reasoning is also available as ``response.thinking`` on the
``LLMResponse`` in an ``after_llm_call`` hook.

Run with:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/26_reasoning_extraction.py
"""

from __future__ import annotations

import asyncio
import os

import coreouto as co
from coreouto._types import AgentConfig
from coreouto.contrib.hooks import thinking_printer_hook


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for this example")

    co.providers.anthropic.register(api_key=api_key, stream=True)

    # 1. Live reasoning: print each thinking fragment as it streams in.
    co.register_hook(co.ON_STREAM_THINKING, thinking_printer_hook())

    # 2. Extraction: collect the complete reasoning of every turn.
    reasoning_turns: list[str] = []

    def collect_thinking(*, thinking: str, **_kwargs: object) -> None:
        reasoning_turns.append(thinking)

    co.register_hook(co.ON_THINKING, collect_thinking)

    agent = co.Agent(
        AgentConfig(
            name="reasoner",
            model="claude-sonnet-4-6",
            provider="anthropic",
            system_prompt="Think carefully, then respond with your answer.",
            provider_config={
                "reasoning_effort": "medium",
                "max_tokens": 4000,
            },
        )
    )

    response = await agent.call(
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
    )
    print(f"\n\n[final] {response.content}")
    print(f"[extracted] {len(reasoning_turns)} reasoning turn(s)")
    for i, thinking in enumerate(reasoning_turns, start=1):
        print(f"  turn {i}: {thinking[:80]}...")


if __name__ == "__main__":
    asyncio.run(main())
