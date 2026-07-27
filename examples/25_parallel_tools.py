"""Example 25: Parallel tool calls.

Two halves make parallel tool calling work:

1. The MODEL must emit multiple tool calls in a single turn. Models don't
   do this reliably by default — they tend to call one tool, wait for the
   result, then call the next. The system prompt must teach them: when
   calls are independent, emit ALL of them in ONE turn.

2. coreouto must be allowed to run them concurrently. Set
   `AgentConfig.parallel_tool_calls=True` so the loop dispatches the
   turn's tool calls with asyncio.gather instead of one at a time.
   (Per-tool opt-out: @register_tool(..., parallelizable=False).)

The tools below each sleep 1s. Serial execution would take ~3s per turn;
parallel execution takes ~1s.

Run with:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/25_parallel_tools.py
"""

from __future__ import annotations

import asyncio
import os
import time

import coreouto as co
from coreouto._types import AgentConfig


@co.register_tool("get_weather")
async def get_weather(city: str) -> str:
    """Return a short, fabricated weather report for `city`."""
    await asyncio.sleep(1)  # simulate network latency
    return f"It is 22C and sunny in {city}."


@co.register_tool("get_time")
async def get_time(city: str) -> str:
    """Return the fabricated local time for `city`."""
    await asyncio.sleep(1)
    return f"It is 3:00 PM in {city}."


@co.register_tool("get_population")
async def get_population(city: str) -> str:
    """Return the fabricated population of `city`."""
    await asyncio.sleep(1)
    return f"{city} has about 14 million people."


# The prompt is the important part of this example. Without explicit
# instructions, most models call tools one at a time even when the calls
# are independent.
SYSTEM_PROMPT = """\
You are a research assistant with access to city-data tools.

How to call tools:
- When you need several INDEPENDENT pieces of information, request ALL of \
them in a SINGLE turn as parallel tool calls. Do NOT call one tool, wait \
for its result, and then call the next — batch the independent calls together.
- Only call a tool in a later turn when its arguments depend on a previous \
tool's result.
- When you have everything you need, respond with your final answer as text \
with no tool calls.
"""


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for this example")

    co.providers.anthropic.register(api_key=api_key)

    config = AgentConfig(
        name="city-facts",
        model="claude-opus-4-8",
        provider="anthropic",
        system_prompt=SYSTEM_PROMPT,
        tools=["get_weather", "get_time", "get_population"],
        # Without this flag the loop still accepts multiple tool calls in a
        # turn but executes them serially.
        parallel_tool_calls=True,
    )
    agent = co.Agent(config)

    start = time.monotonic()
    response = await agent.call("Tell me the weather, local time, and population of Tokyo.")
    elapsed = time.monotonic() - start

    print("Response:", response.content)
    print("Iterations:", response.iterations)
    print(f"Elapsed: {elapsed:.1f}s (~1s per tool turn means the 3 calls ran in parallel)")


if __name__ == "__main__":
    asyncio.run(main())
