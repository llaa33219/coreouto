"""Example 05: Multi-agent orchestration.

`agent_as_tool(preset_name)` wraps a registered preset as a callable
Tool that, when invoked, runs the sub-agent and returns its content.
The returned Tool is not auto-registered; we expose it under a stable
name in the global registry so the parent agent can reference it by
string in its `tools` list.

The hierarchy here is:
  coordinator
    └── delegate_research  (wraps `researcher` sub-agent)

Run with:
    export MINIMAX_API_KEY=...
    python examples/05_multi_agent.py
"""

from __future__ import annotations

import asyncio
import os

import coreouto as co
from coreouto.providers.openai import OpenAIProvider


@co.register_tool("get_weather")
def get_weather(city: str) -> str:
    """Return a short, fabricated weather report for `city`."""
    return f"It is 22C and sunny in {city}."


@co.register_tool("search_web")
def search_web(query: str) -> str:
    """Return fabricated web-search results for `query`."""
    return f"<top results for {query}>"


async def main() -> None:
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY is required for this example")

    co.register_provider(
        "minimax",
        OpenAIProvider(
            api_key=api_key,
            base_url="https://api.minimax.io/v1",
        ),
    )

    co.register_agent_preset(
        "researcher",
        model="MiniMax-M3",
        provider="minimax",
        system_prompt="You gather facts. Use the available tools.",
        tools=["get_weather", "search_web"],
    )
    co.register_agent_preset(
        "writer",
        model="MiniMax-M3",
        provider="minimax",
        system_prompt="You turn raw notes into a clean, short summary.",
        tools=[],
    )

    delegate = co.agent_as_tool("researcher")
    co.register_tool("delegate_research", description=delegate.description)(delegate.handler)

    co.register_agent_preset(
        "coordinator",
        model="MiniMax-M3",
        provider="minimax",
        system_prompt=(
            "You coordinate work between agents. Use `delegate_research` to "
            "gather facts, then respond with your final answer (no tool call)."
        ),
        tools=["delegate_research"],
    )

    response = await co.Agent(co.get_agent_preset("coordinator").to_config()).call(
        "Find today's weather in Berlin and any recent news, then summarise it."
    )
    print("Coordinator:", response.content)


if __name__ == "__main__":
    asyncio.run(main())
