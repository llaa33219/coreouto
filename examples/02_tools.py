"""Example 02: Tool registration.

Shows how to expose Python callables as LLM-callable tools using the
@register_tool decorator. The agent receives both tools in its config
and decides which to call based on the user message.

The tool descriptions are taken from each function's docstring, and the
parameter JSON Schema is derived from the function's type hints.

Run with:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/02_tools.py
"""

from __future__ import annotations

import asyncio
import os

import coreouto as co
from coreouto._types import AgentConfig


@co.register_tool("get_weather")
def get_weather(city: str) -> str:
    """Return a short, fabricated weather report for `city`."""
    return f"It is 22C and sunny in {city}."


@co.register_tool("search_web")
def search_web(query: str) -> str:
    """Return fabricated web-search results for `query`."""
    return f"<top results for {query}>"


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for this example")

    co.providers.anthropic.register(api_key=api_key)

    config = AgentConfig(
        name="weather-news",
        model="claude-opus-4-8",
        provider="anthropic",
        system_prompt=(
            "You are a research assistant. Use the available tools to gather "
            "information, then wrap your final answer in <finish>...</finish> tags."
        ),
        tools=["get_weather", "search_web"],
    )
    agent = co.Agent(config)
    response = await agent.call("What's the weather in Tokyo and find recent news about it?")

    print("Response:", response.content)
    print("Iterations:", response.iterations)


if __name__ == "__main__":
    asyncio.run(main())
