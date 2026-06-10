"""Example 03: Agent presets.

Presets are named bundles of AgentConfig. Register them once, then
materialise an AgentConfig via `preset.to_config()` whenever you need
an agent. The `tools` field references names registered via
@register_tool, so make sure those are declared before the preset.

Run with:
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    export GOOGLE_API_KEY=...
    python examples/03_presets.py
"""

from __future__ import annotations

import asyncio
import os

import coreouto as co


@co.register_tool("get_weather")
def get_weather(city: str) -> str:
    """Return a short, fabricated weather report for `city`."""
    return f"It is 22C and sunny in {city}."


@co.register_tool("search_web")
def search_web(query: str) -> str:
    """Return fabricated web-search results for `query`."""
    return f"<top results for {query}>"


async def main() -> None:
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")
    if not openai_key or not anthropic_key or not google_key:
        raise RuntimeError("OPENAI_API_KEY, ANTHROPIC_API_KEY and GOOGLE_API_KEY are required")

    co.providers.openai.register(api_key=openai_key)
    co.providers.anthropic.register(api_key=anthropic_key)
    co.providers.google.register(api_key=google_key)

    co.register_agent_preset(
        "researcher",
        model="gpt-5.5",
        provider="openai",
        system_prompt="You gather and verify facts using the available tools.",
        tools=["get_weather", "search_web"],
    )
    co.register_agent_preset(
        "summarizer",
        model="claude-sonnet-4-6",
        provider="anthropic",
        system_prompt="You compress the researcher's findings into 3 sentences.",
        tools=[],
    )
    co.register_agent_preset(
        "fact_checker",
        model="gemini-2.5-pro",
        provider="google",
        system_prompt="You verify facts for accuracy.",
        tools=[],
    )

    researcher_config = co.get_agent_preset("researcher").to_config()
    response = await co.Agent(researcher_config).call(
        "Find the weather in Paris and any recent news about it."
    )
    print("Researcher:", response.content)


if __name__ == "__main__":
    asyncio.run(main())
