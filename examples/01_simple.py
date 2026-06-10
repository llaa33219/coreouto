"""Example 01: Minimum viable call.

Demonstrates the smallest possible flow with coreouto:
  1. Register an OpenAI provider using an API key from the environment.
  2. Build an AgentConfig directly (no preset).
  3. Call the agent with asyncio.run and print the response.

Run with:
    export OPENAI_API_KEY=sk-...
    python examples/01_simple.py
"""

from __future__ import annotations

import asyncio
import os

import coreouto as co
from coreouto._types import AgentConfig


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for this example")

    co.providers.openai.register(api_key=api_key)

    config = AgentConfig(
        name="simple",
        model="gpt-5.5",
        provider="openai",
        system_prompt="You are a helpful assistant. Answer concisely.",
    )
    agent = co.Agent(config)
    response = await agent.call("Reply with a single word: 'pong'.")

    print("Response:", response.content)
    print("Iterations:", response.iterations)


if __name__ == "__main__":
    asyncio.run(main())
