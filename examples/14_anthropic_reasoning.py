"""Example 14: Anthropic reasoning effort (adaptive thinking).

Set `provider_config={"reasoning_effort": ...}` to control how hard the
Anthropic model thinks per turn. coreouto emits:

  thinking={"type": "adaptive"},
  output_config={"effort": <level>},

so the model decides *when* to think and you decide *how much*.

Accepted values:
  none   - drop thinking entirely (cheapest)
  low    - fewest thinking tokens (routing, classification)
  medium - balanced (default for most chat/agentic flows)
  high   - thorough reasoning (default if you don't set effort)
  xhigh  - long-horizon coding / research (Opus 4.7+ / Sonnet 4.6+)
  max    - absolute maximum capability (Opus 4.6+ / Fable 5 / Mythos 5)

Run with:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/14_anthropic_reasoning.py
"""

from __future__ import annotations

import asyncio
import os

import coreouto as co
from coreouto._types import AgentConfig


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for this example")

    co.providers.anthropic.register(api_key=api_key)

    deep = co.Agent(
        AgentConfig(
            name="deep-thinker",
            model="claude-sonnet-4-6",
            provider="anthropic",
            system_prompt=(
                "You are a careful research assistant. When you have your "
                "final answer, respond with text (no tool call)."
            ),
            provider_config={
                "reasoning_effort": "high",
                "max_tokens": 8000,
            },
        )
    )

    cheap = co.Agent(
        AgentConfig(
            name="fast-router",
            model="claude-haiku-4-5",
            provider="anthropic",
            system_prompt=(
                "Classify the user's request in one of: BILLING, ACCOUNT, "
                "TECHNICAL, OTHER. Respond with your answer (no tool call)."
            ),
            provider_config={
                "reasoning_effort": "low",
                "max_tokens": 1024,
            },
        )
    )

    deep_response = await deep.call("Explain how RSA key exchange works in two paragraphs.")
    print("Deep:", deep_response.content)
    print("Deep iterations:", deep_response.iterations)

    cheap_response = await cheap.call("I can't log in to my account.")
    print("Cheap:", cheap_response.content)
    print("Cheap iterations:", cheap_response.iterations)


if __name__ == "__main__":
    asyncio.run(main())
