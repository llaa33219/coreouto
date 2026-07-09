"""Example 16: Streaming output with hooks.

Enable ``stream=True`` on the provider to receive text and thinking deltas
in real time via hooks — the final response is identical to the non-streaming
path, but you also get per-token visibility as the model generates.

This example registers two hooks:

* ``on_stream_text`` — prints each text fragment as it arrives (typing effect).
* ``on_stream_thinking`` — prints reasoning fragments to stderr so they don't
  mix with the answer on stdout.

It also fires ``on_thinking`` after each LLM call if the model produced
reasoning content, showing the full thinking text in one piece.

Run with:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/16_streaming_output.py
"""

from __future__ import annotations

import asyncio
import os
import sys

import coreouto as co
from coreouto._types import AgentConfig


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for this example")

    co.providers.anthropic.register(api_key=api_key, stream=True)

    def on_text(*, text: str, **_kwargs: object) -> None:
        print(text, end="", flush=True)

    def on_thinking_stream(*, text: str, **_kwargs: object) -> None:
        print(text, end="", file=sys.stderr, flush=True)

    def on_thinking_full(*, thinking: str, **_kwargs: object) -> None:
        print(f"\n[full thinking] {thinking}\n", file=sys.stderr)

    co.register_hook(co.ON_STREAM_TEXT, on_text)
    co.register_hook(co.ON_STREAM_THINKING, on_thinking_stream)
    co.register_hook(co.ON_THINKING, on_thinking_full)

    agent = co.Agent(
        AgentConfig(
            name="streamer",
            model="claude-sonnet-4-6",
            provider="anthropic",
            system_prompt="Explain concepts clearly. Respond with text.",
            provider_config={
                "reasoning_effort": "medium",
                "max_tokens": 4000,
            },
        )
    )

    response = await agent.call("Explain how public-key cryptography works in two paragraphs.")
    print(f"\n\n[final] {response.content}")
    print(f"[iterations] {response.iterations}")


if __name__ == "__main__":
    asyncio.run(main())
