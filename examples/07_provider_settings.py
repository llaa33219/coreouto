"""Example 07: Provider settings (normalized + pass-through).

`AgentConfig.provider_config` accepts 8 canonical keys that coreouto
translates to each provider's native kwargs. Use `provider_passthrough`
for non-canonical, provider-specific parameters.

Run with:
    export MOONSHOT_API_KEY=...
    python examples/07_provider_settings.py
"""

from __future__ import annotations

import asyncio
import os

import coreouto as co
from coreouto._types import AgentConfig
from coreouto.providers.openai import OpenAIProvider


async def main() -> None:
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        raise RuntimeError("MOONSHOT_API_KEY is required for this example")

    co.register_provider(
        "moonshot",
        OpenAIProvider(
            api_key=api_key,
            base_url="https://api.moonshot.ai/v1",
        ),
    )

    config = AgentConfig(
        name="settings-demo",
        model="kimi-k2.6",
        provider="moonshot",
        system_prompt="You are a helpful assistant. Answer concisely.",
        provider_config={
            "temperature": 0.3,  # lower randomness for more deterministic output
            "max_tokens": 1024,  # cap the response length (works across all providers)
            "top_p": 0.9,  # nucleus sampling threshold
        },
        provider_passthrough={
            # Non-canonical settings go here, e.g.:
            # "response_format": {"type": "json_object"},
        },
    )
    agent = co.Agent(config)
    response = await agent.call("Explain gravity in one sentence.")

    print("Response:", response.content)
    print("Iterations:", response.iterations)


if __name__ == "__main__":
    asyncio.run(main())
