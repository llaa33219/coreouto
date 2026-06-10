"""Example 10: Custom endpoints and proxy routing.

All built-in providers support custom endpoints via constructor arguments.
This example shows how to route to proxies, self-hosted models, or
region-specific endpoints.

Run with:
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    export GOOGLE_API_KEY=...
    export MINIMAX_API_KEY=...
    export ZHIPU_API_KEY=...
    export MOONSHOT_API_KEY=...
    python examples/10_custom_endpoints.py
"""

from __future__ import annotations

import asyncio
import os

import coreouto as co
from coreouto.providers.anthropic import AnthropicProvider
from coreouto.providers.google import GoogleProvider
from coreouto.providers.openai import OpenAIProvider


async def main() -> None:
    co.register_provider(
        "local",
        OpenAIProvider(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
        ),
    )

    co.register_provider(
        "openai-us",
        OpenAIProvider(
            api_key=os.environ.get("OPENAI_API_KEY", "test"),
            base_url="https://api.openai.com/v1",
        ),
    )

    co.register_provider(
        "anthropic-proxy",
        AnthropicProvider(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "test"),
            base_url="https://proxy.example.com",
        ),
    )

    co.register_provider(
        "google-proxy",
        GoogleProvider(
            api_key=os.environ.get("GOOGLE_API_KEY", "test"),
            client_options={"api_endpoint": "https://proxy.example.com"},
        ),
    )

    co.register_provider(
        "minimax",
        OpenAIProvider(
            api_key=os.environ.get("MINIMAX_API_KEY", "test"),
            base_url="https://api.minimax.io/v1",
        ),
    )

    co.register_provider(
        "zhipu",
        OpenAIProvider(
            api_key=os.environ.get("ZHIPU_API_KEY", "test"),
            base_url="https://open.bigmodel.cn/api/paas/v4",
        ),
    )

    co.register_provider(
        "moonshot",
        OpenAIProvider(
            api_key=os.environ.get("MOONSHOT_API_KEY", "test"),
            base_url="https://api.moonshot.ai/v1",
        ),
    )

    print("Registered providers:", co.available_providers())


if __name__ == "__main__":
    asyncio.run(main())
