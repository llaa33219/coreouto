"""Example 06: Custom provider.

Implements the 3-method Provider protocol with a tiny in-process
"echo" provider. It always responds with the last user message prefixed
by "echo: " via a no-tool-call response with content 'echo: ...'. No API
key, no network, no SDK — useful for smoke tests, local development, and
demonstrating how a new provider integrates with the loop.

The protocol is duck-typed (Protocol with @runtime_checkable), so the
class does not need to inherit from anything; it just needs the three
methods with the right signatures.

Run with:
    python examples/06_custom_provider.py
"""

from __future__ import annotations

import asyncio

import coreouto as co
from coreouto._types import (
    AgentConfig,
    LLMResponse,
    Message,
    Usage,
)


class MyEchoProvider:
    async def create(
        self,
        messages: list[Message],
        *,
        model: str,
        tools=None,
        system_prompt=None,
        **kwargs,
    ) -> LLMResponse:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        return LLMResponse(
            content=f"echo: {last_user}",
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    def format_assistant_message(self, response: LLMResponse) -> Message:
        return Message(
            role="assistant",
            content=response.content or "",
            tool_calls=list(response.tool_calls) or None,
        )

    def format_tool_result(self, tool_call, result) -> Message:
        return Message(
            role="tool",
            tool_call_id=tool_call.id if hasattr(tool_call, "id") else tool_call["id"],
            content=result.content,
            name=tool_call.name if hasattr(tool_call, "name") else tool_call["name"],
        )


async def main() -> None:
    co.register_provider("echo", MyEchoProvider())

    config = AgentConfig(
        name="echo-agent",
        model="echo-1",
        provider="echo",
        system_prompt="You are an echo.",
    )
    response = await co.Agent(config).call("hello world")
    print("Echo response:", response.content)


if __name__ == "__main__":
    asyncio.run(main())
