"""Example 08: Missing-<finish> reminder.

When the model returns text without <finish>...</finish> tags, the agent
injects a user message reminding the model to wrap its final answer in
tags, then continues the loop.

This example uses a mock provider to demonstrate the two-turn flow:
  1. The model returns plain text (no <finish> tag).
  2. The agent appends a reminder and calls the LLM again.
  3. The model returns text wrapped in <finish>...</finish> tags.
  4. The agent terminates and returns the extracted answer.

Run with:
    python examples/08_force_finish.py
"""

from __future__ import annotations

import asyncio

import coreouto as co
from coreouto._types import AgentConfig, LLMResponse, Message, Usage


class MockProvider:
    """Provider that first returns plain text, then returns a <finish> tag."""

    def __init__(self):
        self._turn = 0

    async def create(self, messages, *, model, tools=None, system_prompt=None, **kwargs):
        self._turn += 1
        if self._turn == 1:
            return LLMResponse(
                content="Hello there!",
                tool_calls=[],
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        return LLMResponse(
            content="<finish>Hello!</finish>",
            tool_calls=[],
            usage=Usage(prompt_tokens=2, completion_tokens=2, total_tokens=4),
        )

    def format_assistant_message(self, response):
        return Message(
            role="assistant",
            content=response.content or "",
            tool_calls=response.tool_calls if response.tool_calls else None,
        )

    def format_tool_result(self, tool_call, result):
        return Message(
            role="tool",
            content=str(result.content) if hasattr(result, "content") else str(result),
            tool_call_id=tool_call.id if hasattr(tool_call, "id") else tool_call["id"],
            name=tool_call.name if hasattr(tool_call, "name") else tool_call["name"],
        )


async def main() -> None:
    co.register_provider("mock", MockProvider())

    config = AgentConfig(
        name="safe-agent",
        model="mock",
        provider="mock",
    )
    agent = co.Agent(config)
    response = await agent.call("Say hello in one word.")

    print("Response:", response.content)
    print("Iterations:", response.iterations)


if __name__ == "__main__":
    asyncio.run(main())
