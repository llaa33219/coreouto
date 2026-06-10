"""Conversation history: accumulating prior turns and fabricating messages.

coreouto doesn't auto-manage conversation state — that's an application
concern. Instead, `Agent.call()` and `Agent.call_sync()` accept a `history`
parameter (a list of `Message` objects) which is prepended to the initial
message list.

This example shows two patterns:
  1. Accumulating: pass `previous_response.messages` to continue a conversation.
  2. Fabricating: hand-craft a list of `Message` objects to seed the agent
     with any context you want.

Run with:
    python examples/12_history.py
"""

import asyncio

import coreouto as co
from coreouto._types import LLMResponse, Message, Usage


class MockOpenAI:
    async def create(self, messages, *, model, tools=None, system_prompt=None, **kwargs):
        return LLMResponse(
            content=f"<finish>echo: {messages[-1].content}</finish>",
            tool_calls=[],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
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


async def main():
    co.register_provider("mock", MockOpenAI())

    agent = co.Agent(
        co.AgentConfig(
            name="chat-agent",
            model="mock-model",
            provider="mock",
            system_prompt="You are a helpful assistant.",
        )
    )

    print("=== Pattern 1: accumulating a conversation ===")
    r1 = await agent.call("My name is Alice.")
    print(f"  turn 1: {r1.content!r}")

    r2 = await agent.call("What is my name?", history=r1.messages)
    print(f"  turn 2: {r2.content!r}")
    print(
        f"  (the model sees {len(r2.messages)} messages: "
        f"1 system + {len(r1.messages) - 1} from r1 + 1 new user)"
    )

    print()
    print("=== Pattern 2: fabricating history ===")
    fake = [
        Message(role="user", content="What is 2 + 2?"),
        Message(role="assistant", content="4"),
        Message(role="user", content="And 3 + 3?"),
        Message(role="assistant", content="6"),
    ]
    r3 = await agent.call("Continue the pattern.", history=fake)
    print(f"  response: {r3.content!r}")

    print()
    print("=== Pattern 3: empty list == no history ===")
    r4 = await agent.call("Hello.", history=[])
    print(f"  response: {r4.content!r}")


if __name__ == "__main__":
    asyncio.run(main())
