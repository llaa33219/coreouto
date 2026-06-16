"""Injecting user messages into a running agent loop.

coreouto exposes the loop internals via hooks but doesn't manage
conversation state. For "while the loop is running" use cases, the
`Agent.inject_user_message()` method queues a user message that the loop
drains at the start of the next iteration.

This example shows two patterns:
  1. Inject from a hook callback (intercepting a tool result, etc.)
  2. Inject from a concurrent async task (interrupting a long-running agent)

Use cases:
  - Human-in-the-loop: a human can review/correct mid-execution
  - Streaming input: a websocket can push user messages into a running agent
  - Tool-triggered re-prompting: a tool's result can re-prompt the agent
"""

import asyncio

import coreouto as co
from coreouto._types import LLMResponse, Message, ToolCall, Usage


class MockOpenAI:
    async def create(self, messages, *, model, tools=None, system_prompt=None, **kwargs):
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id="finish_1",
                    name="finish",
                    arguments={"content": f"echo: {last_user.content if last_user else '?'}"},
                ),
            ],
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


async def pattern_inject_from_hook():
    co.clear_providers()
    co.clear_hooks()
    co.register_provider("mock", MockOpenAI())

    def interrupt_on_finish(*, content, tool_call_id, **_):
        if "bad" in (content or "").lower():
            agent.inject_user_message("Please revise to be more positive.")

    co.register_hook("on_finish", interrupt_on_finish)

    agent = co.Agent(co.AgentConfig(name="t", model="m", provider="mock"))

    r1 = await agent.call("Write something bad.")
    print(f"  result 1: {r1.content!r}")
    r2 = await agent.call("Now write something nice.")
    print(f"  result 2: {r2.content!r}")


async def pattern_inject_from_concurrent_task():
    co.clear_providers()
    co.clear_hooks()
    co.register_provider("mock", MockOpenAI())

    agent = co.Agent(co.AgentConfig(name="t", model="m", provider="mock"))

    async def delayed_inject():
        await asyncio.sleep(0.01)
        agent.inject_user_message("injected from concurrent task")

    task = asyncio.create_task(delayed_inject())
    _ = task
    response = await agent.call("initial message")
    print(f"  response: {response.content!r}")


async def main():
    print("=== Pattern 1: inject from a hook callback ===")
    await pattern_inject_from_hook()
    print()
    print("=== Pattern 2: inject from a concurrent task ===")
    await pattern_inject_from_concurrent_task()


if __name__ == "__main__":
    asyncio.run(main())
