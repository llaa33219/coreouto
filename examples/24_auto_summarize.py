"""In-loop summarization with the contrib `auto_summarize_hook`.

A long tool loop can accumulate more messages than fit the context window.
`coreouto.contrib.hooks.auto_summarize_hook` compacts the loop *while it
runs*: registered on ON_ITERATION, it accumulates each response's token
usage, and once the running total passes your threshold it calls your
`summarize_fn(messages)` and swaps the loop's message list for the
returned (shorter) one — in place. The loop then continues on top of the
summary.

You own the policy: the threshold, and how the summary is produced. In a
real app `summarize_fn` would call an LLM; here it's a deterministic fake
so the example runs offline.

Two recipe details worth knowing:
  - The summarizer receives the list *including* the assistant message
    from the current iteration. If that message carries tool calls, keep
    it — its tool results are appended right after the hook fires, so
    dropping it would orphan them.
  - The hook keeps a running total, so once past the threshold it
    re-compacts on every later iteration that reports usage. For a
    resetting variant (and for carrying the compacted transcript into the
    next call's history) see 23_loop_history_summarization.py.

Run with:
    python examples/24_auto_summarize.py
"""

from __future__ import annotations

import asyncio

import coreouto as co
from coreouto._types import LLMResponse, Message, ToolCall, Usage
from coreouto.contrib.hooks import auto_summarize_hook


class ScriptedMock:
    """Provider that yields queued LLMResponses and records every create() call."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    async def create(self, messages, *, model, tools=None, system_prompt=None, **kwargs):
        self.calls.append(list(messages))
        return self._responses.pop(0)

    def format_assistant_message(self, response):
        return Message(
            role="assistant",
            content=response.content or "",
            tool_calls=response.tool_calls if response.tool_calls else None,
        )

    def format_tool_result(self, tool_call, result):
        return Message(
            role="tool",
            content=result.content,
            tool_call_id=tool_call.id,
            name=tool_call.name,
        )


def llm(
    content: str | None = None,
    tool_calls: list[ToolCall] | None = None,
    tokens: int | None = 1,
) -> LLMResponse:
    usage = (
        None
        if tokens is None
        else Usage(prompt_tokens=tokens, completion_tokens=1, total_tokens=tokens + 1)
    )
    return LLMResponse(content=content, tool_calls=tool_calls or [], usage=usage)


def describe(messages: list[Message]) -> str:
    """One-line role sketch of a message list, naming tool calls/results."""
    parts = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            names = ", ".join(tc.name for tc in m.tool_calls)
            parts.append(f"assistant(tool_calls=[{names}])")
        elif m.role == "tool":
            parts.append(f"tool({m.name})")
        elif m.content and m.content.startswith("[summary"):
            parts.append("SUMMARY")
        else:
            parts.append(m.role)
    return " → ".join(parts)


def fake_summarizer(messages: list[Message]) -> list[Message]:
    """Stand-in for an LLM summarizer.

    Keeps the leading system message and the trailing assistant message
    (its tool calls are still awaiting their results), and compresses
    everything between into one summary message.
    """
    head, *middle, tail = messages
    lines = []
    for m in middle:
        if m.role == "tool":
            lines.append(f"- {m.name} returned: {m.content}")
        elif m.role == "assistant":
            if m.content:
                lines.append(f"- progress: {m.content}")
            if m.tool_calls:
                names = ", ".join(tc.name for tc in m.tool_calls)
                lines.append(f"- called {names}")
        else:
            lines.append(f"- user asked: {m.content}")
    print(f"[auto_summarize] compacting {len(messages)} messages -> 3:")
    for line in lines:
        print(f"  {line}")
    summary = Message(role="user", content="[summary of earlier loop work]\n" + "\n".join(lines))
    return [head, summary, tail]


FAKE_DB = {
    "population of Seoul": "Seoul: 9.66 million (2024 census)",
    "population of Busan": "Busan: 3.37 million (2024 census)",
}


@co.register_tool("search")
def search(query: str) -> str:
    """Search a (fake) knowledge base for `query`."""
    return FAKE_DB.get(query, "no results")


async def main() -> None:
    agent = co.Agent(co.AgentConfig(name="chat", model="mock-model", provider="mock"))

    # Cumulative tokens: 61 after iteration 1, 122 after iteration 2 —
    # past the threshold of 100, so the hook compacts there. The final
    # response reports no usage (some providers omit it on some responses);
    # the hook skips usage-less responses instead of re-compacting.
    provider = ScriptedMock(
        [
            llm(
                "Checking Seoul.",
                [ToolCall(id="c1", name="search", arguments={"query": "population of Seoul"})],
                tokens=60,
            ),
            llm(
                "Checking Busan.",
                [ToolCall(id="c2", name="search", arguments={"query": "population of Busan"})],
                tokens=60,
            ),
            llm("Seoul has 9.66M people, Busan 3.37M.", tokens=None),
        ]
    )
    co.register_provider("mock", provider)

    co.register_hook(
        co.ON_ITERATION, auto_summarize_hook(threshold=100, summarize_fn=fake_summarizer)
    )

    r = await agent.call("What are the populations of Seoul and Busan?")
    print()
    print(f"answer: {r.content}")
    print()
    print("what the model saw at each iteration:")
    for i, call_messages in enumerate(provider.calls, 1):
        print(f"  iteration {i}: {describe(call_messages)}")
    print()
    print(f"final transcript: {describe(r.messages)}")

    co.clear_hooks()


if __name__ == "__main__":
    asyncio.run(main())
