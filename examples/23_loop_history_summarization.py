"""In-loop summarization → history: carrying a compacted transcript across turns.

Companion to 21_loop_history.py. There, `Response.messages` went straight
back as the next turn's `history`. But a long loop can accumulate more
than fits the context window — so compact the loop *while it runs*: once
cumulative tokens pass a threshold, summarize the older messages, wipe
them, and put the summary in their place. The loop then keeps accumulating
messages normally on top of the summary.

Because the hook mutates the same message list the agent is running on,
the compaction is automatically reflected in `Response.messages` — passing
it on as the next turn's `history` carries the summary (plus everything
piled up after it) instead of the raw pre-summary payloads.

`coreouto.contrib.hooks.auto_summarize_hook` is the ready-made seed for
this pattern. This example uses a small variant that resets its counter
after each compaction, so the post-summary iterations aren't re-compacted
on every iteration. The summarizer itself is a deterministic fake — a real
app would call an LLM here.

Run with:
    python examples/23_loop_history_summarization.py
"""

from __future__ import annotations

import asyncio

import coreouto as co
from coreouto._types import LLMResponse, Message, ToolCall, Usage


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
    content: str | None = None, tool_calls: list[ToolCall] | None = None, tokens: int = 1
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=Usage(prompt_tokens=tokens, completion_tokens=1, total_tokens=tokens + 1),
    )


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


def summarize_loop(messages: list[Message]) -> list[Message]:
    """Compress everything between the system prompt and the latest
    assistant message into one summary message.

    The trailing message is kept verbatim: it is the assistant turn that
    triggered this iteration, and its tool calls are still awaiting their
    results — dropping it would orphan the tool-result messages the loop
    is about to append.
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
    summary = Message(role="user", content="[summary of earlier loop work]\n" + "\n".join(lines))
    return [head, summary, tail]


def make_loop_compactor(*, threshold: int, summarize_fn):
    """ON_ITERATION hook: replace messages with summarize_fn(messages) once
    cumulative tokens pass `threshold`, then reset the counter.

    Returns (hook, compactions) — compactions records the iteration numbers
    where compaction happened, for inspection after the run.
    """
    total = 0
    compactions: list[int] = []

    def hook(*, iteration, messages, response, **_kwargs) -> None:
        nonlocal total
        if response.usage is None:
            return
        total += response.usage.total_tokens
        if total >= threshold:
            messages[:] = summarize_fn(messages)
            compactions.append(iteration)
            total = 0

    return hook, compactions


FAKE_DB = {
    "population of Seoul": "Seoul: 9.66 million (2024 census)",
    "population of Busan": "Busan: 3.37 million (2024 census)",
    "population of Daegu": "Daegu: 2.36 million (2024 census)",
}


@co.register_tool("search")
def search(query: str) -> str:
    """Search a (fake) knowledge base for `query`."""
    return FAKE_DB.get(query, "no results")


async def main() -> None:
    agent = co.Agent(co.AgentConfig(name="chat", model="mock-model", provider="mock"))

    # Cumulative tokens per iteration: 61 → 122 (compacts at threshold=100)
    # → counter resets → 31 → 21, so the post-summary loop runs uncompacted.
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
            llm(
                "Checking Daegu.",
                [ToolCall(id="c3", name="search", arguments={"query": "population of Daegu"})],
                tokens=30,
            ),
            llm("Seoul is the largest at 9.66M, then Busan 3.37M, then Daegu 2.36M.", tokens=20),
            llm(
                "Seoul — per the summary, it was checked first and returned 9.66 million.",
                tokens=20,
            ),
        ]
    )
    co.register_provider("mock", provider)

    compactor, compactions = make_loop_compactor(threshold=100, summarize_fn=summarize_loop)
    co.register_hook(co.ON_ITERATION, compactor)

    q1 = "Rank Seoul, Busan, and Daegu by population."
    r1 = await agent.call(q1)
    print(f"turn 1 answer:     {r1.content}")
    print(f"compacted at loop iterations: {compactions}")
    print(f"turn 1 transcript: {describe(r1.messages)}")

    # Response.messages IS the compacted loop state — the hook mutated the
    # list in place. Drop the leading system message (call() prepends a
    # fresh one) and pass the rest as history.
    transcript = [m for m in r1.messages if m.role != "system"]
    r2 = await agent.call(
        "Which city did you check first, and what did it return?", history=transcript
    )
    print()
    print(f"turn 2 answer:     {r2.content}")
    seen = provider.calls[-1]
    print(f"turn 2 model saw:  {describe(seen)}")
    summary_msg = next(m for m in seen if m.content and m.content.startswith("[summary"))
    print("summary the model received instead of the raw earlier loop:")
    for line in summary_msg.content.splitlines():
        print(f"  {line}")

    co.clear_hooks()


if __name__ == "__main__":
    asyncio.run(main())
