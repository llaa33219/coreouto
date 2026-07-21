"""Summarizing conversation history between turns.

Whatever you attach as `history` must fit the model's context window, so a
long-running conversation can't grow forever. The standard pattern: once
the accumulated history passes a budget, fold the older turns into a
single summary message and keep only the most recent turns verbatim — the
next `call()` sees `[summary] + [recent tail]`.

coreouto doesn't do this for you (no built-in auto-summarization — see
docs/philosophy.md). You own the policy: when to compact, how much to keep
verbatim, and how to summarize. In a real app `summarize_fn` would be an
LLM call; here it's a deterministic fake so the example runs offline.

Since compaction replaces your history store, an earlier summary is folded
into the next one when the budget is exceeded again — summaries stay
bounded no matter how long the conversation runs.

Run with:
    python examples/22_history_summarization.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import coreouto as co
from coreouto._types import LLMResponse, Message, Usage


class RecordingMock:
    """Echo provider that records the messages of every create() call."""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    async def create(self, messages, *, model, tools=None, system_prompt=None, **kwargs):
        self.calls.append(list(messages))
        return LLMResponse(
            content=f"echo: {messages[-1].content}",
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
            content=result.content,
            tool_call_id=tool_call.id,
            name=tool_call.name,
        )


def fake_summarizer(messages: list[Message]) -> str:
    """Stand-in for an LLM summarizer: one compressed line per dropped message."""
    return "\n".join(
        f"- {m.role} said: {' '.join((m.content or '').splitlines())}" for m in messages
    )


def compact_history(
    history: list[Message],
    *,
    budget: int,
    keep_recent: int,
    summarize_fn: Callable[[list[Message]], str],
) -> list[Message] | None:
    """Compact `history` when it exceeds `budget` messages.

    Everything except the last `keep_recent` messages is replaced by a
    single summary message. Returns the compacted list, or None when the
    history is still within budget (so the caller can log the event).
    """
    if len(history) <= budget:
        return None
    old, recent = history[:-keep_recent], history[-keep_recent:]
    summary = Message(
        role="user",
        content=f"[summary of earlier conversation]\n{summarize_fn(old)}",
    )
    return [summary, *recent]


BUDGET = 6  # messages; real apps would count tokens
KEEP_RECENT = 2  # messages to keep verbatim once compacting


async def main() -> None:
    provider = RecordingMock()
    co.register_provider("mock", provider)
    agent = co.Agent(co.AgentConfig(name="chat", model="mock-model", provider="mock"))

    history: list[Message] = []

    turns = [
        "My name is Alice.",
        "I like kimchi.",
        "I live in Seoul.",
        "I work as a designer.",
        "My favorite color is green.",
        "I have a cat named Nabi.",
        "What do you remember about me?",
    ]

    for i, turn in enumerate(turns, 1):
        compacted = compact_history(
            history, budget=BUDGET, keep_recent=KEEP_RECENT, summarize_fn=fake_summarizer
        )
        if compacted is not None:
            print(
                f"--- before turn {i}: history over budget ({len(history)} > {BUDGET}), compacting ---"
            )
            print(compacted[0].content)
            print()
            history = compacted

        r = await agent.call(turn, history=history)
        history = [
            *history,
            Message(role="user", content=turn),
            Message(role="assistant", content=r.content or ""),
        ]
        print(f"turn {i}: {r.content!r}  (history now {len(history)} messages)")

    # The last call saw the compacted history: one summary message plus the
    # recent tail — not the full raw conversation.
    print()
    print("=== what the model saw on the final turn ===")
    for m in provider.calls[-1]:
        first_line = (m.content or "").splitlines()[0] if m.content else ""
        print(f"  {m.role:9s} {first_line}")


if __name__ == "__main__":
    asyncio.run(main())
