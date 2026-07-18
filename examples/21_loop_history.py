"""Loop internals → history: carrying "what the agent did" across turns.

`Response.content` is only the final answer text. The work the loop did to
get there — tool calls, tool results, progress text — happens inside the
loop. Two ways to carry that work into the next turn's `history`:

  A. Raw transcript — `Response.messages` already contains the full loop
     transcript: assistant messages carrying `tool_calls`, plus `tool`-role
     result messages. Pass it back as `history` (dropping the leading
     system message, since `call()` prepends a fresh one) and the next
     turn sees exactly what the agent did.

  B. Curated work log via hooks — raw transcripts can be long (full tool
     payloads). Hooks let you record a compact log of loop internals
     (BEFORE/AFTER_TOOL_CALL, ON_ITERATION, ON_FINISH) between turns, then
     inject a synthesized summary message into the next `history`.

Run with:
    python examples/21_loop_history.py
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


def llm(content: str | None = None, tool_calls: list[ToolCall] | None = None) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
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
        else:
            parts.append(m.role)
    return " → ".join(parts)


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

    # ------------------------------------------------------------------
    # Pattern A: raw transcript — Response.messages goes back as history.
    # The transcript already includes the assistant's tool_calls and the
    # tool-role result messages, so no hooks are needed for the raw form.
    # ------------------------------------------------------------------
    print("=== Pattern A: raw transcript — Response.messages as history ===")
    provider_a = ScriptedMock(
        [
            llm(
                "Let me look that up.",
                [ToolCall(id="c1", name="search", arguments={"query": "population of Seoul"})],
            ),
            llm("Seoul has about 9.66 million people."),
            llm("Compared to Seoul's 9.66M (from my earlier search), Busan is smaller."),
        ]
    )
    co.register_provider("mock", provider_a)

    r1 = await agent.call("What is the population of Seoul?")
    print(f"turn 1 answer:     {r1.content}")
    print(f"turn 1 transcript: {describe(r1.messages)}")

    # call() always prepends a fresh system message, so drop the old one from
    # the transcript before reusing it — otherwise every turn adds a copy.
    transcript = [m for m in r1.messages if m.role != "system"]
    r2 = await agent.call("How does that compare to Busan?", history=transcript)
    print(f"turn 2 answer:     {r2.content}")
    seen = provider_a.calls[-1]
    print(f"turn 2 model saw:  {describe(seen)}")
    print(f"→ tool usage attached to history: {any(m.role == 'tool' for m in seen)}")

    # ------------------------------------------------------------------
    # Pattern B: hook-recorded work log → synthesized history message.
    # Hooks watch the loop from outside; between turns you fold the log
    # into one assistant message so the model "remembers" how it produced
    # its previous answer — without the full raw tool payloads.
    # ------------------------------------------------------------------
    print()
    print("=== Pattern B: hook-recorded work log → history message ===")

    entries: list[str] = []

    def on_tool_call(*, name, arguments, **_kwargs) -> None:
        args = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
        entries.append(f"called {name}({args})")

    def on_tool_result(*, name, result, **_kwargs) -> None:
        entries.append(f"{name} returned: {result.content}")

    def on_iteration(*, response, **_kwargs) -> None:
        # Progress text arrives as content alongside tool calls.
        if response.content and response.tool_calls:
            entries.append(f"progress: {response.content}")

    def on_finish(*, content, **_kwargs) -> None:
        entries.append(f"answered: {content}")

    # Hooks are global — register per conversation, clear_hooks() when done.
    co.register_hook(co.BEFORE_TOOL_CALL, on_tool_call)
    co.register_hook(co.AFTER_TOOL_CALL, on_tool_result)
    co.register_hook(co.ON_ITERATION, on_iteration)
    co.register_hook(co.ON_FINISH, on_finish)

    provider_b = ScriptedMock(
        [
            llm(
                "One moment.",
                [ToolCall(id="c2", name="search", arguments={"query": "population of Busan"})],
            ),
            llm("Busan has about 3.37 million people."),
            llm("Seoul is bigger — Busan has only 3.37M (from my earlier search)."),
        ]
    )
    co.register_provider("mock", provider_b)

    q1 = "What is the population of Busan?"
    b1 = await agent.call(q1)
    print(f"turn 1 answer: {b1.content}")
    print("recorded work log:")
    for e in entries:
        print(f"  - {e}")

    work_log = "\n".join(f"- {e}" for e in entries)
    history = [
        Message(role="user", content=q1),
        Message(role="assistant", content=f"What I did to answer this:\n{work_log}"),
    ]
    b2 = await agent.call("How does that compare to Seoul?", history=history)
    print(f"turn 2 answer: {b2.content}")
    seen_b = provider_b.calls[-1]
    print(f"turn 2 model saw: {describe(seen_b)}")
    print("injected work-log message:")
    for line in seen_b[2].content.splitlines():
        print(f"  {line}")

    co.clear_hooks()


if __name__ == "__main__":
    asyncio.run(main())
