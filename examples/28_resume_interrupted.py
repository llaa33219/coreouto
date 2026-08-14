"""Resuming an interrupted loop from the last answered tool call.

A crashed or cancelled `call()` leaves the transcript in one of two
states:

  A. It ends with a tool result (interrupted mid-API-call). Every tool
     call is answered, so you can resume directly:
     `call("...", history=transcript)`.

  B. It ends with an assistant message whose tool calls have no results
     (interrupted mid-tool-execution). Providers reject dangling tool
     calls, so cut that trailing turn and resume from the last tool
     result instead.

Note: coreouto appends a turn's tool results only after ALL of its tool
calls finish, so a cancellation lands either before or after the whole
batch — case B's tail is always a bare assistant message, never a
partially-answered one. (An alternative to cutting case B is injecting
error tool results for the unanswered calls — what the "tool_result"
error reaction in examples/17-20 does.)

Both cases are demonstrated live against mock providers: scenario A
crashes the second API call (the transcript ends with a tool result),
scenario B cancels the call task while a tool is running (the transcript
ends with an unanswered tool call).

Both resumes use `call(history=...)` with no user message: when
`user_message` is omitted the transcript is sent exactly as declared, so
a history ending in a tool result (or a user message) simply continues.

Run: python examples/28_resume_interrupted.py
"""

import asyncio
import contextlib

import coreouto as co
from coreouto._types import LLMResponse, Message, ToolCall, Usage


def cut_to_last_answered_turn(messages: list[Message]) -> list[Message]:
    """Drop system messages and a trailing turn with unanswered tool calls.

    Only the last turn can be partial: if the final assistant message's
    tool calls lack matching tool results, that turn is removed and the
    history ends at the last tool result (or earlier user message).
    """
    history = [m for m in messages if m.role != "system"]
    for i in range(len(history) - 1, -1, -1):
        msg = history[i]
        if msg.role == "assistant" and msg.tool_calls:
            answered = {m.tool_call_id for m in history[i + 1 :] if m.role == "tool"}
            if {tc.id for tc in msg.tool_calls} - answered:
                del history[i:]
            break
    return history


class MockBase:
    def format_assistant_message(self, response):
        return Message(
            role="assistant",
            content=response.content or "",
            tool_calls=list(response.tool_calls) or None,
        )

    def format_tool_result(self, tool_call, result):
        return Message(
            role="tool",
            content=result.content,
            tool_call_id=tool_call.id,
            name=tool_call.name,
        )


class CrashAfterToolMock(MockBase):
    """Call 1: a tool call. Call 2: crash (as if the connection died after
    the tool result was appended). Call 3 (the resume): final answer."""

    def __init__(self) -> None:
        self.calls = 0

    async def create(self, messages, *, model, tools=None, system_prompt=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                tool_calls=[ToolCall(id="t1", name="lookup", arguments={"query": "coreouto"})],
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        if self.calls == 2:
            raise ConnectionError("connection reset mid-request")
        return LLMResponse(
            content=f"resumed and finished (API call #{self.calls})",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


class ToolCallThenAnswerMock(MockBase):
    """Call 1: a tool call (the tool itself hangs; the caller cancels).
    Call 2 (the resume): final answer."""

    def __init__(self) -> None:
        self.calls = 0

    async def create(self, messages, *, model, tools=None, system_prompt=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                tool_calls=[ToolCall(id="s1", name="slow_lookup", arguments={"query": "x"})],
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        return LLMResponse(
            content=f"finished without the slow tool (API call #{self.calls})",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


def capture_live_messages() -> dict[str, list[Message]]:
    live: dict[str, list[Message]] = {}

    def capture(*, messages: list[Message], **_kwargs) -> None:
        live["messages"] = messages

    co.register_hook(co.BEFORE_LLM_CALL, capture)
    return live


async def scenario_a_resume_directly() -> None:
    print("=== A: interrupted mid-API-call (tail is a tool result) ===")
    co.clear_providers()
    co.clear_hooks()
    co.clear_tools()

    @co.register_tool("lookup")
    def lookup(query: str) -> str:
        """Return fabricated lookup results for `query`."""
        return f"results for {query}"

    co.register_provider("mock", CrashAfterToolMock())
    live = capture_live_messages()
    agent = co.Agent(co.AgentConfig(name="t", model="m", provider="mock", tools=["lookup"]))

    try:
        await agent.call("look something up")
    except ConnectionError as exc:
        print(f"  call crashed: {exc}")

    history = cut_to_last_answered_turn(live["messages"])
    print(f"  transcript tail: role={history[-1].role!r} -> resume as-is")
    resp = await agent.call(history=history)
    print(f"  response: {resp.content!r}")


async def scenario_b_cut_and_resume() -> None:
    print("=== B: interrupted mid-tool (tail is an unanswered tool call) ===")
    co.clear_providers()
    co.clear_hooks()
    co.clear_tools()

    tool_started = asyncio.Event()

    @co.register_tool("slow_lookup")
    async def slow_lookup(query: str) -> str:
        """Hang forever, simulating a tool that never returns."""
        tool_started.set()
        await asyncio.sleep(3600)
        return f"results for {query}"

    co.register_provider("mock", ToolCallThenAnswerMock())
    live = capture_live_messages()
    agent = co.Agent(co.AgentConfig(name="t", model="m", provider="mock", tools=["slow_lookup"]))

    task = asyncio.create_task(agent.call("look something up slowly"))
    await tool_started.wait()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    print("  call cancelled while the tool was running")

    history = cut_to_last_answered_turn(live["messages"])
    print(f"  transcript tail after cut: role={history[-1].role!r} -> resume from here")
    resp = await agent.call(history=history)
    print(f"  response: {resp.content!r}")


async def main() -> None:
    await scenario_a_resume_directly()
    print()
    await scenario_b_cut_and_resume()


if __name__ == "__main__":
    asyncio.run(main())
