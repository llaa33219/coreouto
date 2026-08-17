"""Recovering from provider 400s caused by a malformed tool call.

A provider 400 about tool calls comes in two shapes:

  A. A DANGLING tool call — the history holds an assistant tool call with
     no matching tool result (e.g. resuming an interrupted session). The
     "tool_result" error reaction is the right fix there: it appends the
     missing results, the request becomes valid, the loop continues.
     (See examples/28_resume_interrupted.py for that shape.)

  B. A MALFORMED tool call — a tool call sitting in the history is itself
     invalid (bad argument types, broken JSON, schema violation), so the
     provider rejects every request that includes it. This example is
     about this shape.

Why the "tool_result" reaction loops forever on shape B: it appends an
error result for the last assistant tool calls but leaves the offending
assistant message in the history. The next `create()` sends the same
poisoned history, the provider 400s again, and each repetition piles
another duplicate result onto the same tool call — which is its own 400
reason on strict providers. Nothing in the reaction ever removes the
cause. ("user_message" fails the same way: it adds a note but keeps the
poison. "terminate" works but abandons the whole task.)

The fix uses two pieces that already exist:

  1. A "retry" error rule — `on_provider_error` fires BEFORE each retry
     attempt, and the retry calls `provider.create()` with the *current*
     messages, so a hook gets a chance to repair the history first.
  2. An `on_provider_error` hook that repairs the history in place: cut
     the poisoned turn (the malformed assistant message plus its tool
     results — orphaned results are rejected too) and append a user
     message telling the model what happened, so it continues with full
     context.

Scenario 1 reproduces the loop against a mock strict provider; the mock
aborts the demo after a few 400s purely to keep the reproduction finite
— that abort is a demo device, NOT a recovery strategy. Scenario 2 adds
the repair hook and recovers.

Run: python examples/29_malformed_tool_call.py
"""

import asyncio

import coreouto as co
from coreouto import ErrorRule
from coreouto._types import LLMResponse, Message, ToolCall, Usage
from coreouto.contrib.error_presets import INVALID_TOOL_ERRORS


class Fake400Error(Exception):
    """Stand-in for a provider SDK's HTTP 400 (e.g. openai.BadRequestError)."""

    status_code = 400


class StrictProviderMock:
    """Mock of a provider that validates request history strictly.

    Call 1: emit a malformed tool call (`query` should be a str, but is an
    int). coreouto executes it fine locally — the provider's 400 is about
    the *request payload* on the NEXT call.

    Every later call: scan the incoming history like a strict provider
    would. If any assistant tool call still carries malformed arguments,
    reject the whole request with a 400. Otherwise answer.
    """

    def __init__(self, abort_after: int | None = None) -> None:
        self.calls = 0
        self.rejections = 0
        self.abort_after = abort_after

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

    async def create(self, messages, *, model, tools=None, system_prompt=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                tool_calls=[ToolCall(id="bad1", name="search", arguments={"query": 123})],
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        for msg in messages:
            if msg.role != "assistant" or not msg.tool_calls:
                continue
            for tc in msg.tool_calls:
                if not isinstance(tc.arguments.get("query"), str):
                    self.rejections += 1
                    if self.abort_after is not None and self.rejections > self.abort_after:
                        raise RuntimeError("demo stop — the 400s would repeat forever")
                    raise Fake400Error("400 Bad Request: malformed tool call in history")
        return LLMResponse(
            content="recovered: history is clean, answering normally",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


def cut_poisoned_turn(messages: list[Message]) -> bool:
    """Cut the last assistant tool-call turn and say why, in place.

    The malformed assistant message AND its tool results are dropped —
    results without their call are rejected as orphans. A user message
    replaces the turn so the model knows its call was rejected and can
    continue with full context. Returns True when a turn was cut.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.role == "assistant" and msg.tool_calls:
            del messages[i:]
            messages.append(
                Message(
                    role="user",
                    content=(
                        "Your previous tool call was rejected by the provider "
                        "(HTTP 400: malformed arguments), so that turn was "
                        "removed from the history. Continue without repeating it."
                    ),
                )
            )
            return True
    return False


def brief(m: Message) -> str:
    if m.role == "assistant" and m.tool_calls:
        calls = ", ".join(f"{tc.name}{tc.arguments}" for tc in m.tool_calls)
        return f"assistant tool_calls=[{calls}]"
    content = m.content if isinstance(m.content, str) else str(m.content)
    return f"{m.role}: {content[:70]!r}"


async def scenario_1_preset_loops() -> None:
    print("=== 1: INVALID_TOOL_ERRORS alone — the 400 repeats forever ===")
    co.clear_providers()
    co.clear_hooks()
    co.clear_tools()

    @co.register_tool("search")
    def search(query: str) -> str:
        """Search the web for `query`."""
        return f"results for {query}"

    mock = StrictProviderMock(abort_after=3)
    mock.error_handling = INVALID_TOOL_ERRORS
    co.register_provider("mock", mock)

    live: dict[str, list[Message]] = {}

    def capture(*, messages: list[Message], **_kwargs) -> None:
        live["messages"] = messages

    co.register_hook(co.BEFORE_LLM_CALL, capture)

    agent = co.Agent(co.AgentConfig(name="t", model="m", provider="mock", tools=["search"]))
    try:
        await agent.call("search for something")
    except RuntimeError as exc:
        print(f"  gave up: {exc}")

    msgs = live["messages"]
    print(f"  provider rejections: {mock.rejections} (would never stop)")
    print(
        "  malformed call still in history: "
        + str(any(m.role == "assistant" and m.tool_calls for m in msgs))
    )
    print(
        "  tool results piled onto call 'bad1': "
        + str(sum(1 for m in msgs if m.role == "tool" and m.tool_call_id == "bad1"))
    )
    print("  transcript tail:")
    for m in msgs[-4:]:
        print(f"    {brief(m)}")


async def scenario_2_repair_and_retry() -> None:
    print("=== 2: retry rule + history-repair hook — the loop recovers ===")
    co.clear_providers()
    co.clear_hooks()
    co.clear_tools()

    @co.register_tool("search")
    def search(query: str) -> str:
        """Search the web for `query`."""
        return f"results for {query}"

    mock = StrictProviderMock()
    mock.error_handling = [
        ErrorRule(
            status_code=400,
            content_contains="malformed",
            reaction="retry",
            message="400 on a malformed tool call — repair history, then retry.",
            retry_after=0.0,
            retry_max=1,
        )
    ]
    co.register_provider("mock", mock)

    def repair(*, status_code, messages, **_kwargs):
        if status_code == 400:
            cut_poisoned_turn(messages)

    co.register_hook(co.ON_PROVIDER_ERROR, repair)

    agent = co.Agent(co.AgentConfig(name="t", model="m", provider="mock", tools=["search"]))
    resp = await agent.call("search for something")
    print(f"  provider rejections before repair: {mock.rejections}")
    print(f"  response: {resp.content!r}")
    print("  repaired transcript:")
    for m in resp.messages:
        if m.role != "system":
            print(f"    {brief(m)}")


async def main() -> None:
    await scenario_1_preset_loops()
    print()
    await scenario_2_repair_and_retry()


if __name__ == "__main__":
    asyncio.run(main())
