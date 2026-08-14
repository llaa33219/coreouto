"""Recovering a stalled agent loop: the wakeup pattern.

A provider call can hang silently (dead socket, stuck stream) or the loop
can stop progressing for an unknown reason. coreouto does not supervise the
loop for you — you build the supervisor from three pieces:

  1. Provider `timeout` (seconds) turns a hung HTTP call into an exception.
  2. `TIMEOUT_ERRORS` rules turn that exception into an automatic retry:

        from coreouto.contrib.error_presets import COMMON_HTTP_ERRORS, TIMEOUT_ERRORS
        from coreouto.providers.openai import OpenAIProvider

        provider = OpenAIProvider(
            api_key="sk-...",
            timeout=120,
            error_handling=COMMON_HTTP_ERRORS + TIMEOUT_ERRORS,
        )

  3. The tracker hooks in coreouto.contrib.hooks answer "how long has it
     been?" and "is the loop still working on something?":
       - activity_tracker_hook: seconds since the last loop event
       - api_call_tracker_hook: seconds since the last API request started
       - loop_progress_hook: current phase + is_stalled(after_seconds)

The watchdog below uses them: if nothing happened for STALL_LIMIT seconds
(real deployments use e.g. 1800), it cancels the stuck call, recovers the
live message history from the hooks, and starts a fresh call with that
history so the agent resumes where it left off.

This example runs on a mock provider whose first API call hangs forever,
so the watchdog fires and the wakeup call completes.

Run: python examples/27_wakeup.py
"""

import asyncio
import contextlib

import coreouto as co
from coreouto._types import LLMResponse, Message, Usage
from coreouto.contrib.hooks import (
    activity_tracker_hook,
    api_call_tracker_hook,
    loop_progress_hook,
)

STALL_LIMIT = 3.0  # seconds of silence before we declare the loop dead
CHECK_INTERVAL = 0.5


class HangOnceMock:
    """First API call hangs forever (wedged connection); later calls answer."""

    def __init__(self) -> None:
        self.calls = 0

    async def create(self, messages, *, model, tools=None, system_prompt=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(3600)
        return LLMResponse(
            content=f"finished after {self.calls} API call(s)",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    def format_assistant_message(self, response):
        return Message(role="assistant", content=response.content or "")

    def format_tool_result(self, tool_call, result):
        return Message(
            role="tool",
            content=result.content,
            tool_call_id=tool_call.id,
            name=tool_call.name,
        )


def sanitize_history(messages: list[Message]) -> list[Message]:
    """Prepare a cancelled loop's messages for `call(history=...)`.

    Drops system messages (call() prepends its own) and a trailing
    assistant turn whose tool calls never got their results — providers
    reject dangling tool calls.
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


async def main() -> None:
    co.clear_providers()
    co.clear_hooks()
    co.register_provider("mock", HangOnceMock())

    activity_hook, activity = activity_tracker_hook()
    api_hooks, api = api_call_tracker_hook()
    progress_hooks, progress = loop_progress_hook()

    for event in (co.AFTER_LLM_CALL, co.AFTER_TOOL_CALL, co.ON_ITERATION, co.ON_PROVIDER_ERROR):
        co.register_hook(event, activity_hook)
    for hooks in (api_hooks, progress_hooks):
        for event, fn in hooks.items():
            co.register_hook(event, fn)

    live: dict[str, list[Message]] = {}

    def capture_messages(*, messages: list[Message], **_kwargs) -> None:
        live["messages"] = messages

    co.register_hook(co.BEFORE_LLM_CALL, capture_messages)

    agent = co.Agent(co.AgentConfig(name="worker", model="m", provider="mock"))

    task = asyncio.create_task(agent.call("do the work"))
    wakeups = 0
    while True:
        await asyncio.sleep(CHECK_INTERVAL)
        if task.done():
            break
        silent = activity.seconds_since_last_activity()
        if silent > STALL_LIMIT:
            wakeups += 1
            print(
                f"[watchdog] no activity for {silent:.1f}s "
                f"(phase={progress.phase!r}, api_in_flight={api.in_flight}) — waking up"
            )
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            history = sanitize_history(live["messages"])
            print(f"[watchdog] recovered {len(history)} messages, restarting the call")
            task = asyncio.create_task(
                agent.call("You were interrupted. Resume the task.", history=history)
            )

    print(f"response: {task.result().content!r} (wakeups: {wakeups})")


if __name__ == "__main__":
    asyncio.run(main())
