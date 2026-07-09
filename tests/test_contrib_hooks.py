from __future__ import annotations

from typing import Any

import pytest

from coreouto._types import LLMResponse, Message, ToolResult, Usage
from coreouto.hooks import (
    AFTER_LLM_CALL,
    AFTER_TOOL_CALL,
    ON_ITERATION,
    ON_STREAM_TEXT,
    ON_STREAM_THINKING,
    clear_hooks,
    register_hook,
    trigger,
)


@pytest.fixture(autouse=True)
def _reset_hooks() -> None:
    clear_hooks()


def asyncio_run(coro: Any) -> None:
    import asyncio

    asyncio.run(coro)


def test_token_collection_hook_appends_usage_records() -> None:
    from coreouto.contrib.hooks import token_collection_hook

    sink: list[Usage] = []
    hook, _ = token_collection_hook(sink=sink)
    register_hook(AFTER_LLM_CALL, hook)

    response1 = LLMResponse(
        content="hi",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    response2 = LLMResponse(
        content="ok",
        usage=Usage(prompt_tokens=20, completion_tokens=7, total_tokens=27),
    )

    asyncio_run(trigger(AFTER_LLM_CALL, response=response1))
    asyncio_run(trigger(AFTER_LLM_CALL, response=response2))

    assert len(sink) == 2
    assert sink[0] is response1.usage
    assert sink[1] is response2.usage


def test_token_collection_hook_skips_responses_without_usage() -> None:
    from coreouto.contrib.hooks import token_collection_hook

    sink: list[Usage] = []
    hook, _ = token_collection_hook(sink=sink)
    register_hook(AFTER_LLM_CALL, hook)

    no_usage = LLMResponse(content="nothing")
    with_usage = LLMResponse(
        content="x",
        usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )

    asyncio_run(trigger(AFTER_LLM_CALL, response=no_usage))
    asyncio_run(trigger(AFTER_LLM_CALL, response=with_usage))

    assert sink == [with_usage.usage]


def test_token_collection_hook_factory_returns_sink() -> None:
    from coreouto.contrib.hooks import token_collection_hook

    _, owned = token_collection_hook()
    assert isinstance(owned, list)

    external: list[Usage] = []
    _, returned = token_collection_hook(sink=external)
    assert returned is external


def test_token_collection_hook_uses_owned_sink() -> None:
    from coreouto.contrib.hooks import token_collection_hook

    hook, sink = token_collection_hook()
    register_hook(AFTER_LLM_CALL, hook)

    response = LLMResponse(
        content="hi",
        usage=Usage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
    )
    asyncio_run(trigger(AFTER_LLM_CALL, response=response))

    assert sink == [response.usage]


def test_auto_summarize_hook_triggers_at_threshold() -> None:
    from coreouto.contrib.hooks import auto_summarize_hook

    calls: list[list[Message]] = []

    def summarize(messages: list[Message]) -> list[Message]:
        calls.append(list(messages))
        return [Message(role="system", content="SUMMARY")]

    hook = auto_summarize_hook(threshold=100, summarize_fn=summarize)
    register_hook(ON_ITERATION, hook)

    msgs: list[Message] = [Message(role="user", content="hi")]

    asyncio_run(
        trigger(
            ON_ITERATION,
            iteration=1,
            messages=msgs,
            response=LLMResponse(
                usage=Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30)
            ),
        )
    )
    asyncio_run(
        trigger(
            ON_ITERATION,
            iteration=2,
            messages=msgs,
            response=LLMResponse(
                usage=Usage(prompt_tokens=40, completion_tokens=35, total_tokens=75)
            ),
        )
    )

    assert len(calls) == 1
    assert len(msgs) == 1
    assert msgs[0].content == "SUMMARY"


def test_auto_summarize_hook_mutates_messages_in_place() -> None:
    from coreouto.contrib.hooks import auto_summarize_hook

    summary = [Message(role="system", content="compressed")]

    def summarize(messages: list[Message]) -> list[Message]:
        return summary

    hook = auto_summarize_hook(threshold=1, summarize_fn=summarize)
    register_hook(ON_ITERATION, hook)

    msgs: list[Message] = [
        Message(role="user", content="a"),
        Message(role="user", content="b"),
    ]
    same_list_ref = msgs

    asyncio_run(
        trigger(
            ON_ITERATION,
            iteration=1,
            messages=msgs,
            response=LLMResponse(usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)),
        )
    )

    assert msgs is same_list_ref
    assert msgs == summary


def test_auto_summarize_hook_skips_when_response_has_no_usage() -> None:
    from coreouto.contrib.hooks import auto_summarize_hook

    called: list[bool] = []

    def summarize(messages: list[Message]) -> list[Message]:
        called.append(True)
        return messages

    hook = auto_summarize_hook(threshold=0, summarize_fn=summarize)
    register_hook(ON_ITERATION, hook)

    msgs = [Message(role="user", content="x")]
    asyncio_run(
        trigger(ON_ITERATION, iteration=1, messages=msgs, response=LLMResponse(content="hi"))
    )
    assert called == []


def test_token_limit_warning_hook_fires_when_exceeded() -> None:
    from coreouto.contrib.hooks import token_limit_warning_hook

    fired: list[Usage] = []
    hook = token_limit_warning_hook(limit=50, callback=fired.append)
    register_hook(AFTER_LLM_CALL, hook)

    response = LLMResponse(
        content="x",
        usage=Usage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
    )
    asyncio_run(trigger(AFTER_LLM_CALL, response=response))

    assert len(fired) == 1
    assert fired[0] is response.usage


def test_token_limit_warning_hook_does_not_fire_below_limit() -> None:
    from coreouto.contrib.hooks import token_limit_warning_hook

    fired: list[Usage] = []
    hook = token_limit_warning_hook(limit=100, callback=fired.append)
    register_hook(AFTER_LLM_CALL, hook)

    response = LLMResponse(
        content="x",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    asyncio_run(trigger(AFTER_LLM_CALL, response=response))

    assert fired == []


def test_token_limit_warning_hook_does_not_fire_at_exact_limit() -> None:
    from coreouto.contrib.hooks import token_limit_warning_hook

    fired: list[Usage] = []
    hook = token_limit_warning_hook(limit=50, callback=fired.append)
    register_hook(AFTER_LLM_CALL, hook)

    response = LLMResponse(
        content="x",
        usage=Usage(prompt_tokens=30, completion_tokens=20, total_tokens=50),
    )
    asyncio_run(trigger(AFTER_LLM_CALL, response=response))

    assert fired == []


def test_token_limit_warning_hook_default_callback_prints(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from coreouto.contrib.hooks import token_limit_warning_hook

    hook = token_limit_warning_hook(limit=10)
    register_hook(AFTER_LLM_CALL, hook)

    response = LLMResponse(
        content="x",
        usage=Usage(prompt_tokens=8, completion_tokens=5, total_tokens=13),
    )
    asyncio_run(trigger(AFTER_LLM_CALL, response=response))

    captured = capsys.readouterr()
    assert "13" in captured.out
    assert "10" in captured.out


def test_iteration_notification_hook_fires_at_nth() -> None:
    from coreouto.contrib.hooks import iteration_notification_hook

    fired: list[int] = []
    hook = iteration_notification_hook(every=3, callback=fired.append)
    register_hook(ON_ITERATION, hook)

    for i in (3, 6, 9, 12):
        asyncio_run(trigger(ON_ITERATION, iteration=i))

    assert fired == [3, 6, 9, 12]


def test_iteration_notification_hook_does_not_fire_at_non_nth() -> None:
    from coreouto.contrib.hooks import iteration_notification_hook

    fired: list[int] = []
    hook = iteration_notification_hook(every=5, callback=fired.append)
    register_hook(ON_ITERATION, hook)

    for i in (1, 2, 3, 4, 6, 7, 8, 9, 11):
        asyncio_run(trigger(ON_ITERATION, iteration=i))

    assert fired == []


def test_iteration_notification_hook_fires_at_iteration_zero() -> None:
    from coreouto.contrib.hooks import iteration_notification_hook

    fired: list[int] = []
    hook = iteration_notification_hook(every=1, callback=fired.append)
    register_hook(ON_ITERATION, hook)

    asyncio_run(trigger(ON_ITERATION, iteration=0))
    assert fired == [0]


def test_iteration_notification_hook_default_callback_prints(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from coreouto.contrib.hooks import iteration_notification_hook

    hook = iteration_notification_hook(every=2)
    register_hook(ON_ITERATION, hook)

    asyncio_run(trigger(ON_ITERATION, iteration=4))
    captured = capsys.readouterr()
    assert "4" in captured.out


def test_tool_usage_collection_hook_records_calls() -> None:
    from coreouto.contrib.hooks import tool_usage_collection_hook

    sink: list[tuple[str, str, bool]] = []
    hook, _ = tool_usage_collection_hook(sink=sink)
    register_hook(AFTER_TOOL_CALL, hook)

    result_ok = ToolResult(tool_call_id="1", content="ok", is_error=False)
    result_err = ToolResult(tool_call_id="2", content="bad", is_error=True)

    asyncio_run(trigger(AFTER_TOOL_CALL, name="search", result=result_ok))
    asyncio_run(trigger(AFTER_TOOL_CALL, name="calc", result=result_err))

    assert sink == [
        ("search", "ok", False),
        ("calc", "bad", True),
    ]


def test_tool_usage_collection_hook_factory_returns_sink() -> None:
    from coreouto.contrib.hooks import tool_usage_collection_hook

    _, owned = tool_usage_collection_hook()
    assert isinstance(owned, list)

    external: list[tuple[str, str, bool]] = []
    _, returned = tool_usage_collection_hook(sink=external)
    assert returned is external


def test_stream_printer_hook_prints_deltas(capsys: pytest.CaptureSelector) -> None:
    from coreouto.contrib.hooks import stream_printer_hook

    hook = stream_printer_hook()
    register_hook(ON_STREAM_TEXT, hook)
    asyncio_run(trigger(ON_STREAM_TEXT, text="Hello "))
    asyncio_run(trigger(ON_STREAM_TEXT, text="world"))
    captured = capsys.readouterr()
    assert captured.out == "Hello world"


def test_thinking_printer_hook_prints_deltas(capsys: pytest.CaptureSelector) -> None:
    from coreouto.contrib.hooks import thinking_printer_hook

    hook = thinking_printer_hook()
    register_hook(ON_STREAM_THINKING, hook)
    asyncio_run(trigger(ON_STREAM_THINKING, text="Reasoning "))
    asyncio_run(trigger(ON_STREAM_THINKING, text="step"))
    captured = capsys.readouterr()
    assert captured.out == "Reasoning step"
