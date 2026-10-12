"""Example 32: Hooks on a System One-driven agent.

System One providers fire three extra events — before_turn_input (observe),
on_turn_input (inspect and modify the collected blocks in place), and
on_goal_complete (intercept the completion declaration before the
answer-selection phase). Per-turn System One results ride the existing
after_llm_call event; systemone_collection_hook gathers them for you.

Run with:
    export TYPESAFE_API_KEY=...
    python examples/32_systemone_hooks.py
"""

from __future__ import annotations

import os

import coreouto as co
from coreouto.contrib.hooks import systemone_collection_hook
from coreouto.systemone.typesafe import register as register_typesafe


@co.register_tool("click_ok")
def click_ok() -> str:
    """Click the OK button."""
    return "clicked OK"


@co.register_tool("wait")
def wait() -> str:
    """Wait for the screen to change."""
    return "waited"


def current_screen(messages: list[co.Message]) -> list:
    return [
        co.TextBlock(text="active window: Settings"),
        co.ImageBlock(data=capture_png(), mime_type="image/png"),
    ]


def capture_png() -> bytes:
    """Return the current screen as PNG bytes (platform-specific; stubbed)."""
    return b""


async def main() -> None:
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise RuntimeError("TYPESAFE_API_KEY is required for this example")

    register_typesafe(api_key=os.environ["TYPESAFE_API_KEY"])
    co.register_provider(
        "clef",
        co.SystemOneProvider("typesafe", turn_input=current_screen),
    )

    # collect every per-turn result (goal probability, tool selection)
    collect, results = systemone_collection_hook()
    co.register_hook(co.AFTER_LLM_CALL, collect)

    # modify the turn input in place: drop the screenshot when privacy mode is on
    privacy_mode = False

    def maybe_redact(blocks: list, **_kwargs) -> None:
        if privacy_mode:
            blocks[:] = [b for b in blocks if not isinstance(b, co.ImageBlock)]
            blocks.append(co.TextBlock(text="screenshot withheld (privacy mode)"))

    co.register_hook(co.ON_TURN_INPUT, maybe_redact)

    # intercept the completion declaration before the answer phase
    def log_completion(state: dict, result: co.SystemOneResult, **_kwargs) -> None:
        probability = result.answers["goal_complete"].noul
        print(f"goal declared complete (p={probability:.2f}); running answer selection")

    co.register_hook(co.ON_GOAL_COMPLETE, log_completion)

    preset = co.register_agent_preset(
        "operator",
        model="clef",
        provider="clef",
        system_prompt="Dismiss the dialog on screen using the available tools.",
        tools=["click_ok", "wait"],
    )

    response = await co.Agent(preset.to_config()).call(
        "There is a dialog blocking the app.",
        answer_format={
            "dialog_gone": co.NoulQuestion(
                instructions="Is the dialog gone from the final screen?",
            ),
        },
    )

    print("dialog gone probability:", response.answer.answers["dialog_gone"].noul)
    print(f"results collected: {len(results)}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
