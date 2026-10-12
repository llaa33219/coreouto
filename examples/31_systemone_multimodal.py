"""Example 31: A screen-watching System One agent (multimodal turn input).

Clef (Cloudflare) and other multimodal System One models score images alongside
the state. `SystemOneProvider(turn_input=...)` calls your callback every loop
iteration, so the model always decides on the *current* screen. The callback
receives the transcript and returns content blocks (or None); sync or async.

Run with:
    export TYPESAFE_API_KEY=...
    export TYPESAFE_BASE_URL=https://api.cloudflare.com/client/v4/accounts/<acct>/ai/v1
    python examples/31_systemone_multimodal.py
"""

from __future__ import annotations

import os

import coreouto as co
from coreouto.systemone.typesafe import register as register_typesafe


@co.register_tool("click_ok")
def click_ok() -> str:
    """Click the OK button."""
    return "clicked OK"


@co.register_tool("wait")
def wait() -> str:
    """Wait for the screen to change."""
    return "waited"


def current_screen(messages: list[co.Message]) -> list[co.ImageBlock] | None:
    png = capture_screen_png()
    return [co.ImageBlock(data=png, mime_type="image/png")] if png else None


def capture_screen_png() -> bytes | None:
    """Return the current screen as PNG bytes (platform-specific; stubbed)."""
    return None


async def main() -> None:
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise RuntimeError("TYPESAFE_API_KEY is required for this example")

    register_typesafe(api_key=os.environ["TYPESAFE_API_KEY"])
    co.register_provider(
        "clef",
        co.SystemOneProvider("typesafe", turn_input=current_screen),
    )

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


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
