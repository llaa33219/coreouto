"""Example 30: A System One-driven agent (TypeSafe Jev).

A SystemOneProvider wraps a System One model in the Provider protocol, so the
agent loop runs on *selections* instead of generated text: each iteration the
model decides whether the goal is complete and, if not, which tool to call
next. Tools must be selectable — required parameters need an enum (a
`Literal` type works). `answer_format` declares the typed questions the
final answer must answer; its SystemOneResult lands on `Response.answer`.

Run with:
    export TYPESAFE_API_KEY=...
    python examples/30_systemone_agent.py
"""

from __future__ import annotations

import os
from typing import Literal

import coreouto as co
from coreouto.systemone.typesafe import register as register_typesafe


@co.register_tool("restart_service")
def restart_service() -> str:
    """Restart the unhealthy service."""
    return "service restarted"


@co.register_tool("scale_fleet")
def scale_fleet(direction: Literal["up", "down"]) -> str:
    """Scale the fleet up or down."""
    return f"fleet scaled {direction}"


@co.register_tool("page_oncall")
def page_oncall() -> str:
    """Page the on-call engineer."""
    return "on-call paged"


async def main() -> None:
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise RuntimeError("TYPESAFE_API_KEY is required for this example")

    register_typesafe(api_key=api_key)  # System One model "typesafe"
    co.register_provider("jev", co.SystemOneProvider("typesafe"))

    preset = co.register_agent_preset(
        "incident",
        model="jev-latest",
        provider="jev",
        system_prompt="Resolve the incident using the available tools.",
        tools=["restart_service", "scale_fleet", "page_oncall"],
    )

    response = await co.Agent(preset.to_config()).call(
        "API latency is 10x normal and error rate is climbing.",
        answer_format={
            "resolution": co.ChoiceQuestion(
                instructions="How was the incident resolved?",
                criteria={
                    "auto_remediated": "a tool action fixed it",
                    "escalated": "humans were paged",
                },
            ),
        },
    )

    for message in response.messages:
        if message.role == "assistant" and message.tool_calls:
            for call in message.tool_calls:
                print(f"decided: {call.name}({call.arguments})")
    print("resolution:", response.answer.answers["resolution"].choice)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
