"""Example 09: Dynamic agent-to-agent dispatch.

`make_delegate_tool()` creates a single tool that dispatches to any
registered agent preset by name. Unlike `agent_as_tool` (static 1:1),
this lets a coordinator choose which sub-agent to call at runtime.

The hierarchy here is:
  coordinator
    └── call_agent  (dynamic dispatch to researcher, writer, or critic)

Run with:
    export OPENAI_API_KEY=sk-...
    python examples/09_agent_delegation.py
"""

from __future__ import annotations

import asyncio
import os

import coreouto as co


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for this example")

    co.providers.openai_response.register(api_key=api_key)

    co.register_agent_preset(
        "researcher",
        model="gpt-5.5",
        provider="openai-response",
        system_prompt="You research topics thoroughly. Wrap your final response in <finish>...</finish> tags.",
    )
    co.register_agent_preset(
        "writer",
        model="gpt-5.5",
        provider="openai-response",
        system_prompt="You write clear, concise summaries. Wrap your final response in <finish>...</finish> tags.",
    )
    co.register_agent_preset(
        "critic",
        model="gpt-5.5",
        provider="openai-response",
        system_prompt="You review text for accuracy and clarity. Wrap your final response in <finish>...</finish> tags.",
    )

    dispatcher = co.make_delegate_tool()
    co.register_tool(
        dispatcher.name,
        description=dispatcher.description,
    )(dispatcher.handler)

    co.register_agent_preset(
        "coordinator",
        model="gpt-5.5",
        provider="openai-response",
        provider_config={"reasoning_effort": "medium"},
        system_prompt=(
            "You coordinate work between agents. Use `call_agent` to delegate. "
            "Pass agent_name (researcher, writer, or critic) and a message. "
            "Wrap your final result in <finish>...</finish> tags when done."
        ),
        tools=["call_agent"],
    )

    response = await co.Agent(
        co.get_agent_preset("coordinator").to_config(),
    ).call(
        "Research the benefits of cold showers, write a short summary, "
        "then critique it for accuracy."
    )
    print("Coordinator:", response.content)


if __name__ == "__main__":
    asyncio.run(main())
