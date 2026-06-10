"""Multi-agent orchestration helpers.

`agent_as_tool` wraps a registered agent preset as a callable `Tool`,
allowing a parent agent to delegate sub-tasks to specialised child agents.
The returned tool is **not** auto-registered; the caller is responsible
for wiring it into a parent's tool list (for example, by passing it
directly to `AgentConfig(tools=[...])` or by wrapping it with
`register_tool` if a global name is desired).
"""

from __future__ import annotations

from coreouto.agent import Agent
from coreouto.presets import get_agent_preset
from coreouto.tools import Tool

_DEFAULT_DESCRIPTION_TEMPLATE = (
    "Delegate a sub-task to the {preset} agent. Input is the task description."
)

_TASK_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": "The task description to pass to the sub-agent.",
        }
    },
    "required": ["task"],
}

_DELEGATE_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "agent_name": {
            "type": "string",
            "description": "The name of a registered agent preset to call.",
        },
        "message": {
            "type": "string",
            "description": "The message to pass to the target agent.",
        },
    },
    "required": ["agent_name", "message"],
}

_DEFAULT_DELEGATE_DESCRIPTION = (
    "Call a registered agent by name and pass it a message. "
    "The agent's response is returned as a string."
)


def agent_as_tool(
    preset_name: str,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool:
    preset = get_agent_preset(preset_name)
    config = preset.to_config()
    agent = Agent(config)

    tool_name = name if name is not None else f"call_{preset_name}"
    tool_description = (
        description
        if description is not None
        else _DEFAULT_DESCRIPTION_TEMPLATE.format(preset=preset_name)
    )

    async def handler(task: str) -> str:
        return (await agent.call(task)).content

    return Tool(
        name=tool_name,
        description=tool_description,
        parameters=_TASK_PARAMETERS,
        handler=handler,
    )


def make_delegate_tool(
    *,
    name: str = "call_agent",
    description: str | None = None,
) -> Tool:
    """Create a tool that dispatches to any registered agent preset by name.

    The returned tool accepts two arguments:
    - `agent_name: str` — the name of a registered agent preset
    - `message: str` — the message to pass to that agent

    The handler looks up the preset via `get_agent_preset`, constructs a
    fresh `Agent` from its config, calls it, and returns the response
    content. If the preset is not registered, `KeyError` propagates.
    """
    tool_description = description if description is not None else _DEFAULT_DELEGATE_DESCRIPTION

    async def delegate(agent_name: str, message: str) -> str:
        preset = get_agent_preset(agent_name)
        agent = Agent(preset.to_config())
        return (await agent.call(message)).content

    return Tool(
        name=name,
        description=tool_description,
        parameters=_DELEGATE_PARAMETERS,
        handler=delegate,
    )
