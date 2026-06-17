# Multi-agent

coreouto supports multi-agent workflows through `agent_as_tool`, which wraps a registered preset as a callable tool. A parent agent can then delegate sub-tasks to specialized child agents.

## The `agent_as_tool` helper

Wrap a preset as a tool:

```python
import coreouto as co
from coreouto.providers.openai import OpenAIProvider

co.register_provider("moonshot", OpenAIProvider(
    api_key="...",
    base_url="https://api.moonshot.ai/v1",
))
co.providers.openai.register(api_key="sk-...")

@co.register_tool("search")
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"
```

The returned `Tool` has:
- **Name:** `call_researcher` (default: `call_{preset_name}`)
- **Description:** "Delegate a sub-task to the researcher agent. Input is the task description."
- **Parameters:** a single `task` string field

## Registering the tool for a parent agent

`agent_as_tool` returns a `Tool` object but does not register it globally. You need to wire it into the parent's tool list yourself:

```python
# Register it globally with a custom name
co.register_tool("delegate_research")(research_tool.handler)

# Or pass it when building the parent config
# (requires manual wiring — see below)
```

The simplest approach: register the handler as a tool with a name of your choosing:

```python
# Re-register under a clearer name
co.register_tool("delegate_to_researcher")(research_tool.handler)
```

## Using it in a parent agent

```python
# Parent orchestrator
co.register_agent_preset(
    "orchestrator",
    model="kimi-k2.6",
    provider="moonshot",
    system_prompt="You are an orchestrator. Delegate research tasks to the researcher.",
    tools=["delegate_to_researcher"],
)

response = co.Agent(
    co.get_agent_preset("orchestrator").to_config()
).call_sync("Research the latest developments in fusion energy and summarize them.")
print(response.content)
```

The orchestrator agent will call `delegate_to_researcher` with a task string. That triggers a full agent loop in the child (the researcher), which can use its own tools. The child's final response comes back as the tool result.

## Custom name and description

```python
research_tool = co.agent_as_tool(
    "researcher",
    name="research",
    description="Run a deep research task. Pass the research question as input.",
)
```

## Message isolation

Each child agent gets its own message history. The child starts fresh with its system prompt and the task string as the user message. It does not see the parent's conversation.

This is deliberate. Child agents are sub-tasks, not continuations. They should be focused and stateless from the parent's perspective.

## Why this is not built into the core

Multi-agent orchestration is a design choice, not a default. coreouto's core loop is single-agent: call, loop, return. Wiring agents together is an application-level concern.

`agent_as_tool` exists as a helper because the pattern is common, but it's just one way to compose agents. You might prefer:

- **Sequential pipelines:** call one agent, pass its output to the next
- **Fan-out:** call multiple agents in parallel with `asyncio.gather`
- **Custom orchestration:** build your own loop with full control

The point is that coreouto gives you the pieces (agents, tools, presets) and you decide how they connect.

## Example: sequential pipeline

```python
import coreouto as co

co.providers.openai.register(api_key="sk-...")
co.providers.anthropic.register(api_key="sk-ant-...")

co.register_agent_preset(
    "researcher", model="gpt-5.5", provider="openai",
    system_prompt="Research the topic thoroughly.",
)
co.register_agent_preset(
    "writer", model="claude-sonnet-4-6", provider="anthropic",
    system_prompt="Write a clear summary based on the research.",
)

async def pipeline(topic: str) -> str:
    research = co.Agent(co.get_agent_preset("researcher").to_config())
    writer = co.Agent(co.get_agent_preset("writer").to_config())

    findings = await research.call(f"Research: {topic}")
    summary = await writer.call(f"Summarize this research:\n{findings.content}")
    return summary.content
```

This gives you full control over what passes between agents.

## Example: fan-out with asyncio

```python
import asyncio
import coreouto as co

async def research_topic(topic: str) -> str:
    agent = co.Agent(co.get_agent_preset("researcher").to_config())
    result = await agent.call(f"Research: {topic}")
    return result.content

async def fan_out(topics: list[str]) -> list[str]:
    results = await asyncio.gather(*[research_topic(t) for t in topics])
    return list(results)
```

## Dynamic dispatch (`make_delegate_tool`)

`agent_as_tool` creates a static 1:1 binding: one tool per preset. `make_delegate_tool` is the dynamic alternative. It creates a single tool that accepts an `agent_name` argument, letting the parent agent choose which sub-agent to call at runtime.

### Tool signature

- **Name:** `call_agent` (customizable via `name=`)
- **Parameters:** `agent_name: str`, `message: str`
- **Behavior:** looks up the preset, builds a fresh `Agent`, calls it, returns the response content

### Creating and registering

```python
import coreouto as co

dispatcher = co.make_delegate_tool()
co.register_tool(dispatcher.name, description=dispatcher.description)(dispatcher.handler)
```

### Example: coordinator with dynamic dispatch

```python
co.register_agent_preset(
    "researcher", model="gpt-5.5", provider="openai",
    system_prompt="You research topics thoroughly.",
)
co.register_agent_preset(
    "writer", model="claude-sonnet-4-6", provider="anthropic",
    system_prompt="You write clear summaries.",
)

dispatcher = co.make_delegate_tool()
co.register_tool(dispatcher.name, description=dispatcher.description)(dispatcher.handler)

co.register_agent_preset(
    "coordinator", model="kimi-k2.6", provider="moonshot",
    system_prompt=(
        "You coordinate work. Use `call_agent` to delegate tasks. "
        "Pass the agent name and a message. When you are done, respond with text and no tool calls to end the loop. "
        "If you need to share progress while still doing more work, call the `continue_loop` tool."
    ),
    tools=["call_agent"],
)
```

The coordinator can now call `call_agent(agent_name="researcher", message="...")` or `call_agent(agent_name="writer", message="...")` based on the task. Unlike `agent_as_tool`, you don't need a separate tool registration per sub-agent.
