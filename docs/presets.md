# Presets

A preset is a named bundle of agent configuration. Define it once, use it anywhere. Presets separate the "what" (model, tools, system prompt) from the "when" (actually calling the agent).

## Registering a preset

```python
import coreouto as co

preset = co.register_agent_preset(
    "researcher",
    model="gemini-2.5-pro",
    provider="google",
    system_prompt="You are a research assistant. Cite your sources.",
    tools=["search"],
    max_iterations=50,
    description="A research agent that searches the web.",
)
```

All arguments except `name`, `model`, and `provider` are optional.

| Parameter              | Type                | Default | Description                         |
|------------------------|---------------------|---------|-------------------------------------|
| `name`                 | `str`               | --      | Unique identifier (required)        |
| `model`                | `str`               | --      | Model name (required)               |
| `provider`             | `str`               | --      | Provider key (required)             |
| `system_prompt`        | `str \| None`       | `None`  | System message for the agent; if `None`, the [default system prompt](prompts.md#the-default-system-prompt) is used |
| `tools`                | `list[str]`         | `[]`    | Tool names available to the agent   |
| `max_iterations`       | `int \| None`       | `None`  | Max loop iterations (None = unlimited)                 |
| `description`          | `str \| None`       | `None`  | Human-readable description          |
| `provider_config`      | `dict[str, Any]`    | `{}`    | Canonical settings (see [Normalized settings](providers.md#normalized-settings)) |
| `provider_passthrough` | `dict[str, Any]`    | `{}`    | Non-canonical settings sent through to the SDK unchanged |
| `parallel_tool_calls`  | `bool`              | `False` | Forwarded to `AgentConfig`; see [Agent — Parallel tool execution](agent.md#parallel-tool-execution) |

## Using a preset

Convert to an `AgentConfig` and pass to `Agent`:

```python
agent = co.Agent(preset.to_config())
response = await agent.call("What happened in fusion energy this year?")
```

Or synchronously:

```python
response = co.Agent(preset.to_config()).call_sync("What happened in fusion energy this year?")
```

## `to_config()`

The `to_config()` method returns an `AgentConfig` with the preset's values:

```python
config = preset.to_config()
# AgentConfig(name="researcher", model="gemini-2.5-pro", provider="google", ...)
```

You can modify the config before passing it to an agent:

```python
config = preset.to_config()
config.max_iterations = 100
agent = co.Agent(config)
```

## Retrieving presets

Get a preset by name:

```python
preset = co.get_agent_preset("researcher")
print(preset.model)       # "gemini-2.5-pro"
print(preset.description) # "A research agent that searches the web."
```

List all registered preset names:

```python
names = co.list_agent_presets()  # => ["researcher", "writer", ...]
```

Clear all presets (useful in tests):

```python
co.clear_agent_presets()
```

## Separation of preset from invocation

Presets exist as a registry layer between configuration and execution. This matters for a few reasons:

**Reuse.** Register a preset once, use it from multiple places in your codebase. Different modules can create agents from the same preset without passing config objects around.

**Multi-agent.** Presets are the handoff point for `agent_as_tool`. A parent agent delegates to a child by preset name, not by passing config objects. See [Multi-agent](multi-agent.md).

**Testing.** Swap the provider in a preset to test with a mock:

```python
# In production:
co.register_agent_preset("researcher", model="gemini-2.5-pro", provider="google", tools=["search"])

# In tests:
co.register_agent_preset("researcher", model="test", provider="mock", tools=["search"])
```

**Discovery.** `list_agent_presets()` lets you see what's available at runtime, which is useful for building tooling, dashboards, or CLI interfaces on top of coreouto.

## Example: multiple presets

```python
import coreouto as co

co.providers.google.register(api_key="...")

@co.register_tool("search")
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

@co.register_tool("write_draft")
def write_draft(topic: str) -> str:
    """Write a draft on a topic."""
    return f"Draft about: {topic}"

# A researcher that searches
co.register_agent_preset(
    "researcher",
    model="gemini-2.5-pro",
    provider="google",
    system_prompt="You research topics thoroughly.",
    tools=["search"],
)

# A writer that drafts
co.register_agent_preset(
    "writer",
    model="gemini-2.5-pro",
    provider="google",
    system_prompt="You write clear, concise prose.",
    tools=["write_draft"],
)

# Use them independently
research = co.Agent(co.get_agent_preset("researcher").to_config()).call_sync("Find info on fusion energy.")
draft = co.Agent(co.get_agent_preset("writer").to_config()).call_sync("Write an intro about fusion energy.")
```
