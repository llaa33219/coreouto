# Quickstart

Walk through installing coreouto, registering a provider and tools, defining an agent preset, and making your first call. By the end you'll have a working agent with a mock provider (no API key needed).

## 1. Install

```bash
pip install coreouto
```

For a specific provider:

```bash
pip install coreouto[openai]      # OpenAI
pip install coreouto[anthropic]   # Anthropic
pip install coreouto[google]      # Google Generative AI
pip install coreouto[all]         # everything
```

## 2. Register a provider

Providers are the bridge between coreouto and your LLM. Register one with a string key:

```python
import os

import coreouto as co
from coreouto.providers.openai import OpenAIProvider

# Real provider (requires an API key):
co.register_provider("zhipu", OpenAIProvider(
    api_key=os.environ["ZHIPU_API_KEY"],
    base_url="https://open.bigmodel.cn/api/paas/v4",
))
```

For testing without API keys, you can use a mock provider:

```python
from coreouto._types import LLMResponse, Message, ToolCall, Usage

class MockProvider:
    """Returns a canned response on every call."""

    async def create(self, messages, *, model, tools=None, system_prompt=None, **kwargs):
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id="finish_1",
                    name="finish",
                    arguments={"content": "Fusion energy made significant progress in 2025."},
                ),
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=8, total_tokens=18),
        )

    def format_assistant_message(self, response):
        return Message(
            role="assistant",
            content=response.content or "",
            tool_calls=response.tool_calls if response.tool_calls else None,
        )

    def format_tool_result(self, tool_call, result):
        return Message(role="tool", content=str(result.content), tool_call_id=tool_call.id, name=tool_call.name)

co.register_provider("mock", MockProvider())
```

## 3. Register a tool

Tools are functions the agent can call. The decorator extracts type hints into JSON Schema automatically:

```python
@co.register_tool("search")
def search(query: str) -> str:
    """Search the web for `query`."""
    return f"Top results for: {query}"
```

## 4. Define an agent preset

A preset bundles a model, provider, system prompt, and tool list into a reusable configuration:

```python
preset = co.register_agent_preset(
    "researcher",
    model="glm-5.1",
    provider="mock",       # swap to "zhipu" for real calls
    system_prompt="You are a research assistant. Use the search tool when needed.",
    tools=["search"],
)
```

## 5. Call the agent

Create an `Agent` from the preset's config and call it:

```python
import asyncio

async def main():
    agent = co.Agent(preset.to_config())
    response = await agent.call("What's new in fusion energy?")
    print(response.content)
    print(f"Iterations: {response.iterations}")

asyncio.run(main())
```

Or use the synchronous wrapper if you're not in an async context:

```python
agent = co.Agent(preset.to_config())
response = agent.call_sync("What's new in fusion energy?")
print(response.content)
```

## Complete example

Here's the whole thing in one file, using a mock provider:

```python
import coreouto as co
from coreouto._types import LLMResponse, Message, ToolCall, Usage


class MockProvider:
    """Provider that returns a canned response."""

    async def create(self, messages, *, model, tools=None, system_prompt=None, **kwargs):
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id="finish_1",
                    name="finish",
                    arguments={"content": "Fusion energy saw major breakthroughs in 2025."},
                ),
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=8, total_tokens=18),
        )

    def format_assistant_message(self, response):
        return Message(
            role="assistant",
            content=response.content or "",
            tool_calls=response.tool_calls if response.tool_calls else None,
        )

    def format_tool_result(self, tool_call, result):
        return Message(
            role="tool", content=str(result.content),
            tool_call_id=tool_call.id, name=tool_call.name,
        )


@co.register_tool("search")
def search(query: str) -> str:
    """Search the web for `query`."""
    return f"Results for: {query}"


co.register_provider("mock", MockProvider())

preset = co.register_agent_preset(
    "researcher",
    model="test-model",
    provider="mock",
    system_prompt="You are a research assistant.",
    tools=["search"],
)

response = co.Agent(preset.to_config()).call_sync("What's new in fusion energy?")
print(response.content)
# => "Fusion energy saw major breakthroughs in 2025."
```

## Next steps

- [Agent](agent.md) -- learn about `call()`, `call_sync()`, and `Response`
- [Providers](providers.md) -- set up a real provider
- [Tools](tools.md) -- build more complex tools
- [Hooks](hooks.md) -- add logging, token tracking, or custom behavior
