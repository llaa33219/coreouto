# Providers

A provider is the bridge between coreouto and an LLM backend. It translates messages and tool definitions into API calls and parses the responses. coreouto ships four built-in providers and lets you write your own.

## Built-in providers

### OpenAI

```python
import coreouto as co

co.providers.openai.register(api_key="sk-...")

co.register_agent_preset(
    "writer",
    model="gpt-5.5",
    provider="openai",
)
```

Supports `base_url` for OpenAI-compatible endpoints:

```python
co.providers.openai.register(
    api_key="sk-...",
    base_url="http://localhost:11434/v1",
    name="local",
)
```

Recommended models: `gpt-5.5` (flagship, reasoning), `gpt-5.4-mini` (cheap).

Install: `pip install coreouto[openai]`

### Anthropic

```python
co.providers.anthropic.register(api_key="sk-ant-...")

co.register_agent_preset(
    "writer",
    model="claude-opus-4-8",
    provider="anthropic",
)
```

Recommended models: `claude-opus-4-8` (flagship, 1M context), `claude-sonnet-4-6` (best speed/intelligence), `claude-haiku-4-5` (fast/cheap).

Install: `pip install coreouto[anthropic]`

### Google Generative AI

```python
co.providers.google.register(api_key="...")

co.register_agent_preset(
    "writer",
    model="gemini-2.5-pro",
    provider="google",
)
```

Recommended models: `gemini-2.5-pro` (flagship), `gemini-2.5-flash` (best price/perf).

Install: `pip install coreouto[google]`

### OpenAI Response API

Uses OpenAI's Response API instead of the Chat Completions API:

```python
co.providers.openai_response.register(api_key="sk-...")

co.register_agent_preset(
    "writer",
    model="gpt-5.5",
    provider="openai-response",
    provider_config={"reasoning_effort": "medium"},
)
```

`reasoning_effort` accepts `none|minimal|low|medium|high|xhigh`.

Install: `pip install coreouto[openai]`

## Custom endpoints

All built-in providers support custom endpoints via constructor arguments. This is useful for proxying, self-hosting, and region routing.

**OpenAI / OpenAI Responses** — `base_url` constructor arg (see [OpenAI-compatible endpoints](#openai)):

```python
co.providers.openai.register(
    api_key="sk-...",
    base_url="http://localhost:11434/v1",
    name="local",
)
```

**Anthropic** — `base_url` constructor arg:

```python
co.providers.anthropic.register(
    api_key="sk-ant-...",
    base_url="https://proxy.example.com",
    name="anthropic-proxy",
)
```

**Google** — `client_options` constructor arg (a dict passed to the Gemini SDK):

```python
co.register_provider(
    "google-proxy",
    co.providers.GoogleProvider(
        api_key="...",
        client_options={"api_endpoint": "https://proxy.example.com"},
    ),
)
```

`client_options` can also include `transport` (e.g. `{"transport": "grpc"}`) or any other option the `google-generativeai` SDK accepts.

**OpenAI-compatible providers** — any provider with an OpenAI-compatible endpoint can be registered with `OpenAIProvider` and a custom `base_url`:

```python
from coreouto.providers.openai import OpenAIProvider

co.register_provider("minimax", OpenAIProvider(
    api_key="...",
    base_url="https://api.minimax.io/v1",
))

co.register_provider("zhipu", OpenAIProvider(
    api_key="...",
    base_url="https://open.bigmodel.cn/api/paas/v4",
))

co.register_provider("moonshot", OpenAIProvider(
    api_key="...",
    base_url="https://api.moonshot.ai/v1",
))
```

Recommended models:
- MiniMax: `MiniMax-M3` (flagship, 1M context, multimodal)
- Zhipu GLM: `glm-5.1` (flagship), `glm-4.7-flash` (free!)
- Moonshot Kimi: `kimi-k2.6` (flagship, 256K context)

Install: `pip install coreouto[openai]` (these providers use the OpenAI SDK under the hood).

**Note:** `provider_config` does **not** support `base_url` per-call because it is an SDK client-level argument, not a per-call kwarg. Set it once at construction time.

**Multi-endpoint routing:** Register multiple providers with different endpoints and switch between them by name:

```python
co.providers.openai.register(api_key="sk-...", name="openai-us")
co.providers.anthropic.register(
    api_key="sk-ant-...",
    base_url="https://proxy.example.com",
    name="anthropic-proxy",
)
co.register_provider("minimax", OpenAIProvider(
    api_key="...", base_url="https://api.minimax.io/v1",
))
co.register_provider("zhipu", OpenAIProvider(
    api_key="...", base_url="https://open.bigmodel.cn/api/paas/v4",
))
co.register_provider("moonshot", OpenAIProvider(
    api_key="...", base_url="https://api.moonshot.ai/v1",
))
```

## Registering a provider

`register_provider` takes a string key and a provider instance:

```python
co.register_provider("my-provider", MyProvider())
```

The key is what you pass as `provider` in `AgentConfig` or `register_agent_preset`.

## Provider discovery

Check which providers are registered:

```python
co.available_providers()  # => ["anthropic", "google", "openai", "openai-response"]
```

Get a specific provider:

```python
p = co.get_provider("openai")
```

Clear all providers (useful in tests):

```python
co.clear_providers()
```

## Writing a custom provider

A provider must implement three methods. Here's the protocol:

```python
from coreouto._types import LLMResponse, Message, ToolCall, ToolResult
from coreouto.tools import Tool


class MyProvider:
    async def create(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Call the LLM and return a structured response."""
        # Your API call here.
        # Return an LLMResponse with content, tool_calls, and usage.
        ...

    def format_assistant_message(self, response: LLMResponse) -> Message:
        """Convert an LLMResponse into an assistant Message for the history."""
        return Message(
            role="assistant",
            content=response.content or "",
            tool_calls=response.tool_calls or None,
        )

    def format_tool_result(
        self, tool_call: ToolCall, result: ToolResult
    ) -> Message:
        """Convert a tool result into a tool Message for the history."""
        return Message(
            role="tool",
            content=result.content,
            tool_call_id=tool_call.id,
            name=tool_call.name,
        )
```

### Full example: a mock provider for testing

```python
from coreouto._types import LLMResponse, Message, Usage


class MockProvider:
    """Returns a canned response. Useful for tests and local dev."""

    def __init__(self, response_text: str = "Done."):
        self._response_text = response_text

    async def create(self, messages, *, model, tools=None, system_prompt=None, **kwargs):
        return LLMResponse(
            content=self._response_text,
            tool_calls=[],
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )

    def format_assistant_message(self, response):
        return Message(role="assistant", content=response.content or "")

    def format_tool_result(self, tool_call, result):
        return Message(
            role="tool",
            content=str(result.content),
            tool_call_id=tool_call.id,
            name=tool_call.name,
        )


import coreouto as co
co.register_provider("mock", MockProvider("Fusion energy is progressing."))

preset = co.register_agent_preset(
    "test-agent", model="test", provider="mock",
)
response = co.Agent(preset.to_config()).call_sync("Hello")
print(response.content)  # => "Fusion energy is progressing."
```

### Key details

- `create` is always `async`. It receives the full message list, the model name, resolved tool objects, and the system prompt.
- Return an `LLMResponse` with `content` (text), `tool_calls` (list of `ToolCall`), and `usage` (token counts).
- `format_assistant_message` turns that response into a `Message` for the conversation history.
- `format_tool_result` turns a tool execution result into a `Message` for the conversation history.
- The provider is duck-typed (uses `Protocol`). No base class inheritance required.

## Normalized settings

`AgentConfig.provider_config` accepts **8 canonical keys** that coreouto automatically translates to each provider's native kwarg names. This lets you write provider-agnostic configuration:

```python
config = co.AgentConfig(
    name="writer",
    model="gpt-5.5",
    provider="openai",
    provider_config={
        "temperature": 0.7,
        "max_tokens": 2048,
    },
)
```

The same `provider_config` works across all four built-in providers.

### Canonical names and per-provider mapping

| canonical | openai | openai-response | anthropic | google |
|---|---|---|---|---|
| `temperature` | `temperature` | `temperature` | `temperature` | `temperature` |
| `top_p` | `top_p` | `top_p` | `top_p` | `top_p` |
| `max_tokens` | `max_tokens` | `max_output_tokens` | `max_tokens` | `max_output_tokens` |
| `top_k` | — (not supported) | — | `top_k` | `top_k` |
| `stop` | `stop` | — (not supported) | `stop_sequences` | `stop_sequences` |
| `seed` | `seed` | — | — | — |
| `metadata` | — | — | `metadata` | — |
| `reasoning_effort` | — | `{"reasoning": {"effort": value}}` | — | — |

- Standard keys whose value is `None` are dropped before sending to the SDK.
- Standard keys not supported by the chosen provider raise `ValueError` with a helpful message.
- Non-canonical keys raise `ValueError`; use `provider_passthrough` for those (see below).

### Routing across endpoints

`base_url` is constructor-only on each provider instance. To call different endpoints, register multiple providers with different `base_url` values:

```python
co.providers.openai.register(api_key="sk-...", name="openai-us")
co.providers.openai.register(api_key="sk-...", base_url="https://eu.api.openai.com/v1", name="openai-eu")
```

## Explicit pass-through

For provider-specific parameters that are not part of the canonical 8, use `provider_passthrough`. This dict is merged **after** the normalized settings, so passthrough values win on key conflicts:

```python
config = co.AgentConfig(
    name="writer",
    model="gpt-5.5",
    provider="openai",
    provider_config={"temperature": 0.7, "max_tokens": 2048},
    provider_passthrough={"response_format": {"type": "json_object"}},
)
```

`provider_passthrough` is also available on `AgentPreset`:

```python
preset = co.register_agent_preset(
    "writer",
    model="gpt-5.5",
    provider="openai",
    provider_passthrough={"response_format": {"type": "json_object"}},
)
```

> Using a non-canonical key in `provider_config` raises `ValueError`; use `provider_passthrough` instead.
