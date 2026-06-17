# Agent

The `Agent` class is the core of coreouto. It takes an `AgentConfig`, runs an internal loop calling the LLM and executing tools, and returns a `Response` when the model calls the built-in `finish` tool with its final answer.

## Creating an agent

An agent needs an `AgentConfig`, which you can build directly or get from a preset:

```python
import coreouto as co

# From a preset:
preset = co.register_agent_preset(
    "writer", model="claude-opus-4-8", provider="anthropic",
    system_prompt="You write clearly and concisely.",
)
agent = co.Agent(preset.to_config())

# From config directly:
config = co.AgentConfig(
    name="writer",
    model="claude-opus-4-8",
    provider="anthropic",
    system_prompt="You write clearly and concisely.",
    tools=[],
    max_iterations=50,
)
agent = co.Agent(config)
```

## `AgentConfig` fields

| Field                  | Type                | Default | Description                              |
|------------------------|---------------------|---------|------------------------------------------|
| `name`                 | `str`               | --      | Agent identifier                         |
| `model`                | `str`               | --      | Model name passed to the provider        |
| `provider`             | `str`               | --      | Key of a registered provider             |
| `system_prompt`        | `str \| None`       | `None`  | System message prepended to the conversation |
| `tools`                | `list[str]`         | `[]`    | Names of registered tools available to this agent |
| `max_iterations`       | `int \| None`       | `None`  | Max loop iterations before raising `MaxIterationsError`; `None` = unlimited (default) |
| `provider_config`      | `dict[str, Any]`    | `{}`    | Canonical settings (see [Normalized settings](providers.md#normalized-settings)); translated to provider-specific kwargs |
| `provider_passthrough` | `dict[str, Any]`    | `{}`    | Non-canonical settings sent through to the SDK unchanged |
| `parallel_tool_calls`  | `bool`              | `False` | Run multiple tool calls in a turn concurrently via `asyncio.gather` (see [Parallel tool execution](#parallel-tool-execution)) |

If `system_prompt` is `None`, a default system prompt is injected automatically explaining the `finish` tool termination protocol.

## Calling the agent

### Async: `call()`

The primary interface. Returns a `Response`:

```python
response = await agent.call("Write a haiku about Python.")
print(response.content)
```

You can pass an `override` config to change settings for a single call:

```python
override = co.AgentConfig(
    name="writer", model="claude-sonnet-4-6", provider="anthropic",
    system_prompt="Be poetic.",
)
response = await agent.call("Write something.", override=override)
```

### Sync: `call_sync()`

Wraps `call()` with `asyncio.run()`. Raises `RuntimeError` if an event loop is already running:

```python
response = agent.call_sync("Write a haiku about Python.")
print(response.content)
```

Use `call_sync()` from scripts and CLI tools. Use `call()` inside async code, web frameworks, or anywhere an event loop exists.

### Conversation history

`call()` and `call_sync()` accept an optional `history: list[Message]` parameter. When provided, the history is prepended to the message list (after the system prompt, before the new user message). coreouto does not store conversation state between calls — that is the caller's responsibility.

Two patterns are common:

**Accumulate** — pass the messages from a previous `Response` to continue the conversation:

```python
r1 = await agent.call("My name is Alice.")
r2 = await agent.call("What is my name?", history=r1.messages)
```

**Fabricate** — hand-craft a list of `Message` objects to seed the agent with any context you want:

```python
from coreouto._types import Message

fake = [
    Message(role="user", content="What is 2 + 2?"),
    Message(role="assistant", content="4"),
    Message(role="user", content="And 3 + 3?"),
    Message(role="assistant", content="6"),
]
response = await agent.call("Continue the pattern.", history=fake)
```

The history is prepended as-is — no implicit processing. If your `cfg.system_prompt` is set AND your history's first message is `role="system"`, you will get two system messages. Slice `history[1:]` if you want to avoid that, or use `override=AgentConfig(system_prompt=...)` to set a different one for the new call.

### Injecting user messages into a running loop

For long-running agents that need external input mid-execution (human-in-the-loop, streaming input, tool-triggered re-prompting), use `Agent.inject_user_message(content)`:

```python
agent = co.Agent(config)

async def push_message_later():
    await asyncio.sleep(1)
    agent.inject_user_message("stop and reconsider")

asyncio.create_task(push_message_later())
response = await agent.call("start the task")
```

The message is queued (thread-safe via `asyncio.Queue`) and drained at the start of the next iteration. It fires the `on_user_injection` hook with `message` and `messages` kwargs for observability.

You can call `inject_user_message()` from anywhere: another thread, another async task, a hook callback, or even before `call()` starts (the queue persists across calls). The loop yields to the scheduler at the start of each iteration, so concurrent tasks get a chance to run.

## Parallel tool execution

When the model issues multiple tool calls in a single turn (very common with Claude and GPT-4.1+), coreouto can run them concurrently. Set `AgentConfig.parallel_tool_calls=True`:

```python
config = co.AgentConfig(
    name="researcher",
    model="claude-sonnet-4-6",
    provider="anthropic",
    tools=["search", "fetch_page", "summarize"],
    parallel_tool_calls=True,
)
```

When this flag is set AND every tool in the turn has `Tool.parallelizable=True` (the default), the agent loop dispatches the turn's tool calls with `asyncio.gather` instead of awaiting them one at a time. Tool result messages are appended to the conversation in input order, not completion order, so the next LLM call sees the same `tool_call_id` ↔ result pairing the model produced.

### Sync tools

Sync (non-`async def`) tool handlers are automatically offloaded to a worker thread via `asyncio.to_thread`. This prevents a long-running sync tool from blocking the event loop while other tools are dispatched in parallel. If your sync tool touches state that isn't safe across threads (a connection pool, a temp file, etc.), keep it parallelizable only when the tool is itself thread-safe, or mark it `parallelizable=False` to keep it serial.

### When the loop falls back to serial

If **any** tool in a turn has `parallelizable=False`, the entire turn runs serially — even when `parallel_tool_calls=True` and every other tool is parallelizable. This is a safety default: the loop won't assume a non-parallelizable tool is safe to overlap with anything else in the same turn.

### Per-provider wire shape

The agent loop preserves text and tool_use ordering when forwarding the conversation to the next LLM call. Anthropic, OpenAI Responses, and Google accept the interleave natively (text blocks followed by tool_use blocks, in the order the model emitted them). OpenAI Chat Completions flattens the text to a single string and emits tool_calls as a separate field — the API doesn't support true interleave, so mid-turn text ordering is lost on that provider. See [Providers — Multimodal support](providers.md#multimodal-support) for per-provider text/tool_use serialization.

### Knobs

| Where | What | Default |
|---|---|---|
| `AgentConfig.parallel_tool_calls` | Master switch: gather when True, serial when False | `False` (unchanged behavior) |
| `Tool.parallelizable` (or `@register_tool(parallelizable=...)`) | Per-tool opt-out | `True` |
| Concurrency cap | None — gather launches all eligible tools at once | n/a |

If you need a concurrency cap, write a hook on `BEFORE_TOOL_CALL` that records start times and use it to monitor, or wrap individual tools with your own semaphore via `provider_passthrough` / a custom provider.

## The `Response` object

`call()` and `call_sync()` both return a `Response`:

| Field            | Type              | Description                                          |
|------------------|-------------------|------------------------------------------------------|
| `content`        | `str`             | The final text extracted from the `finish` tool call's `content` argument |
| `messages`       | `list[Message]`   | Full message history (system, user, assistant, tool) |
| `iterations`     | `int`             | How many LLM calls were made                         |
| `usage`          | `list[Usage]`     | Token usage per LLM call                             |

Each `Usage` entry has `prompt_tokens`, `completion_tokens`, and `total_tokens`.

## `MaxIterationsError`

By default `max_iterations` is `None`, which means the loop has no iteration ceiling — it keeps going until the model calls the `finish` tool (see [How the loop works](#how-the-loop-works)). Set `max_iterations` to a positive int to cap the loop and raise `MaxIterationsError` after that many iterations without a `finish` tool call:

```python
try:
    response = await agent.call("Do something complex.")
except co.MaxIterationsError as e:
    print(f"Agent didn't finish: {e}")
```

```python
config = co.AgentConfig(
    name="deep-researcher",
    model="claude-opus-4-8",
    provider="anthropic",
    tools=["search"],
    max_iterations=200,  # cap the loop
)
```

## How the loop works

1. Build the message list: system prompt (default or configured) + history (if any) + user message.
2. Call the LLM via the registered provider.
3. Inspect `response.tool_calls`:
   - If any tool call has `name == "finish"` and no other tool calls were made, extract the `content` argument from the first such call and return a `Response`.
   - If a `finish` call appears alongside other tool calls, execute only the other calls, append their results, and continue. The `finish` content will be returned on a clean subsequent turn.
   - If the response has no tool calls at all (just text, or an empty response), the loop continues. The model will see its previous response in the next iteration and is expected to either call a tool or call `finish`.
4. If there are non-`finish` tool calls, execute each one, append the results to the message list, and go to step 2.
5. If `max_iterations` is set and exceeded, raise `MaxIterationsError`. Default is `None` (unlimited).

> **Minimal termination.** `finish` is the only signal that ends the loop. Without a `finish` call, the loop keeps running until `max_iterations` is reached. There is no "fail if no `finish`" path — a model that returns text without calling `finish` will simply be re-asked.

> **Mixed-turn behavior.** When the model calls `finish` together with other tools in the same turn, coreouto executes the other tools first and asks the LLM again rather than returning the `finish` content immediately. This keeps tool results in the conversation and avoids dropping unfinished work.

## Hooks during the loop

Six hook events fire during the loop. See [Hooks](hooks.md) for details:

- `before_llm_call` -- before each LLM request
- `after_llm_call` -- after each LLM response
- `before_tool_call` -- before each tool execution
- `after_tool_call` -- after each tool result
- `on_iteration` -- at the end of each iteration
- `on_finish` -- when the model calls the `finish` tool
- `on_user_injection` -- when a user message is injected via `Agent.inject_user_message`

## Provider config and the `finish` reminder

### `provider_config`

`AgentConfig.provider_config` is a `dict[str, Any]` of **canonical settings** that coreouto normalizes to each provider's native kwarg names (e.g., `max_tokens` automatically becomes `max_output_tokens` for OpenAI Responses and Google). Use it for the 8 cross-provider settings: `temperature`, `top_p`, `max_tokens`, `top_k`, `stop`, `seed`, `metadata`, `reasoning_effort`. See [Normalized settings](providers.md#normalized-settings) for the full mapping table.

For non-canonical, provider-specific settings (like OpenAI's `response_format` or Anthropic's `thinking`), use `AgentConfig.provider_passthrough` instead -- it is sent through to the SDK unchanged.

```python
config = co.AgentConfig(
    name="writer",
    model="claude-opus-4-8",
    provider="anthropic",
    provider_config={"temperature": 0.3, "max_tokens": 1024},
)
```

### What if the model forgets to call `finish`?

The loop keeps running. If the model returns plain text (or an empty response) without calling `finish`, the agent does not raise — it simply iterates again. The previous assistant message stays in the conversation so the model can see what it produced and correct course (call a tool or call `finish`). The only termination is `finish` itself, or `max_iterations` if set.

### Tracking finish events with hooks

```python
import coreouto as co

def log_finish(*, content, messages, iterations, tool_call_id, **kwargs):
    print(f"Agent finished after {iterations} iterations with: {content}")

co.register_hook(co.ON_FINISH, log_finish)
```
