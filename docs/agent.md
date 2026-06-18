# Agent

The `Agent` class is the core of coreouto. It takes an `AgentConfig`, runs an internal loop calling the LLM and executing tools, and returns a `Response` when the provider's end-of-turn signal classifies as END — the text of that turn becomes the final answer. The rule is "if the value is not END, keep going", and the END set is defined per-provider from each provider's documented vocabulary. To output text without ending the loop (e.g. share progress before calling more tools), the model calls the built-in `continue_loop` tool.

The end-of-turn field is provider-specific: Anthropic uses `stop_reason`, OpenAI Chat Completions uses `finish_reason`, OpenAI Responses uses `status` (normalized by coreouto to `completed` / `incomplete:<reason>`), and Google Gemini uses `finishReason`. See [How the loop works](#how-the-loop-works) for the exact END values per provider.

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

If `system_prompt` is `None`, a default system prompt is injected automatically that tells the model to use tools and explains the `continue_loop` tool. The model does not need to know how the loop terminates — coreouto reads the provider's end-of-turn signal directly. For the full text of the default and guidance on writing your own, see [System prompts](prompts.md).

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
| `content`        | `str`             | The text of the model's final turn (the turn that ended the loop) |
| `stop_reason`    | `Literal["finish", "max_iterations", "max_tokens", "refusal", "content_filter", "length", "incomplete", "failed", "cancelled"]` | Why the loop terminated. `"finish"` for a clean end-of-turn, `"max_iterations"` when the iteration cap was hit. Provider-specific non-clean terminations are also surfaced: `"max_tokens"` / `"length"` (token cap), `"refusal"` (Anthropic refusal), `"content_filter"` (OpenAI content filter), `"incomplete"` (OpenAI Responses incomplete), `"failed"` / `"cancelled"` (OpenAI Responses terminal states). |
| `messages`       | `list[Message]`   | Full message history (system, user, assistant, tool) |
| `iterations`     | `int`             | How many LLM calls were made                         |
| `usage`          | `list[Usage]`     | Token usage per LLM call                             |

Each `Usage` entry has `prompt_tokens`, `completion_tokens`, and `total_tokens`.

## `MaxIterationsError`

By default `max_iterations` is `None`, which means the loop has no iteration ceiling — it keeps going until the provider emits its natural end-of-turn signal (see [How the loop works](#how-the-loop-works)). Set `max_iterations` to a positive int to cap the loop and raise `MaxIterationsError` after that many iterations without the model terminating:

```python
try:
    response = await agent.call("Do something complex.")
except co.MaxIterationsError as e:
    print(f"Agent didn't terminate: {e}")
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

1. Build the message list: system prompt (default or configured) + history (if any) + user message. Drain any pending user messages injected via `Agent.inject_user_message`.
2. Send the messages and resolved tool definitions to the LLM via the registered provider.
3. Append the assistant's response (with any tool calls) to the message list.
4. Fire the `on_iteration` hook.
5. **Classify the provider's end-of-turn signal** to decide whether the loop ends. The rule is: **if the value is not END, keep going.** The END set is defined per-provider from each provider's documented end-of-turn vocabulary:

   | Provider | Native field | END values (loop terminates) |
   |---|---|---|
   | Anthropic | `stop_reason` | `"end_turn"`, `"max_tokens"`, `"stop_sequence"`, `"pause_turn"`, `"refusal"` |
   | OpenAI Chat Completions | `finish_reason` | `"stop"`, `"length"`, `"content_filter"`, `"function_call"` (legacy) |
   | OpenAI Responses | `status` (normalized to `completed` or `incomplete:<reason>`) | `"failed"`, `"cancelled"`, `"incomplete"` (and `incomplete:<reason>` variants) |
   | Google Gemini | `finish_reason` (enum) | any explicit terminal value: `MAX_TOKENS`, `SAFETY`, `RECITATION`, `LANGUAGE`, `OTHER`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, `MALFORMED_FUNCTION_CALL`, `UNEXPECTED_TOOL_CALL`, `FINISH_REASON_UNSPECIFIED`, `IMAGE_*`, `NO_IMAGE` |
   | any other provider | `stop_reason` (treated as opaque) | unknown — falls back to the historical "no tool calls = end" rule |

   For Google Gemini and OpenAI Responses the field is always END-like (e.g. `STOP` and `completed` are the natural end-of-turn values), so the loop must additionally check whether the response contained tool calls: `STOP` / `completed` is END only when there are no tool calls in the response; any non-STOP finish reason is END regardless of tool calls.

   For Anthropic and OpenAI Chat Completions, the field itself encodes "I just called a tool" (`tool_use` and `tool_calls` respectively). Any other value on those providers — including `None` and any future API addition — is treated as CONTINUE, so the loop never silently ends on an unrecognized signal.

6. **If the loop ends**:
   - The response's `content` becomes `Response.content`.
   - `Response.stop_reason` is one of the literal values listed in [`Response.stop_reason`](#the-response-object) — usually `"finish"`, but provider-specific non-clean terminations (length cap, refusal, content filter, incomplete, failed, cancelled) are surfaced verbatim so callers can distinguish them.
   - The `on_finish` hook fires with `content`, `messages`, and `iterations` (no `tool_call_id`).
   - The `Response` is returned to the caller.
7. **If the loop continues**, execute each tool call (this includes the built-in `continue_loop` — the loop just keeps going), append the results, and return to step 2.
8. If `max_iterations` is set and exceeded, raise `MaxIterationsError`. Default is `None` (unlimited).

> **Provider-driven termination.** The loop ends when the underlying provider's end-of-turn field carries one of the documented END values (see the table above). coreouto does not infer termination from "no tool calls" — it defers to the provider's signal. If the model never produces an end-of-turn signal, the loop keeps running until `max_iterations` is reached.

> **`continue_loop` is not a finish signal.** When the model calls `continue_loop` (with or without other tool calls in the same turn), coreouto executes it like any other tool, appends the result, and continues the loop. The loop only ends when the provider emits a real end-of-turn signal on a turn with no tool calls.

## Hooks during the loop

Seven hook events fire during the loop. See [Hooks](hooks.md) for details:

- `before_llm_call` -- before each LLM request
- `after_llm_call` -- after each LLM response
- `before_tool_call` -- before each tool execution
- `after_tool_call` -- after each tool result
- `on_iteration` -- at the end of each iteration
- `on_finish` -- when the loop terminates (the provider emitted its end-of-turn signal)
- `on_user_injection` -- when a user message is injected via `Agent.inject_user_message`

## Provider config

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

### What if the model never terminates?

The loop keeps calling the LLM. The agent does not raise on a text-only turn — the provider's end-of-turn signal is the termination trigger, not an error. The previous assistant message stays in the conversation so the model can see what it produced and either call `continue_loop` (if it wants to emit more text and keep going) or simply produce a text-only turn on a later iteration to end the loop. The only terminations are the provider's end-of-turn signal, or `max_iterations` if set (in which case `MaxIterationsError` is raised).

### Tracking finish events with hooks

```python
import coreouto as co

def log_finish(*, content, messages, iterations, **kwargs):
    print(f"Agent finished after {iterations} iterations with: {content}")

co.register_hook(co.ON_FINISH, log_finish)
```
