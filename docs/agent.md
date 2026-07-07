# Agent

The `Agent` class is the core of coreouto. It takes an `AgentConfig`, runs an internal loop calling the LLM and executing tools, and returns a `Response` when the model produces a response with **text content and no tool calls**. That response's text content becomes the final answer. A response with tool calls means the model is mid-task: the loop executes the tools and continues.

Unrecoverable provider terminations (token cap, refusal, content filter, server failure) end the loop immediately, even if the response carries tool calls — those tool calls are dropped as a security/correctness invariant.

There are no reserved tools. The only tools in the loop are the ones you register.

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
| `retry_intervals`      | `list[float] \| None` | `None` | Seconds to wait between retries when a provider call raises; `None` (or `[]`) = no retry, the exception propagates immediately (default). See [Retrying failed API calls](#retrying-failed-api-calls) |
| `provider_config`      | `dict[str, Any]`    | `{}`    | Canonical settings (see [Normalized settings](providers.md#normalized-settings)); translated to provider-specific kwargs |
| `provider_passthrough` | `dict[str, Any]`    | `{}`    | Non-canonical settings sent through to the SDK unchanged |
| `parallel_tool_calls`  | `bool`              | `False` | Run multiple tool calls in a turn concurrently via `asyncio.gather` (see [Parallel tool execution](#parallel-tool-execution)) |

If `system_prompt` is `None`, a default system prompt is injected automatically that tells the model to use tools and explains the termination rule: call tools to work, respond with text and no tool calls to finish. For the full text of the default and guidance on writing your own, see [System prompts](prompts.md).

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
| `stop_reason`    | `Literal["finish", "max_iterations", "max_tokens", "refusal", "content_filter", "length", "incomplete", "failed", "cancelled"]` | Why the loop terminated. `"finish"` = clean text-only termination (text content, no tool calls). `"max_iterations"` = iteration cap hit. The remaining literals are unrecoverable provider terminations: `"max_tokens"` / `"length"` (token cap), `"refusal"` (Anthropic refusal), `"content_filter"` (OpenAI content filter), `"incomplete"` (OpenAI Responses incomplete), `"failed"` / `"cancelled"` (OpenAI Responses terminal states). |
| `messages`       | `list[Message]`   | Full message history (system, user, assistant, tool) |
| `iterations`     | `int`             | How many LLM calls were made                         |
| `usage`          | `list[Usage]`     | Token usage per LLM call                             |

Each `Usage` entry has `prompt_tokens`, `completion_tokens`, and `total_tokens`.

## `MaxIterationsError`

By default `max_iterations` is `None`, which means the loop has no iteration ceiling — it keeps going until the model produces a response with text content and no tool calls (or an unrecoverable provider termination occurs). Set `max_iterations` to a positive int to cap the loop and raise `MaxIterationsError` after that many iterations without the model terminating:

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

## Retrying failed API calls

By default `retry_intervals` is `None`, so when a provider call raises (network error, timeout, rate limit, 5xx, etc.) the exception propagates immediately and the loop ends — coreouto does not retry on its own.

Set `retry_intervals` to a list of seconds to wait before each retry attempt. The first failure triggers a sleep of `intervals[0]` then a retry; if that fails, a sleep of `intervals[1]` then another retry; and so on. When every interval is exhausted, the last exception is re-raised. An empty list behaves like `None` (no retry).

```python
config = co.AgentConfig(
    name="resilient",
    model="claude-sonnet-4-6",
    provider="anthropic",
    retry_intervals=[1, 3, 5, 10, 30],  # up to 5 retries at these delays
)
```

A few notes:

- `retry_intervals=[1, 3, 5]` means **1 initial attempt + up to 3 retries** (4 total `provider.create` calls in the worst case).
- Intervals must be non-negative. `0` is allowed and means "retry immediately with no delay".
- The retry wraps every `provider.create` call in the loop, on every iteration — not just the first.
- **Any `Exception` raised by the provider triggers a retry.** This covers network, timeout, rate-limit, and server errors uniformly without coreouto importing provider-specific exception types. A programmer error (bad kwargs) will simply fail again on each retry and surface once the intervals are spent.
- Unrecoverable provider *terminations* (`max_tokens`, `refusal`, content filter, `SAFETY`, …) are **not** exceptions — they are normal responses that end the loop. Retry only applies when the provider call itself throws.

The same field is exposed on presets:

```python
preset = co.register_agent_preset(
    "resilient", model="claude-sonnet-4-6", provider="anthropic",
    retry_intervals=[1, 3, 5, 10, 30],
)
```

### Observing retries with `on_retry`

Each retry fires the `on_retry` hook **before** the sleep, with `{attempt, interval, error, messages, model}`. Use it to log or metric retries:

```python
def log_retry(*, attempt, interval, error, **kwargs):
    print(f"retry #{attempt} after {interval}s: {type(error).__name__}: {error}")

co.register_hook(co.ON_RETRY, log_retry)
```

`attempt` is 1-indexed (the first retry is attempt `1`). `before_llm_call` / `after_llm_call` bracket the logical per-iteration LLM call and fire once per iteration regardless of how many retries happened underneath.

## How the loop works

coreouto's termination rule is simple: **a response with text content and no tool calls ends the loop; the assistant text is the final answer.** A response with tool calls continues the loop. A response with no content and no tool calls (e.g. a thinking-only turn) also continues the loop — the model is treated as still working.

Each iteration follows these steps:

1. Drain any pending user messages injected via `Agent.inject_user_message`, firing `on_user_injection` for each.
2. Increment the iteration counter. If `max_iterations` is set and exceeded, raise `MaxIterationsError`.
3. Fire `before_llm_call`; call the provider with the messages and the user's resolved tools (no injected tools).
4. Fire `after_llm_call`; append the assistant message to the conversation; fire `on_iteration`.
5. **Classify the response into one of four branches:**

   - **Unrecoverable provider termination** — `max_tokens`, `refusal`, `length`, `content_filter`, `SAFETY`, `RECITATION`, `LANGUAGE`, `OTHER`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, `IMAGE_*`, `NO_IMAGE`, `failed`, `cancelled`, `incomplete:content_filter`. The provider refused, truncated, or filtered the response. Terminate immediately **without executing any tool calls** the response may carry. `Response.stop_reason` surfaces the provider signal. `Response.content` falls back to the response text (often empty).
   - **Text content + no tool calls** — the model is done. The assistant text is the final answer. `Response.stop_reason` is `"finish"`.
   - **No content + no tool calls** — the model produced neither text nor tool calls (e.g. a thinking-only turn, or an otherwise empty response). The loop continues (re-prompts) rather than terminating with an empty answer. Still bounded by `max_iterations`.
   - **Tool calls present (recoverable turn)** — execute the tool calls, append the results, and return to step 1.

6. On termination: `Response.content` is the assistant text (or fallback); the `on_finish` hook fires with `{content, messages, iterations}`; the `Response` is returned to the caller.

> **A response with text content and no tool calls ends the loop. A response with tool calls — or with no content and no tool calls — continues it.** The assistant text from the final (text-only, no-tool-call) response becomes `Response.content`. This is the only termination rule for the model — there is no special tool to call, no confirmation step, no re-prompting.

### Progress mid-task

Since there is no special tool for mid-task output, the model shares progress by including text alongside a tool call in the same turn. Most providers support assistant text and tool calls in a single message. That text lives in the assistant message in the conversation. (A response with only a tool call and no text continues the loop; a response with text and no tool calls ends it.)

To surface progress text to an end user (e.g. stream it), register an `on_iteration` hook and read `response.content`:

```python
import coreouto as co

def show_progress(*, response, **kwargs):
    if response.content:
        print(f"Progress: {response.content}")

co.register_hook("on_iteration", show_progress)
```

The `on_iteration` hook fires every iteration with the full `LLMResponse`, so `response.content` captures any progress text the model emitted alongside its tool calls.

### `max_output_tokens` truncation pitfall

> **`max_output_tokens` too small truncates tool calls.** If the provider's output token limit (`max_tokens` / `max_output_tokens`) is set too low, the model's response can be cut off mid-generation. When the cut falls inside a tool-call's JSON, the tool call becomes unparseable and is dropped — so the turn ends with no executable tool call. Depending on the provider, this surfaces either as an unrecoverable `max_tokens`/`length` termination (the loop ends with `stop_reason="max_tokens"`, often with empty or partial content) or, worse, as a content-only turn that terminates the loop prematurely before the intended tool work happened. Always set the output token limit high enough to fit the model's full response including any tool-call JSON.

## Hooks during the loop

Eight hook events fire during the loop. See [Hooks](hooks.md) for details:

- `before_llm_call` -- before each LLM request
- `after_llm_call` -- after each LLM response
- `before_tool_call` -- before each tool execution
- `after_tool_call` -- after each tool result
- `on_iteration` -- at the end of each iteration
- `on_finish` -- when the loop terminates (text content with no tool calls, or an unrecoverable provider termination)
- `on_user_injection` -- when a user message is injected via `Agent.inject_user_message`
- `on_retry` -- before each retry sleep when `retry_intervals` is set (see [Retrying failed API calls](#retrying-failed-api-calls))

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

The loop keeps calling the LLM. The loop ends when the model produces a response with text content and no tool calls, when an unrecoverable provider termination occurs, or when `max_iterations` is reached. If the model keeps issuing tool calls indefinitely and `max_iterations` is not set, the loop runs forever. Set `max_iterations` to guard against that.

### Tracking finish events with hooks

```python
import coreouto as co

def log_finish(*, content, messages, iterations, **kwargs):
    print(f"Agent finished after {iterations} iterations with: {content}")

co.register_hook(co.ON_FINISH, log_finish)
```
