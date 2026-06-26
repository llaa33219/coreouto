# Hooks

Hooks let you inject behavior at specific points in the agent loop without modifying the core. Register a function for an event and it fires every time that event occurs.

## The seven events

| Event              | When it fires                        | Keyword arguments                                      |
|--------------------|--------------------------------------|--------------------------------------------------------|
| `before_llm_call`  | Before each LLM request              | `messages`, `model`, `tools`                           |
| `after_llm_call`   | After each LLM response              | `response`, `messages`                                 |
| `before_tool_call` | Before each tool executes            | `name`, `arguments`                                    |
| `after_tool_call`  | After each tool result               | `name`, `result`                                       |
| `on_iteration`     | At the end of each loop iteration    | `iteration`, `messages`, `response`                    |
| `on_finish`        | When the loop terminates (text content with no tool calls, or an unrecoverable provider termination) | `content`, `messages`, `iterations` |
| `on_user_injection`| When a user message is injected via `Agent.inject_user_message` | `message`, `messages` |

### The on_finish event

Fires when the loop terminates, which happens when the model produces a response with text content and no tool calls (a clean termination — the assistant text is the final answer) or when an unrecoverable provider termination is reached (`max_tokens` / `length` / `refusal` / `content_filter` / `incomplete:content_filter` / `SAFETY` / `failed` / `cancelled`). The `content` kwarg is the final answer — the assistant text from the terminating response, or the response text as a fallback when the loop ended via a provider termination. See [Agent](agent.md#tracking-finish-events-with-hooks) for usage.

## Registering a hook

Pass the event name and a callable:

```python
import coreouto as co

def log_llm_call(messages, model, tools, **kwargs):
    print(f"Calling {model} with {len(messages)} messages")

co.register_hook("before_llm_call", log_llm_call)
```

Hooks can be sync or async:

```python
async def async_log(messages, model, tools, **kwargs):
    # e.g., write to an async logging service
    print(f"Calling {model}")

co.register_hook("before_llm_call", async_log)
```

Hooks should accept `**kwargs` to stay forward-compatible if new arguments are added.

## Ordering

Hooks fire in the order they were registered. Register your logging hook first and your metrics hook second, and logging happens first:

```python
co.register_hook("after_llm_call", log_response)    # fires first
co.register_hook("after_llm_call", track_metrics)    # fires second
```

## Inspecting and clearing hooks

```python
hooks = co.get_hooks("before_llm_call")  # => [log_llm_call, async_log]

co.clear_hooks("after_llm_call")   # clear one event
co.clear_hooks()                    # clear all events
```

## Contrib hooks

`coreouto.contrib.hooks` ships five ready-made hook factories. They're opt-in: import and register the ones you need.

### Token collection

Collects token usage from every LLM call into a list:

```python
from coreouto.contrib.hooks import token_collection_hook

hook, usage_log = token_collection_hook()
co.register_hook("after_llm_call", hook)

# After the agent runs:
# usage_log => [Usage(prompt_tokens=..., ...), ...]
```

### Auto-summarize

Summarizes the message history when total tokens exceed a threshold. You provide the summarization function:

```python
from coreouto.contrib.hooks import auto_summarize_hook

def my_summarizer(messages):
    # Return a shorter list of messages
    return messages[:1] + messages[-2:]

hook = auto_summarize_hook(threshold=100_000, summarize_fn=my_summarizer)
co.register_hook("on_iteration", hook)
```

### Token limit warning

Prints (or calls a function) when a single response exceeds a token limit:

```python
from coreouto.contrib.hooks import token_limit_warning_hook

hook = token_limit_warning_hook(limit=8000)
co.register_hook("after_llm_call", hook)
# Prints: "WARNING: token limit 8000 exceeded, current 12500"
```

With a custom callback:

```python
def on_limit(usage):
    raise RuntimeError(f"Token limit exceeded: {usage.total_tokens}")

hook = token_limit_warning_hook(limit=8000, callback=on_limit)
co.register_hook("after_llm_call", hook)
```

### Iteration notification

Prints (or calls a function) every N iterations:

```python
from coreouto.contrib.hooks import iteration_notification_hook

hook = iteration_notification_hook(every=10)
co.register_hook("on_iteration", hook)
# Prints: "INFO: reached iteration 10"
```

### Tool usage collection

Records every tool call with its result:

```python
from coreouto.contrib.hooks import tool_usage_collection_hook

hook, log = tool_usage_collection_hook()
co.register_hook("after_tool_call", hook)

# After the agent runs:
# log => [("search", "Results for: ...", False), ...]
```

Each entry is a tuple of `(tool_name, result_content, is_error)`.

## Writing your own hooks

A hook is any callable that accepts the event's keyword arguments (plus `**kwargs` for safety):

```python
def my_hook(**kwargs):
    # kwargs depend on the event
    # e.g., for "after_tool_call": name, result
    # e.g., for "on_iteration": iteration, messages, response
    pass
```

For stateful hooks, use a closure or a class:

```python
def make_counter():
    count = 0
    def hook(**kwargs):
        nonlocal count
        count += 1
        if count % 5 == 0:
            print(f"Agent has made {count} LLM calls")
    return hook

co.register_hook("after_llm_call", make_counter())
```

Or with a class:

```python
class TimingHook:
    def __init__(self):
        self.call_count = 0

    def __call__(self, **kwargs):
        self.call_count += 1

timer = TimingHook()
co.register_hook("after_llm_call", timer)
# After agent runs: timer.call_count => number of LLM calls
```

## Hook constants

The event name strings are also exported as constants for convenience:

```python
co.BEFORE_LLM_CALL   # "before_llm_call"
co.AFTER_LLM_CALL    # "after_llm_call"
co.BEFORE_TOOL_CALL  # "before_tool_call"
co.AFTER_TOOL_CALL   # "after_tool_call"
co.ON_ITERATION      # "on_iteration"
co.ON_FINISH         # "on_finish"
co.ON_USER_INJECTION # "on_user_injection"
```

Use whichever form you prefer. The string literals are more readable in examples; the constants protect against typos.
