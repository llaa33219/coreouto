# System prompts

The system prompt is the first message the model sees in every `call()` / `call_sync()`. It defines the agent's identity and the meaning of the two reserved tools: `finish` (end the loop) and `continue_loop` (share progress without ending). coreouto ships a default system prompt that gets injected automatically when you don't provide one — read this page before writing your own.

## The default system prompt

If `AgentConfig.system_prompt` is `None` (or an empty string), `Agent.call()` prepends this message to the conversation:

```text
You are an agent. Use tools to gather information as needed.

The `finish` tool ends the agent loop and returns its `content` argument
to the caller as the final answer. Call `finish` when you are done. A
text-only turn without `finish` will be re-prompted, so you must call
`finish` to actually close the loop.

The `continue_loop` tool sends text to the user mid-task but keeps the
loop running for more tools. Use it for progress updates before calling
more tools or before calling `finish`.
```

The text lives in `coreouto.agent._DEFAULT_SYSTEM_PROMPT` (`src/coreouto/agent.py`). The model-driven termination policy means the model MUST call `finish` to close the loop — a text-only response is re-prompted. The default prompt states this explicitly so the model doesn't have to guess.

## When it applies

The default is selected by the branch in `Agent.call()`:

```python
if cfg.system_prompt:
    messages.append(Message(role="system", content=cfg.system_prompt))
else:
    messages.append(Message(role="system", content=_DEFAULT_SYSTEM_PROMPT))
```

Conditions that trigger the default:

- `cfg.system_prompt is None`
- `cfg.system_prompt == ""` (falsy)

The system message is always the first message in the list, followed by `history` (if provided), followed by the new `user_message`. See [Agent — Conversation history](agent.md#conversation-history).

## Why this matters

The default tells the model the two things it needs to know to interact with the loop correctly:

1. **`finish` is the only way to end the loop.** Without this hint, the model may produce a text-only response expecting that to terminate the agent — under coreouto's model-driven policy, that response is re-prompted and the loop never closes (until `max_iterations` triggers `MaxIterationsError`).
2. **`continue_loop` exists for mid-loop progress.** Without it, the only way to share progress is to call `finish` — which forces a one-shot call pattern.

## Writing your own

When you set `system_prompt=...` on an `AgentConfig` or preset, you fully replace the default. At minimum, your prompt must tell the model about the `finish` tool — otherwise the model will never know how to close the loop. Mentioning `continue_loop` is optional; mention it if you want the model to share progress text without ending the loop.

### Minimal replacement

The shortest prompt that still produces a working agent:

```python
system_prompt = (
    "You are a research assistant.\n\n"
    "Use tools to gather information. When you are done, call the `finish` "
    "tool with your final answer as the `content` argument. Use the "
    "`continue_loop` tool to share progress without ending the loop."
)
```

### Common mistakes

- **Forgetting `finish`.** The model never closes the loop. The agent hits `max_iterations` and raises `MaxIterationsError` — the only signal is in the error message, not in `Response.content`.
- **Forgetting `continue_loop` exists.** Without it, the only way to share progress mid-loop is to call `finish`, which forces you into a one-shot call.
- **Conflicting with the tool list.** "Never use the search tool" + a `search` tool in the config confuses the model. The system prompt should be consistent with the tools you register.
- **Contradicting the user's task.** "You must always respond in English" + a user message in Korean leads to unpredictable behavior. Either translate at the boundary or remove the constraint.

### Verifying your prompt

The cheapest sanity check is the mock-provider pattern from the [Quickstart](quickstart.md). With a queued `finish` tool call in the final response, you can confirm that the agent:

- Calls tools when it should.
- Calls `continue_loop` for progress text.
- Calls `finish` on its final turn so the loop ends and the `finish` `content` becomes the final answer.

For behavioral assertions in tests, use the `before_llm_call` and `after_llm_call` hooks to inspect the message list. See [Hooks](hooks.md).

## How it reaches the model

The system message is delivered through the same `messages: list[Message]` channel as user and assistant turns. `Agent.call()` passes the full list to `provider.create(...)`. The `system_prompt` keyword on `Provider.create` is **not** used by the core loop — `Agent.call()` always passes `system_prompt=None` to the provider. The provider is expected to read the system role from `messages` and translate it to whatever the underlying SDK needs (e.g. OpenAI's `instructions` parameter, Anthropic's top-level `system` field, Google's system instruction).

If you write a custom provider, do not rely on the `system_prompt` kwarg of `create()` — pull it from `messages` instead:

```python
async def create(self, messages, *, model, tools=None, system_prompt=None, **kwargs):
    system = next((m.content for m in messages if m.role == "system"), None)
    # pass `system` to your SDK however it expects to receive a system message
    ...
```

See [Providers — Writing a custom provider](providers.md#writing-a-custom-provider).

## See also

- [Agent — `AgentConfig` fields](agent.md#agentconfig-fields) — the `system_prompt` field
- [Presets — `system_prompt` parameter](presets.md#registering-a-preset) — registering presets with a system prompt
- [Multi-agent — Message isolation](multi-agent.md#message-isolation) — child agents each start with their own system prompt
- [Hooks — `on_user_injection`](hooks.md#the-seven-events) — injecting user messages into a running loop
