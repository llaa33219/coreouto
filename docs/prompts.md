# System prompts

The system prompt is the first message the model sees in every `call()` / `call_sync()`. It defines the agent's identity and the meaning of the built-in `continue_loop` tool. coreouto ships a default system prompt that gets injected automatically when you don't provide one — read this page before writing your own.

## The default system prompt

If `AgentConfig.system_prompt` is `None` (or an empty string), `Agent.call()` prepends this message to the conversation:

```text
You are an agent. Use tools to gather information as needed.

The `continue_loop` tool sends text to the user mid-task but keeps the loop
running for more tools. Use it for progress updates before calling more tools.
```

The text lives in `coreouto.agent._DEFAULT_SYSTEM_PROMPT` (`src/coreouto/agent.py`). It is intentionally short: the model does not need to know how the loop terminates — coreouto reads the provider's end-of-turn signal directly, so the model is free to end its turn with text only (and no tool call) whenever it is done.

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

The default is intentionally minimal. coreouto reads the provider's native end-of-turn signal directly (`end_turn` for Anthropic, `stop` for OpenAI Chat, `completed` for OpenAI Responses, `STOP` for Gemini, ...), so the model does not need to be told the rule. All it needs to know is that `continue_loop` exists for mid-loop progress.

## Writing your own

When you set `system_prompt=...` on an `AgentConfig` or preset, you fully replace the default. The only mechanical concern is to make sure the model knows about `continue_loop` if you want it to share progress without ending the loop. Everything else — termination, tool execution, error handling — is coreouto's job, not the prompt's.

### Minimal replacement

The shortest prompt that still produces a working agent:

```python
system_prompt = (
    "You are a research assistant.\n\n"
    "Use tools to gather information. To send text to the user without ending "
    "the loop, call the `continue_loop` tool."
)
```

### Common mistakes

- **Over-explaining termination.** The model does not need to know how coreouto detects the end of the loop; it just needs to respond with text only (and no tool call) when it is done. The provider's end-of-turn signal handles the rest.
- **Forgetting `continue_loop` exists.** Without it, the only way to share progress mid-loop is to end the loop, which forces you into a one-shot call.
- **Conflicting with the tool list.** "Never use the search tool" + a `search` tool in the config confuses the model. The system prompt should be consistent with the tools you register.
- **Contradicting the user's task.** "You must always respond in English" + a user message in Korean leads to unpredictable behavior. Either translate at the boundary or remove the constraint.

### Verifying your prompt

The cheapest sanity check is the mock-provider pattern from the [Quickstart](quickstart.md). Set `stop_reason` to the appropriate provider value on the mock response and confirm that the agent:

- Calls tools when it should.
- Calls `continue_loop` for progress text.
- Responds with text only on its final turn so the provider emits the end-of-turn signal and the loop ends.

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
