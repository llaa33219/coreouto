# System prompts

The system prompt is the first message the model sees in every `call()` / `call_sync()`. It defines the agent's identity and explains the termination rule: call tools to work, respond with text and no tool calls to end the loop. coreouto ships a default system prompt that gets injected automatically when you don't provide one — read this page before writing your own.

## The default system prompt

If `AgentConfig.system_prompt` is `None` (or an empty string), `Agent.call()` prepends this message to the conversation:

```text
You are an agent. Use tools to gather information and take actions.

The loop runs as long as you call tools. To deliver your final answer and
end the loop, respond with text and no tool calls — that text becomes the
response. To share progress while still working, include the text alongside
a tool call in the same turn.
```

The text lives in `coreouto.agent._DEFAULT_SYSTEM_PROMPT` (`src/coreouto/agent.py`). The termination policy means the model ends the loop by producing a response with text content and no tool calls — the assistant text becomes the final answer. The default prompt states this explicitly so the model doesn't have to guess.

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

1. **A response with text content and no tool calls ends the loop.** Without this hint, the model may keep calling tools indefinitely (hitting `max_iterations`) or may never produce a bare text answer.
2. **Text alongside a tool call shares progress.** The model can include progress text in the same turn as a tool call. That text lives in the assistant message. To surface it to an end user, register an `on_iteration` hook and read `response.content` — see [Agent — Progress mid-task](agent.md#progress-mid-task).

## Writing your own

When you set `system_prompt=...` on an `AgentConfig` or preset, you fully replace the default. At minimum, your prompt must tell the model that a response with text content and no tool calls ends the loop — otherwise the model may keep calling tools forever or produce a bare answer too early.

### Minimal replacement

The shortest prompt that still produces a working agent:

```python
system_prompt = (
    "You are a research assistant.\n\n"
    "Use tools to gather information. When you are done, respond with text "
    "and no tool calls — that text becomes the response. To share progress "
    "while still working, include the text alongside a tool call."
)
```

### Common mistakes

- **No termination hint.** The model keeps calling tools indefinitely and hits `max_iterations`. Your prompt must tell the model that a response with text content and no tool calls ends the loop.
- **Ending too early.** The model produces a bare text answer before using tools. If your task requires tool use, say so explicitly ("use tools to gather information before answering").
- **Conflicting with the tool list.** "Never use the search tool" + a `search` tool in the config confuses the model. The system prompt should be consistent with the tools you register.
- **Contradicting the user's task.** "You must always respond in English" + a user message in Korean leads to unpredictable behavior. Either translate at the boundary or remove the constraint.

### Verifying your prompt

The cheapest sanity check is the mock-provider pattern from the [Quickstart](quickstart.md). With a queued text-only final response, you can confirm that the agent:

- Calls tools when it should.
- Ends with a text-only answer (no tool calls) so the loop terminates and the text becomes the final answer.

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
