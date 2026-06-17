# System prompts

The system prompt is the first message the model sees in every `call()` / `call_sync()`. It defines the agent's identity, the loop's termination rules, and the meaning of the built-in `continue_loop` tool. coreouto ships a default system prompt that gets injected automatically when you don't provide one — read this page before writing your own.

## The default system prompt

If `AgentConfig.system_prompt` is `None` (or an empty string), `Agent.call()` prepends this message to the conversation:

```text
You are an agent. Use tools to gather information as needed.

The loop ends when your turn has NO tool calls — that text becomes the final
answer returned to the user. To finish, respond with text only.

The `continue_loop` tool does the opposite: it sends text to the user mid-task
but keeps the loop running for more tools. Use it for progress updates before
calling more tools.

Example:
  User: Find news and summarize.
  You: [search]                          -> loop continues
  You: continue_loop(content="Found 3 articles.")  -> text to user, loop continues
  You: [fetch article]                   -> loop continues
  You: Here is a summary.                -> no tool call -> loop ends, returned to user
```

The text lives in `coreouto.agent._DEFAULT_SYSTEM_PROMPT` (`src/coreouto/agent.py`). The four-paragraph structure is:

1. **Identity.** "You are an agent. Use tools to gather information as needed."
2. **Termination rule.** "The loop ends when your turn has NO tool calls — that text becomes the final answer. To finish, respond with text only."
3. **`continue_loop` explanation.** The inverse of termination: emit text without ending the loop.
4. **Worked example.** A four-step trace showing the difference between `[tool call]`, `continue_loop`, and text-only termination.

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

The default is not cosmetic. The model's first turn decides whether the loop ends or continues. If the model is unaware of the termination rule, it tends to:

- Emit a long text answer on turn 1 — the loop ends before any tools are called.
- Call a tool, then add a chatty paragraph "I searched and found..." — the text becomes the final answer, the tool result is dropped from the loop.

The default prompt prevents both failure modes by stating the rule explicitly and by teaching the `continue_loop` inverse.

## Writing your own

When you set `system_prompt=...` on an `AgentConfig` or preset, you fully replace the default. Your prompt must cover the same ground, or the loop will misbehave. At minimum, it should make the following three things clear:

| Requirement | Why |
|---|---|
| How the loop ends | The model must know "text only" = termination. Otherwise turn 1 will end prematurely. |
| What `continue_loop` does | The tool is always available; if the model doesn't know about it, it can't use it to share progress. |
| That tool results are loop input, not the final answer | A model that treats a tool result as "the answer" will produce text and end the loop right after the first tool call. |

### Minimal replacement

The shortest prompt that still produces a working agent:

```python
system_prompt = (
    "You are a research assistant.\n\n"
    "The loop ends when your turn has no tool calls. To finish, respond with "
    "text only. To send text to the user without ending the loop, call the "
    "`continue_loop` tool. Tool results become input for your next turn; they "
    "are not the final answer."
)
```

### Common mistakes

- **Dropping the termination rule.** "You are a helpful assistant" alone is not enough — the model will treat the first reasonable-looking output as done.
- **Forgetting `continue_loop` exists.** Without it, the only way to share progress mid-loop is to end the loop, which forces you into a one-shot call.
- **Conflicting with the tool list.** "Never use the search tool" + a `search` tool in the config confuses the model. The system prompt should be consistent with the tools you register.
- **Contradicting the user's task.** "You must always respond in English" + a user message in Korean leads to unpredictable behavior. Either translate at the boundary or remove the constraint.

### Verifying your prompt

The cheapest sanity check is the mock-provider pattern from the [Quickstart](quickstart.md). With a controlled response, you can confirm that the agent:

- Calls tools when it should.
- Calls `continue_loop` for progress text.
- Responds with text only when it intends to end the loop.

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
