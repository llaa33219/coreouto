# coreouto

A minimal, extensible Python agent library. An agent is called with a message, runs an internal loop where it can use tools, and returns its response when the model calls the built-in `finish` tool — the `content` argument of the `finish` call becomes the final answer. The rule is **model-driven**: the model declares its intent to end the loop through a tool call, not through a provider's natural end-of-turn signal. Unrecoverable provider terminations (token cap, refusal, content filter, server failure) still end the loop. To output text without ending the loop, the model calls the built-in `continue_loop` tool. Everything else is an opt-in extension.

```python
import coreouto as co

@co.register_tool("search")
def search(query: str) -> str:
    """Search the web."""
    return f"Results for {query}"

co.providers.openai.register(api_key="...")

preset = co.register_agent_preset(
    "researcher", model="gpt-5.5", provider="openai",
    system_prompt="You are a research assistant.",
    tools=["search"],
)
# response = await co.Agent(preset.to_config()).call("What's new in fusion energy?")
```

## Install

```bash
pip install coreouto
# with provider extras
pip install coreouto[openai]
pip install coreouto[anthropic]
pip install coreouto[google]
pip install coreouto[all]
```

## The five philosophies

**Minimalism.** coreouto implements only what an agent system needs to function. There is one loop, one termination rule (the model calls `finish` to close the loop; the provider's natural end-of-turn field without `finish` is treated as CONTINUE), and one inverse tool (`continue_loop`) for when the model wants to emit text without ending the loop. If a feature can live outside the library, it does. The core stays small so you can read the whole thing in an afternoon.

**Extensibility.** Providers, tools, presets, and hooks are all open for extension. You can swap the LLM backend, add custom tools, define agent presets, and inject behavior at every stage of the loop. Nothing is locked down.

**Explicitness.** You declare what you want. No hidden auto-features, no magic wiring. If a tool is available to an agent, you listed it. If a hook runs, you registered it. The system does what you tell it and nothing more.

**Fragmentation.** Features are broken into independent pieces. Changing your provider doesn't break your tools. Adding a hook doesn't touch your presets. Each concern lives in its own module and can be understood, tested, and replaced on its own.

**Conciseness.** Code using coreouto should be short and obvious. Five lines to set up an agent. One line to call it. No boilerplate, no ceremony, no configuration files. If you can say it in fewer words, do.

## Documentation

- [Quickstart](quickstart.md) -- end-to-end setup, from install to first call
- [Agent](agent.md) -- the `Agent` class, `call()`, `call_sync()`, `Response`
- [Providers](providers.md) -- built-in providers, custom providers, the 3-method protocol
- [Tools](tools.md) -- `@register_tool`, type hints to JSON Schema, class-based tools
- [Presets](presets.md) -- named agent configurations, `register_agent_preset`, `to_config()`
- [Hooks](hooks.md) -- the 7 hook events, ordering, contrib hooks, writing your own
- [Multi-agent](multi-agent.md) -- `agent_as_tool`, delegation patterns, message isolation
- [System prompts](prompts.md) -- the default system prompt, writing your own, provider interaction
- [Philosophy](philosophy.md) -- deep dive on the five principles and what's intentionally absent

## License

MIT
