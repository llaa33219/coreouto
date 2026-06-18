<p align="center">
  <img src="./logo.svg" alt="coreouto" width="800">
</p>

# coreouto

**A minimal, extensible agent library for Python.**

Built on five philosophies: **minimalism, extensibility, explicitness, fragmentation, conciseness.**

The whole library reduces to one idea: an agent is called with a message, runs an internal loop, and returns its response when the provider's end-of-turn signal classifies as END — the text of that turn becomes the final answer. The exact field and value depend on the provider: Anthropic's `stop_reason` (`end_turn`, `max_tokens`, `stop_sequence`, `refusal`), OpenAI Chat Completions' `finish_reason` (`stop`, `length`, `content_filter`), OpenAI Responses' `status` (`failed`, `cancelled`, `incomplete:content_filter`), Google Gemini's `finishReason` (`MAX_TOKENS`, `SAFETY`, ...). To output text without ending the loop (e.g. share progress before calling more tools), the model calls the `continue_loop` tool. Everything else — providers, tools, presets, hooks, multi-agent — is an opt-in extension.

```python
import os

import coreouto as co
from coreouto.providers.openai import OpenAIProvider

@co.register_tool("search")
def search(query: str) -> str:
    """Search the web for `query`."""
    return f"<results for {query}>"

co.register_provider("minimax", OpenAIProvider(
    api_key=os.environ["MINIMAX_API_KEY"],
    base_url="https://api.minimax.io/v1",
))

preset = co.register_agent_preset(
    "researcher", model="MiniMax-M3", provider="minimax",
    system_prompt="You are a research assistant.",
    tools=["search"],
)
response = co.Agent(preset.to_config()).call_sync("Find me recent news about fusion energy.")
print(response.content)
```

## Install

```bash
pip install coreouto
# with providers
pip install coreouto[openai]
pip install coreouto[anthropic]
pip install coreouto[google]
pip install coreouto[all]
```

## Documentation

See [`docs/`](./docs/) for the full documentation set:

- [Philosophy](./docs/philosophy.md) — the five principles
- [Quickstart](./docs/quickstart.md)
- [Agent](./docs/agent.md)
- [Providers](./docs/providers.md)
- [Tools](./docs/tools.md)
- [Presets](./docs/presets.md)
- [Hooks](./docs/hooks.md)
- [Multi-agent](./docs/multi-agent.md)
- [System prompts](./docs/prompts.md) — the default prompt and how to write your own

## License

Apache 2.0 — see [LICENSE](./LICENSE).
