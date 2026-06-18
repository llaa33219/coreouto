# AGENTS.md

This file guides AI coding agents working on the coreouto codebase.

## What coreouto is

A minimal Python agent library for PyPI. The entire core is one loop:
**call → internal loop → `Response` when the provider's end-of-turn signal classifies as END** (the text of that turn becomes the final answer). The exact field and END values depend on the provider — see `_should_terminate` in `src/coreouto/agent.py`. The rule is "if the value is not END, keep going". To output text without ending the loop, the model calls the `continue_loop` tool.

## The five philosophies (NON-NEGOTIABLE)

1. **Minimalism** — implement only the minimum for an agent system. Let the user extend.
2. **Extensibility** — almost everything must be customizable.
3. **Explicitness** — the user declares everything. No auto-features (no auto-agent-to-agent, no built-in auto-summarization). Built-in hooks live in `coreouto/contrib/hooks.py` and are opt-in.
4. **Fragmentation** — features are broken into independent pieces. One feature changing doesn't break others.
5. **Conciseness** — code using coreouto should be obvious to read.

## Layout

```
src/coreouto/
  __init__.py        # public API
  _types.py          # Pydantic v2 message/response models
  agent.py           # Agent class and the core loop
  tools.py           # @register_tool decorator and Tool class
  presets.py         # agent preset registration
  hooks.py           # hook event constants and ordered async dispatch
  multi_agent.py     # agent_as_tool() helper
  sync.py            # call_sync() — fails loudly if a loop is running
  contrib/
    hooks.py         # 5 opt-in hook recipes
  providers/
    base.py          # Provider protocol (3 methods)
    __init__.py      # string-keyed registry
    openai.py
    anthropic.py
    google.py
    openai_response.py
tests/
docs/
examples/
```

## Build / test commands

- `pip install -e ".[dev,all]"` — editable install
- `pytest -q` — full test suite (uses `MockProvider`, no real API calls)
- `ruff check src tests examples` — lint
- `ruff format --check src tests examples` — format check
- `python -m build` — build wheel + sdist
- `twine check dist/*` — verify metadata

## Critical invariants

- **No real API calls in tests.** Use the `MockProvider` seam from `tests/conftest.py`.
- **Async-first.** `Agent.call()` is `async def`. `call_sync()` raises `RuntimeError` if an event loop is running.
- **No `nest_asyncio` anywhere.**
- **Provider protocol has exactly 3 methods**: `create`, `format_assistant_message`, `format_tool_result`.

## Adding code

- New provider → implement the 3-method protocol and `register_provider(name, instance)`.
- New tool → `@register_tool("name")` over a sync/async callable. Type hints are extracted to JSON Schema.
- New hook event → add a string constant in `hooks.py`; fire via `await trigger(EVENT, ctx)`.
- New built-in hook recipe → add to `coreouto/contrib/hooks.py` as a factory returning a hook function.
