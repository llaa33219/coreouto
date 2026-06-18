# Philosophy

coreouto is built on five principles. These aren't aspirational statements. They're constraints that shape every design decision. This document explains each one and, just as importantly, what coreouto chooses to leave out.

## Minimalism

An agent library needs to do exactly one thing well: run a loop where an LLM calls tools and eventually returns a result. coreouto's core is that loop, the termination rule that ends it (the model calls the `finish` tool to close the loop; see [How the loop works](agent.md#how-the-loop-works)), the `continue_loop` tool that lets the model emit text without ending the loop, and the data structures that flow through it. That's it.

Minimalism means the whole library fits in your head. You can read `agent.py` in ten minutes and understand every line. There's no framework magic, no hidden state machines, no configuration hierarchies. When something goes wrong, you can trace the execution path without grep-searching through layers of abstraction.

This also means coreouto doesn't try to be a framework. It doesn't have a CLI, a server mode, a deployment pipeline, or a plugin system. Those are all useful things, but they're separate concerns. coreouto is a library: you import it, you call it, it returns.

The practical benefit: when you need to do something unusual, you can. There's no framework fighting you. You can inspect the message list, override the config per-call, swap the provider mid-execution, or write a hook that does something nobody anticipated. The small core means the surface area for "things that could break" is small too.

## Extensibility

Every major component in coreouto is replaceable. Providers, tools, presets, and hooks all follow simple interfaces that you can implement yourself.

**Providers** follow a 3-method protocol: `create`, `format_assistant_message`, `format_tool_result`. No base class required. If your object has those methods with the right signatures, it works. This means you can wrap any LLM API, local model, or even a human-in-the-loop system as a provider.

**Tools** are just functions with type hints. The decorator inspects the hints and builds JSON Schema. If you need more control, `register_tool_class` lets you register a method from any object. Tools are stateless by default but can hold state through closures or class instances.

**Presets** are data. An `AgentPreset` is a Pydantic model you can serialize, store, load from a database, or generate programmatically. The registry is a plain dict. You can clear it, replace it, or inspect it at runtime.

**Hooks** are callables registered to named events. Sync or async, class or function, closure or lambda. The hook system doesn't care. It calls them in order with the event's keyword arguments.

The result: you can build almost anything on top of coreouto without forking it. Need a custom LLM backend? Write a provider. Need to log every tool call? Write a hook. Need to compose agents? Use `agent_as_tool` or build your own orchestration. The library stays small while your application does whatever it needs.

## Explicitness

Nothing happens in coreouto unless you made it happen.

If an agent has access to a tool, you listed it in the config. If a hook fires, you registered it. If a provider is used, you named it. There's no auto-discovery, no scanning for decorated functions, no implicit wiring.

This matters because agent systems are already hard to reason about. The LLM is non-deterministic. Tools can have side effects. The loop can run for dozens of iterations. Adding implicit behavior on top of that makes debugging a nightmare.

Explicitness shows up in small ways too. `agent_as_tool` returns a tool but doesn't register it. You decide if and how to wire it in. `register_hook` doesn't return a deregistration function; you call `clear_hooks` when you want to remove hooks. The system does what you ask, when you ask, and not before.

The trade-off is that you write a few more lines of setup code. That's intentional. A five-line setup that you control is better than a one-line setup that does three things you didn't ask for.

## Fragmentation

Each piece of coreouto works independently. You can use the tool system without presets. You can use presets without hooks. You can use hooks without multi-agent. Changing one piece doesn't break another.

This shows up in the module structure: `agent.py`, `tools.py`, `presets.py`, `hooks.py`, `multi_agent.py`, `providers/`. Each has its own registry (a module-level dict) and its own set of functions. They connect through the `Agent` class, which pulls from each registry as needed, but the registries don't know about each other.

Fragmentation means you can adopt coreouto incrementally. Start with just a provider and an agent. Add tools when you need them. Add hooks for observability later. Add multi-agent when your architecture calls for it. No piece requires another to function (except that agents need a provider).

It also means you can replace pieces independently. Switch from OpenAI to Anthropic, or any other provider, by changing one registration call. Swap your hook implementation without touching your tools. The blast radius of any change is small.

## Conciseness

Code using coreouto should be short and readable. Five lines to define a tool. Three lines to register a preset. One line to call the agent. No boilerplate, no ceremony.

Conciseness isn't about being clever. It's about removing the parts that don't carry meaning. A `@register_tool` decorator that extracts type hints means you don't write a separate JSON Schema. A `to_config()` method on presets means you don't construct `AgentConfig` by hand. A `call_sync()` wrapper means you don't write `asyncio.run()` every time.

The goal: someone reading your code should understand what the agent does within a few seconds. The system prompt says what it is. The tool list says what it can do. The call sends a message and gets a response. No hidden layers between the reader and the logic.

## What we intentionally do NOT include

These features are common in other agent frameworks. coreouto leaves them out on purpose.

### Auto-summarization

Some frameworks automatically summarize the conversation when it gets too long. coreouto doesn't. Summarization is a lossy operation with domain-specific trade-offs. When to summarize, how to summarize, and what to preserve are decisions only you can make.

Instead, coreouto gives you the `auto_summarize_hook` in `contrib/hooks.py`. You provide the summarization function and the threshold. The hook fires at the right time and calls your function. You control the behavior; coreouto controls the timing.

### Agent-to-agent communication

Some frameworks let agents message each other directly, forming graphs or networks. coreouto doesn't. Agent-to-agent communication is an architecture choice, not a primitive.

`agent_as_tool` gives you a clean delegation pattern: parent calls child, child returns a result. If you need something more complex (parallel agents, shared state, message passing), you build it on top of the primitives. coreouto doesn't dictate the topology.

### Automatic retries

If the LLM returns a malformed response or a tool fails, coreouto doesn't retry automatically. The error goes back to the LLM as a tool result, and the LLM decides what to do. If you want retries with backoff, you write a hook or wrap the provider.

Automatic retries hide problems. If your tool is failing intermittently, you want to know about it, not have the system silently retry until it works.

### Built-in RAG, memory, or vector stores

Retrieval-augmented generation, long-term memory, and vector databases are application-level concerns. They belong in your tools, not in the core library. coreouto gives you the tool system to build whatever retrieval or memory mechanism your application needs.

### Streaming

The core loop collects the full LLM response before processing it. Streaming is a UI concern. If you need to stream tokens to a client, you can do it at the provider level or through hooks. The core loop doesn't need to know about it.

### Configuration files

coreouto has no YAML, TOML, or JSON config format. Everything is Python code. You register providers, tools, presets, and hooks in Python. This keeps the system explicit, debuggable, and composable. Configuration files add a layer of indirection that makes it harder to trace what's happening.

### Observability built in

coreouto doesn't ship with logging, tracing, or metrics. It ships with hooks. The `contrib/hooks.py` module gives you starting points (token collection, tool usage logging), but the actual observability stack is your choice. Want OpenTelemetry? Write a hook. Want to log to a file? Write a hook. The hook system is the extension point; the implementation is yours.

## Summary

coreouto is small on purpose. It gives you a loop, tools, providers, presets, and hooks. Everything else is an application-level concern that you build on top of these primitives. The library stays out of your way so you can focus on what your agent actually needs to do.
