# Decision models

A decision model does not generate text. Given a `state` and a set of typed `questions`, it returns typed answers with calibrated probabilities — whether something is true, which option to pick, where on a scale something falls. coreouto supports these alongside chat providers, so an agent loop can be *driven by decisions* instead of prose.

The built-in implementation wraps TypeSafe AI's System One endpoint (Jev) via the official `typesafe-sdk`.

```bash
pip install coreouto[typesafe]
```

## The pieces

- **Questions**: `NoulQuestion` (yes/no, answered as a 0.0–1.0 probability), `ChoiceQuestion` (pick one of enumerated options), `ScoreQuestion` (rate on an ordered scale).
- **`DecisionModel`**: a 1-method protocol — `decide(state, *, questions, model) -> DecisionResult`. Register instances with `register_decision_model(name, model)`.
- **`DecisionProvider`**: wraps a decision model in the Provider protocol so `Agent` runs on it. Register it like any provider with `register_provider(name, DecisionProvider(...))`.
- **`answer_format`**: passed to `Agent.call()`, declares the typed questions the final answer must answer.

## How the loop works under a decision provider

Each iteration, `DecisionProvider` builds a state from the transcript — the goal (system prompt + first user message), the full history of tool selections and results, and the current situation — and asks:

- `goal_complete` (noul): has the goal been fully achieved? At probability ≥ 0.5 the loop ends.
- `next_tool` (choice): which of the registered tools to call next.
- one choice question per required `enum` parameter of each tool, so the model also *selects the arguments*.

If the goal is not complete, the chosen tool call is emitted and executed like any provider's tool call. If it is complete:

- without `answer_format`: the loop ends with an empty `Response.content` — nothing is synthesized. The raw `DecisionResult` is on `LLMResponse.raw`.
- with `answer_format`: one more `decide()` runs against the full history with your answer questions, and its `DecisionResult` becomes `Response.answer`.

With **no tools registered**, the agent is in single-answer mode: goal detection is skipped and the `answer_format` questions are answered immediately. (`answer_format` is required in that case — without tools there is nothing else to decide.)

## Example: a decision-driven agent

```python
import coreouto as co
from coreouto.decisions.typesafe import register as register_typesafe

@co.register_tool("restart_service")
def restart_service() -> str:
    """Restart the service."""
    return "restarted"

@co.register_tool("page_oncall")
def page_oncall() -> str:
    """Page the on-call engineer."""
    return "paged"

register_typesafe(api_key="...")                    # registers decision model "typesafe"
co.register_provider("jev", co.DecisionProvider("typesafe"))

preset = co.register_agent_preset(
    "incident", model="jev-latest", provider="jev",
    system_prompt="Resolve the incident with the available tools.",
    tools=["restart_service", "page_oncall"],
)
response = co.Agent(preset.to_config()).call_sync(
    "API latency is 10x normal.",
    answer_format={
        "resolution": co.ChoiceQuestion(
            instructions="How was the incident resolved?",
            criteria={"auto_remediated": "a tool fixed it", "escalated": "humans paged"},
        ),
    },
)
print(response.answer.answers["resolution"].choice)
```

## Tool compatibility

A decision model can only *select* — it cannot author free-form values. Any tool with a required parameter that has no JSON-schema `enum` is rejected with a clear error at call time:

```
tool 'search' is incompatible with decision providers: parameter 'query'
requires free-form input (no enum). Decision models can only select among
enumerated options — give the parameter an enum (e.g. a Literal type) or a
default.
```

Parameters typed as `Literal[...]` map to `enum` automatically (see [Tools](tools.md)), so `def scale(direction: Literal["up", "down"])` works out of the box. Optional non-enum parameters are omitted and their defaults apply.

## Composing with LLM agents

`agent_as_tool` accepts `answer_format`: the sub-agent runs on a decision provider, selects its final answer in the declared format, and the selections are returned to the parent as a JSON tool result.

```python
co.register_agent_preset("triage", model="jev-latest", provider="jev")
triage = co.agent_as_tool(
    "triage",
    answer_format={
        "team": co.ChoiceQuestion(
            instructions="Which team should handle this?",
            criteria={"billing": "payments, refunds", "tech": "bugs, outages"},
        ),
    },
)
# register `triage` under a name and list it in the parent agent's tools
```

`answer_format` is a decision-provider feature. Passing it to a chat provider raises a clear `ValueError`.

## Writing your own decision model

Implement the 1-method protocol and register it:

```python
class MyDecisionModel:
    async def decide(self, state, *, questions, model, **kwargs):
        ...  # return co.DecisionResult(answers={...})

co.register_decision_model("mine", MyDecisionModel())
co.register_provider("mine-loop", co.DecisionProvider("mine"))
```

`decide()` receives JSON-safe `state` (`{"goal", "history", "current"}`) and the question map, and returns a `DecisionResult` whose answers are keyed by question name.
