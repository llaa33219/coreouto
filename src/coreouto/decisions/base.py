"""Decision-model abstraction.

A decision model does not generate text. Given a `state` and a set of typed
`questions`, it returns typed answers with calibrated probabilities. coreouto
uses this to drive the agent loop by *deciding* — whether the goal is
complete, which tool to call next, which enumerated argument value to pick —
instead of writing prose.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from coreouto._types import Usage


class NoulQuestion(BaseModel):
    """A yes/no question, answered as a probability of "yes" (0.0-1.0)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["noul"] = "noul"
    instructions: str


class ChoiceQuestion(BaseModel):
    """Pick exactly one option. `criteria` maps each option to a description."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str]


class ScoreQuestion(BaseModel):
    """Rate on an ordered scale. `criteria` lists the levels, low to high."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str]


Question = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion,
    Field(discriminator="type"),
]

# An answer format declares which typed questions the final answer must
# answer. Passed to `Agent.call(answer_format=...)`.
AnswerFormat = dict[str, Question]


class NoulAnswer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = None


class ScoreAnswer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["score"] = "score"
    score: float
    legend: dict[str, Any] = Field(default_factory=dict)
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = None


Answer = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer,
    Field(discriminator="type"),
]


class DecisionResult(BaseModel):
    """The outcome of one `DecisionModel.decide()` call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    answers: dict[str, Answer]
    model: str | None = None
    usage: Usage | None = None
    raw: Any | None = None

    def answers_json(self) -> str:
        """Serialize the answers to a JSON object string (keys preserved)."""
        import json

        return json.dumps(
            {key: answer.model_dump(mode="json") for key, answer in self.answers.items()},
            ensure_ascii=False,
        )


@runtime_checkable
class DecisionModel(Protocol):
    """A decision model: one method, no text generation, no tool calls.

    Implementations wrap a vendor SDK (see `coreouto.decisions.typesafe`) and
    are registered by name with `register_decision_model`. To drive an agent
    loop with one, wrap it in `coreouto.decisions.bridge.DecisionProvider`.
    """

    async def decide(
        self,
        state: Any,
        *,
        questions: dict[str, Question],
        model: str,
        **kwargs: Any,
    ) -> DecisionResult: ...
