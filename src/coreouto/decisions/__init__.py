"""String-keyed decision-model registry.

Maps names to DecisionModel instances (duck-typed, no isinstance check),
mirroring the provider registry.
"""

from __future__ import annotations

from coreouto.decisions.base import (
    Answer,
    AnswerFormat,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionModel,
    DecisionResult,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)

__all__ = [
    "Answer",
    "AnswerFormat",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "DecisionModel",
    "DecisionResult",
    "NoulAnswer",
    "NoulQuestion",
    "Question",
    "ScoreAnswer",
    "ScoreQuestion",
    "available_decision_models",
    "clear_decision_models",
    "get_decision_model",
    "register_decision_model",
]

_DECISION_MODELS: dict[str, DecisionModel] = {}


def register_decision_model(name: str, model: DecisionModel) -> None:
    _DECISION_MODELS[name] = model


def get_decision_model(name: str) -> DecisionModel:
    try:
        return _DECISION_MODELS[name]
    except KeyError:
        raise KeyError(
            f"decision model not registered: {name!r}. available: {list(_DECISION_MODELS)}"
        ) from None


def available_decision_models() -> list[str]:
    return sorted(_DECISION_MODELS)


def clear_decision_models() -> None:
    _DECISION_MODELS.clear()


def __getattr__(name: str):
    if name in ("typesafe",):
        import importlib

        try:
            return importlib.import_module(f"coreouto.decisions.{name}")
        except Exception as exc:
            raise AttributeError(f"module 'coreouto.decisions' has no attribute '{name}'") from exc
    raise AttributeError(f"module 'coreouto.decisions' has no attribute '{name}'")
