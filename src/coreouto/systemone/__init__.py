"""String-keyed System One model registry.

Maps names to SystemOneModel instances (duck-typed, no isinstance check),
mirroring the provider registry.
"""

from __future__ import annotations

from coreouto.systemone.base import (
    Answer,
    AnswerFormat,
    ChoiceAnswer,
    ChoiceQuestion,
    MediaBlock,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneModel,
    SystemOneResult,
    TurnBlock,
)

__all__ = [
    "Answer",
    "AnswerFormat",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "MediaBlock",
    "NoulAnswer",
    "NoulQuestion",
    "Question",
    "ScoreAnswer",
    "ScoreQuestion",
    "SystemOneModel",
    "SystemOneResult",
    "TurnBlock",
    "available_systemone_models",
    "clear_systemone_models",
    "get_systemone_model",
    "register_systemone_model",
]

_SYSTEMONE_MODELS: dict[str, SystemOneModel] = {}


def register_systemone_model(name: str, model: SystemOneModel) -> None:
    _SYSTEMONE_MODELS[name] = model


def get_systemone_model(name: str) -> SystemOneModel:
    try:
        return _SYSTEMONE_MODELS[name]
    except KeyError:
        raise KeyError(
            f"System One model not registered: {name!r}. available: {list(_SYSTEMONE_MODELS)}"
        ) from None


def available_systemone_models() -> list[str]:
    return sorted(_SYSTEMONE_MODELS)


def clear_systemone_models() -> None:
    _SYSTEMONE_MODELS.clear()


def __getattr__(name: str):
    if name in ("typesafe",):
        import importlib

        try:
            return importlib.import_module(f"coreouto.systemone.{name}")
        except Exception as exc:
            raise AttributeError(f"module 'coreouto.systemone' has no attribute '{name}'") from exc
    raise AttributeError(f"module 'coreouto.systemone' has no attribute '{name}'")
