"""TypeSafe AI (Jev) System One model, wrapping the official typesafe-sdk.

Install with: pip install coreouto[typesafe]
"""

from __future__ import annotations

import base64
from typing import Any

from coreouto._types import AudioBlock, ImageBlock, Usage, VideoBlock
from coreouto.systemone import register_systemone_model
from coreouto.systemone.base import (
    ChoiceAnswer,
    ChoiceQuestion,
    MediaBlock,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneResult,
)

try:
    from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score
except ImportError:
    AsyncTypeSafeClient = None


def _to_sdk_question(question: Question) -> Any:
    if isinstance(question, NoulQuestion):
        return Noul(instructions=question.instructions)
    if isinstance(question, ChoiceQuestion):
        return Choice(instructions=question.instructions, criteria=question.criteria)
    if isinstance(question, ScoreQuestion):
        return Score(instructions=question.instructions, criteria=question.criteria)
    raise TypeError(f"unsupported question type: {type(question).__name__}")


def _from_sdk_answer(answer: Any) -> NoulAnswer | ChoiceAnswer | ScoreAnswer:
    if answer.type == "noul":
        return NoulAnswer(noul=answer.noul)
    if answer.type == "choice":
        return ChoiceAnswer(
            choice=answer.choice,
            probabilities=dict(answer.probabilities),
            confidence=answer.confidence,
        )
    if answer.type == "score":
        return ScoreAnswer(
            score=answer.score,
            legend={str(k): v for k, v in answer.legend.items()},
            probabilities={str(k): v for k, v in answer.probabilities.items()},
            confidence=answer.confidence,
        )
    raise TypeError(f"unsupported answer type: {answer.type!r}")


def _media_to_extra_body(blocks: list[MediaBlock]) -> dict[str, list[str]]:
    """Encode media blocks as base64 data URIs in the request body fields
    typesafe-compatible multimodal endpoints (e.g. Cloudflare Clef) define:
    `images`, `audio`, `videos` — siblings of `state`, not part of it.
    """
    body: dict[str, list[str]] = {}
    targets = ((ImageBlock, "images"), (AudioBlock, "audio"), (VideoBlock, "videos"))
    for block in blocks:
        if block.url is not None:
            raise ValueError(
                "typesafe-compatible endpoints do not accept remote media URLs — "
                "pass raw bytes instead (e.g. ImageBlock(data=..., mime_type=...))"
            )
        assert block.data is not None and block.mime_type is not None
        data_uri = f"data:{block.mime_type};base64,{base64.b64encode(block.data).decode()}"
        for block_type, field in targets:
            if isinstance(block, block_type):
                body.setdefault(field, []).append(data_uri)
                break
    return body


class TypeSafeSystemOneModel:
    """SystemOneModel backed by TypeSafe AI's System One endpoint (Jev)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        client: AsyncTypeSafeClient | None = None,
        timeout: float | None = None,
    ) -> None:
        # timeout: per-request timeout in seconds, forwarded to the SDK client.
        # Ignored when `client` is provided (configure it on the client itself).
        if client is not None:
            self._client = client
        else:
            if AsyncTypeSafeClient is None:
                raise ImportError(
                    "The typesafe-sdk package is required. "
                    "Install it with: pip install coreouto[typesafe]"
                )
            client_kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
            if timeout is not None:
                client_kwargs["timeout"] = timeout
            self._client = AsyncTypeSafeClient(**client_kwargs)

    async def decide(
        self,
        state: Any,
        *,
        questions: dict[str, Question],
        model: str,
        media: list[MediaBlock] | None = None,
        **kwargs: Any,
    ) -> SystemOneResult:
        sdk_questions = {key: _to_sdk_question(q) for key, q in questions.items()}
        if media:
            extra_body = {**_media_to_extra_body(media), **kwargs.pop("extra_body", {})}
            kwargs["extra_body"] = extra_body
        resp = await self._client.system_one(state, sdk_questions, model=model, **kwargs)

        answers = {key: _from_sdk_answer(a) for key, a in resp.answers.items()}

        usage = None
        if resp.usage is not None:
            prompt = resp.usage.input_tokens or 0
            completion = resp.usage.output_tokens or 0
            usage = Usage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
            )

        return SystemOneResult(answers=answers, model=resp.model, usage=usage, raw=resp)


def register(
    api_key: str | None = None,
    base_url: str | None = None,
    name: str = "typesafe",
    timeout: float | None = None,
) -> None:
    register_systemone_model(
        name, TypeSafeSystemOneModel(api_key=api_key, base_url=base_url, timeout=timeout)
    )
