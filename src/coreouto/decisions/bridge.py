"""DecisionProvider — drives the agent loop with a decision model.

Wraps a `DecisionModel` in the Provider protocol so `Agent` can run on it.
Each loop iteration asks the model whether the goal is complete and, if not,
which tool to call next. Tools are selected, never authored: every tool
parameter must be enumerable (JSON-schema `enum`) so it can be posed as a
choice question. Tools requiring free-form input are rejected with a clear
error.
"""

from __future__ import annotations

import inspect
import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from coreouto._types import (
    AudioBlock,
    ImageBlock,
    LLMResponse,
    Message,
    TextBlock,
    ToolCall,
    ToolResult,
    VideoBlock,
)
from coreouto.decisions import get_decision_model
from coreouto.decisions.base import (
    AnswerFormat,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionModel,
    MediaBlock,
    NoulAnswer,
    NoulQuestion,
    Question,
    TurnBlock,
)
from coreouto.hooks import BEFORE_TURN_INPUT, ON_GOAL_COMPLETE, ON_TURN_INPUT, trigger
from coreouto.tools import Tool

# Called once per loop iteration with the current transcript; returns this
# turn's extra input — TextBlocks (merged into the state's "context") and/or
# media blocks (sent as `media`, e.g. a fresh screenshot) — or None.
TurnInput = Callable[
    [list[Message]],
    list[TurnBlock] | Awaitable[list[TurnBlock] | None] | None,
]

GOAL_COMPLETE_KEY = "goal_complete"
NEXT_TOOL_KEY = "next_tool"
_ARG_KEY_PREFIX = "arg__"

# Noul answers are probabilities of "yes"; at or above this threshold the
# model's declaration counts as "goal complete".
GOAL_COMPLETE_THRESHOLD = 0.5

_GOAL_COMPLETE_INSTRUCTIONS = (
    "Has the goal been fully achieved? Consider the goal, the full history of "
    "tool calls and their results, and the current situation. Answer yes only "
    "if no further tool calls are needed to satisfy the goal."
)

_NEXT_TOOL_INSTRUCTIONS = (
    "Which tool should be called next to make progress toward the goal? "
    "Pick the single most useful tool given the history so far."
)


def _arg_key(tool_name: str, param_name: str) -> str:
    return f"{_ARG_KEY_PREFIX}{tool_name}__{param_name}"


def _build_state(messages: list[Message], system_prompt: str | None) -> dict[str, Any]:
    """Serialize the transcript into a JSON-safe state for the decision model.

    - goal: the system prompt plus the first user message (the request).
    - history: one entry per prior step — user injections, tool selections
      with their arguments, and tool results.
    - current: the most recent tool results awaiting consideration.
    - context: per-turn text from `turn_input`, when provided (added by
      `create`, not here).
    """
    goal_parts: list[str] = []
    if system_prompt:
        goal_parts.append(f"system: {system_prompt}")
    history: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    goal_captured = False

    for m in messages:
        if m.role == "system":
            if not system_prompt and isinstance(m.content, str):
                goal_parts.append(f"system: {m.content}")
        elif m.role == "user":
            text = m.content if isinstance(m.content, str) else str(m.content)
            if not goal_captured:
                goal_captured = True
                goal_parts.append(f"user: {text}")
            else:
                history.append({"role": "user", "content": text})
        elif m.role == "assistant":
            for tc in m.tool_calls or []:
                history.append(
                    {"decision": "call_tool", "tool": tc.name, "arguments": tc.arguments}
                )
        elif m.role == "tool":
            entry = {"tool": m.name, "result": m.content}
            history.append({"tool_result": entry})
            current = [entry]

    return {"goal": "\n".join(goal_parts), "history": history, "current": current}


def _validate_tools(tools: list[Tool]) -> None:
    """Reject tools the decision model cannot drive: any required parameter
    without an `enum` demands free-form input the model cannot produce.
    Optional non-enum parameters are omitted (their defaults apply).
    """
    for tool in tools:
        schema = tool.parameters or {}
        properties: dict[str, Any] = schema.get("properties", {})
        required: set[str] = set(schema.get("required", []))
        for param_name, param_schema in properties.items():
            if "enum" not in param_schema and param_name in required:
                raise ValueError(
                    f"tool {tool.name!r} is incompatible with decision providers: "
                    f"parameter {param_name!r} requires free-form input (no enum). "
                    f"Decision models can only select among enumerated options — "
                    f"give the parameter an enum (e.g. a Literal type) or a default."
                )


def _enum_choice_question(
    tool: Tool, param_name: str, param_schema: dict[str, Any]
) -> ChoiceQuestion:
    options = [str(v) for v in param_schema["enum"]]
    return ChoiceQuestion(
        instructions=(
            f"What value should the parameter {param_name!r} of tool {tool.name!r} "
            f"take, given the goal and history?"
        ),
        criteria={opt: opt for opt in options},
    )


def _coerce_enum_value(selected: str, param_schema: dict[str, Any]) -> Any:
    for value in param_schema["enum"]:
        if str(value) == selected:
            return value
    return selected


class DecisionProvider:
    """Provider protocol implementation backed by a DecisionModel.

    Loop semantics per iteration:
    - With tools: ask `goal_complete` (noul) + `next_tool` (choice) + one
      choice question per required enum parameter of every tool. If the goal
      is declared complete and an `answer_format` was requested, run one more
      decide() with the answer questions against the full history and return
      its result as `LLMResponse.answer` (content stays empty — nothing is
      synthesized). Otherwise emit the chosen tool call.
    - Without tools (single-answer mode): skip goal detection entirely and
      answer the `answer_format` questions immediately.
    """

    supports_answer_format = True

    def __init__(
        self,
        decision_model: DecisionModel | str,
        *,
        error_handling: list | None = None,
        turn_input: TurnInput | None = None,
    ) -> None:
        if isinstance(decision_model, str):
            decision_model = get_decision_model(decision_model)
        self._decision_model = decision_model
        self.error_handling = error_handling
        self._turn_input = turn_input

    async def create(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        answer_format: AnswerFormat | None = kwargs.pop("answer_format", None)
        tools = list(tools or [])
        _validate_tools(tools)
        state = _build_state(messages, system_prompt)
        context, media = await self._collect_turn_input(messages)
        if context is not None:
            state["context"] = context

        if not tools:
            # Single-answer mode: no goal detection, answer immediately.
            if answer_format is None:
                raise ValueError(
                    "a decision provider with no tools can only answer typed "
                    "questions — pass answer_format to Agent.call()."
                )
            result = await self._decide(state, answer_format, model, media, kwargs)
            return LLMResponse(
                content=None,
                tool_calls=[],
                stop_reason="finish",
                usage=result.usage,
                answer=result,
                raw=result,
            )

        questions = self._iteration_questions(tools)
        result = await self._decide(state, questions, model, media, kwargs)

        if self._goal_complete(result):
            await trigger(ON_GOAL_COMPLETE, state=state, result=result, model=model)
            if answer_format is not None:
                final = await self._decide(state, answer_format, model, media, kwargs)
                return LLMResponse(
                    content=None,
                    tool_calls=[],
                    stop_reason="finish",
                    usage=final.usage,
                    answer=final,
                    raw=final,
                )
            return LLMResponse(
                content=None,
                tool_calls=[],
                stop_reason="finish",
                usage=result.usage,
                raw=result,
            )

        tool_call = self._extract_tool_call(result, tools)
        return LLMResponse(
            content=None,
            tool_calls=[tool_call],
            stop_reason=None,
            usage=result.usage,
            raw=result,
        )

    async def _collect_turn_input(
        self, messages: list[Message]
    ) -> tuple[str | None, list[MediaBlock] | None]:
        if self._turn_input is None:
            return None, None
        await trigger(BEFORE_TURN_INPUT, messages=messages)
        blocks = self._turn_input(messages)
        if inspect.isawaitable(blocks):
            blocks = await blocks
        # ON_TURN_INPUT hooks receive the live list and may mutate it in
        # place (filter, replace, annotate) before the blocks are routed.
        blocks = list(blocks or [])
        await trigger(ON_TURN_INPUT, blocks=blocks, messages=messages)
        text_parts: list[str] = []
        media: list[MediaBlock] = []
        for block in blocks:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, (ImageBlock, AudioBlock, VideoBlock)):
                media.append(block)
            else:
                raise ValueError(
                    f"turn_input cannot return {type(block).__name__}: text info "
                    "belongs in TextBlock (added to the state), media in "
                    "ImageBlock/AudioBlock/VideoBlock (sent as media)."
                )
        return ("\n".join(text_parts) or None), (media or None)

    async def _decide(
        self,
        state: dict[str, Any],
        questions: dict[str, Question],
        model: str,
        media: list[MediaBlock] | None,
        kwargs: dict[str, Any],
    ) -> Any:
        return await self._decision_model.decide(
            state, questions=questions, model=model, media=media, **kwargs
        )

    def _iteration_questions(self, tools: list[Tool]) -> dict[str, Question]:
        questions: dict[str, Question] = {
            GOAL_COMPLETE_KEY: NoulQuestion(instructions=_GOAL_COMPLETE_INSTRUCTIONS),
            NEXT_TOOL_KEY: ChoiceQuestion(
                instructions=_NEXT_TOOL_INSTRUCTIONS,
                criteria={t.name: t.description for t in tools},
            ),
        }
        for tool in tools:
            schema = tool.parameters or {}
            required: set[str] = set(schema.get("required", []))
            for param_name, param_schema in schema.get("properties", {}).items():
                if "enum" in param_schema and param_name in required:
                    questions[_arg_key(tool.name, param_name)] = _enum_choice_question(
                        tool, param_name, param_schema
                    )
        return questions

    @staticmethod
    def _goal_complete(result: Any) -> bool:
        answer = result.answers.get(GOAL_COMPLETE_KEY)
        return isinstance(answer, NoulAnswer) and answer.noul >= GOAL_COMPLETE_THRESHOLD

    @staticmethod
    def _extract_tool_call(result: Any, tools: list[Tool]) -> ToolCall:
        answer = result.answers.get(NEXT_TOOL_KEY)
        if not isinstance(answer, ChoiceAnswer):
            raise ValueError(
                f"decision model did not answer {NEXT_TOOL_KEY!r} as a choice; got: {answer!r}"
            )
        chosen = answer.choice
        tool_by_name = {t.name: t for t in tools}
        if chosen not in tool_by_name:
            raise ValueError(
                f"decision model selected unknown tool {chosen!r}. "
                f"available: {sorted(tool_by_name)}"
            )
        tool = tool_by_name[chosen]
        schema = tool.parameters or {}
        required: set[str] = set(schema.get("required", []))
        arguments: dict[str, Any] = {}
        for param_name, param_schema in schema.get("properties", {}).items():
            if "enum" in param_schema and param_name in required:
                arg_answer = result.answers.get(_arg_key(tool.name, param_name))
                if isinstance(arg_answer, ChoiceAnswer):
                    arguments[param_name] = _coerce_enum_value(arg_answer.choice, param_schema)
        return ToolCall(
            id=f"decision_{uuid.uuid4().hex[:12]}",
            name=chosen,
            arguments=arguments,
        )

    def format_assistant_message(self, response: LLMResponse) -> Message:
        return Message(
            role="assistant",
            content=response.content or "",
            tool_calls=list(response.tool_calls) if response.tool_calls else None,
        )

    def format_tool_result(self, tool_call: ToolCall, result: ToolResult) -> Message:
        content = (
            result.content
            if result.content is not None
            else json.dumps(
                [b.model_dump(mode="json") for b in result.blocks or []],
                ensure_ascii=False,
                default=str,
            )
        )
        return Message(
            role="tool",
            tool_call_id=tool_call.id,
            content=content,
            name=tool_call.name,
        )
