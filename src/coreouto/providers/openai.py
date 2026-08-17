from __future__ import annotations

import json
from typing import Any

from coreouto._types import LLMResponse, Message, TextBlock, ToolCall, ToolResult, Usage
from coreouto.providers import register_provider
from coreouto.tools import Tool

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


class OpenAIProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        client: AsyncOpenAI | None = None,
        stream: bool = False,
        error_handling: list | None = None,
        timeout: float | None = None,
    ) -> None:
        # timeout: per-request timeout in seconds, forwarded to the SDK client.
        # Ignored when `client` is provided (configure it on the client itself).
        if client is not None:
            self._client = client
        else:
            if AsyncOpenAI is None:
                raise ImportError(
                    "The openai package is required. Install it with: pip install coreouto[openai]"
                )
            client_kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
            if timeout is not None:
                client_kwargs["timeout"] = timeout
            self._client = AsyncOpenAI(**client_kwargs)
        self._stream = stream
        self.error_handling = error_handling

    async def create(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        msgs: list[dict[str, Any]] = []
        if system_prompt is not None:
            msgs.append({"role": "system", "content": system_prompt})

        for m in messages:
            if m.role == "system":
                msgs.append({"role": "system", "content": m.content})
            elif m.role == "user":
                msgs.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                seen: set[str] = set()
                if isinstance(m.content, str):
                    content = m.content
                else:
                    parts: list[str] = []
                    for item in m.content:
                        if isinstance(item, TextBlock):
                            parts.append(item.text)
                        elif isinstance(item, ToolCall):
                            seen.add(item.id)
                    content = "".join(parts)
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": content,
                }
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in m.tool_calls
                        if tc.id not in seen
                    ]
                msgs.append(msg)
            elif m.role == "tool":
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.tool_call_id,
                        "content": m.content,
                    }
                )

        openai_tools = (
            [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            if tools
            else None
        )

        use_stream = kwargs.pop("stream", self._stream)
        on_stream_text = kwargs.pop("_on_stream_text", None)
        on_stream_thinking = kwargs.pop("_on_stream_thinking", None)
        request_kwargs = dict(
            model=model,
            messages=msgs,
            tools=openai_tools if openai_tools else None,
            **kwargs,
        )

        if use_stream:
            # Low-level create(stream=True), NOT the SDK's .stream() helper.
            # The helper exists for auto-parsing (structured outputs / parsed
            # tool calls) and hard-rejects every tool without `strict: True`
            # (ValueError before any HTTP request). coreouto's tool schemas
            # are not strict-compatible (no additionalProperties: false,
            # optional params omitted from `required`), so the helper makes
            # streaming unusable. We only need text deltas and the final
            # completion, so we accumulate raw chunks ourselves.
            stream = await self._client.chat.completions.create(stream=True, **request_kwargs)
            resp = await self._consume_stream(stream, model, on_stream_text, on_stream_thinking)
        else:
            resp = await self._client.chat.completions.create(**request_kwargs)

        message = resp.choices[0].message
        content = message.content
        thinking = getattr(message, "reasoning_content", None)
        raw_tool_calls = getattr(message, "tool_calls", None) or []
        parsed_tool_calls: list[ToolCall] = []
        for tc in raw_tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                args = {}
            parsed_tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = None
        if resp.usage is not None:
            usage = Usage(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
            )

        return LLMResponse(
            content=content,
            tool_calls=parsed_tool_calls,
            usage=usage,
            stop_reason=getattr(resp.choices[0], "finish_reason", None),
            thinking=thinking,
            raw=resp,
        )

    @staticmethod
    async def _consume_stream(
        stream: Any,
        model: str,
        on_stream_text: Any = None,
        on_stream_thinking: Any = None,
    ) -> Any:
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
        from openai.types.chat.chat_completion_message_tool_call import (
            ChatCompletionMessageToolCall,
            Function,
        )

        completion_id = ""
        created = 0
        completion_model = model
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_deltas: dict[int, dict[str, str]] = {}
        finish_reason: Any = None
        usage: Any = None

        async for chunk in stream:
            completion_id = getattr(chunk, "id", None) or completion_id
            created = getattr(chunk, "created", None) or created
            completion_model = getattr(chunk, "model", None) or completion_model
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            chunk_choice = choices[0]
            if getattr(chunk_choice, "finish_reason", None) is not None:
                finish_reason = chunk_choice.finish_reason
            delta = getattr(chunk_choice, "delta", None)
            if delta is None:
                continue
            text = getattr(delta, "content", None)
            if text:
                content_parts.append(text)
                if on_stream_text is not None:
                    await on_stream_text(text)
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
                if on_stream_thinking is not None:
                    await on_stream_thinking(reasoning)
            for tc_delta in getattr(delta, "tool_calls", None) or []:
                index = getattr(tc_delta, "index", None) or 0
                slot = tool_deltas.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if getattr(tc_delta, "id", None):
                    slot["id"] = tc_delta.id
                fn = getattr(tc_delta, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["arguments"] += fn.arguments

        tool_calls = [
            ChatCompletionMessageToolCall(
                id=slot["id"],
                type="function",
                function=Function(name=slot["name"], arguments=slot["arguments"]),
            )
            for _, slot in sorted(tool_deltas.items())
        ]
        message = ChatCompletionMessage(
            role="assistant",
            content="".join(content_parts) or None,
            tool_calls=tool_calls or None,
        )
        if reasoning_parts:
            message.reasoning_content = "".join(reasoning_parts)
        return ChatCompletion(
            id=completion_id,
            choices=[Choice(index=0, finish_reason=finish_reason or "stop", message=message)],
            created=created,
            model=completion_model,
            object="chat.completion",
            usage=usage,
        )

    def format_assistant_message(self, response: LLMResponse) -> Message:
        tool_calls = None
        if response.tool_calls:
            tool_calls = [
                ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
                for tc in response.tool_calls
            ]
        return Message(
            role="assistant",
            content=response.content or "",
            tool_calls=tool_calls,
        )

    def format_tool_result(self, tool_call: ToolCall, result: ToolResult) -> Message:
        if result.blocks is not None:
            block_types = sorted({b.type for b in result.blocks if b.type != "text"})
            detected = ", ".join(f"{t} block detected" for t in block_types) or "multimodal blocks"
            raise ValueError(
                f"OpenAI Chat Completions does not support multimodal tool results "
                f"({detected}). Use the 'openai-response' provider instead, which "
                f"supports tool results with text + image content."
            )
        return Message(
            role="tool",
            tool_call_id=tool_call.id,
            content=result.content,
            name=tool_call.name,
        )


try:
    provider = OpenAIProvider() if AsyncOpenAI is not None else None
except Exception:
    provider = None


def register(
    api_key: str | None = None,
    base_url: str | None = None,
    name: str = "openai",
    timeout: float | None = None,
) -> None:
    register_provider(name, OpenAIProvider(api_key=api_key, base_url=base_url, timeout=timeout))
