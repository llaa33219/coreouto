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
    ) -> None:
        if client is not None:
            self._client = client
        else:
            if AsyncOpenAI is None:
                raise ImportError(
                    "The openai package is required. Install it with: pip install coreouto[openai]"
                )
            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._stream = stream

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
            async with self._client.chat.completions.stream(**request_kwargs) as s:
                if on_stream_text is not None or on_stream_thinking is not None:
                    async for event in s:
                        if event.type == "content.delta" and on_stream_text is not None:
                            await on_stream_text(event.delta)
                        elif event.type == "chunk" and on_stream_thinking is not None:
                            choices = getattr(event.chunk, "choices", None) or []
                            if choices:
                                reasoning = getattr(choices[0].delta, "reasoning_content", None)
                                if reasoning:
                                    await on_stream_thinking(reasoning)
                resp = await s.get_final_completion()
        else:
            resp = await self._client.chat.completions.create(**request_kwargs)

        choice = resp.choices[0].message
        content = choice.content
        thinking = getattr(choice, "reasoning_content", None)
        raw_tool_calls = getattr(choice, "tool_calls", None) or []
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
            stop_reason=getattr(choice, "finish_reason", None),
            thinking=thinking,
            raw=resp,
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
) -> None:
    register_provider(name, OpenAIProvider(api_key=api_key, base_url=base_url))
