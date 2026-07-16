from __future__ import annotations

import base64
import json
from typing import Any

from coreouto._types import (
    AudioBlock,
    DocumentBlock,
    ImageBlock,
    LLMResponse,
    Message,
    TextBlock,
    ToolCall,
    ToolResult,
    Usage,
    VideoBlock,
)
from coreouto.providers import register_provider
from coreouto.tools import Tool

_MIME_EXTENSIONS: dict[str, str] = {
    "application/pdf": "pdf",
    "application/json": "json",
    "text/plain": "txt",
    "text/csv": "csv",
    "text/html": "html",
    "text/markdown": "md",
}


def _doc_extension(mime_type: str | None) -> str:
    if mime_type is None:
        return "bin"
    return _MIME_EXTENSIONS.get(mime_type, mime_type.split("/", 1)[-1] or "bin")


class OpenAIResponseProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        stream: bool = False,
        error_handling: list | None = None,
    ) -> None:
        if client is not None:
            self._client = client
            self._api_key = None
            self._base_url = None
        else:
            self._client = None
            self._api_key = api_key
            self._base_url = base_url
        self._stream = stream
        self.error_handling = error_handling

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required for OpenAIResponseProvider. "
                "Install with: pip install coreouto[openai]"
            ) from exc
        self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    async def create(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        system_parts: list[str] = []
        if system_prompt:
            system_parts.append(system_prompt)

        responses_input: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
                continue
            if msg.role == "user":
                responses_input.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": msg.content}],
                    }
                )
            elif msg.role == "assistant":
                if isinstance(msg.content, str):
                    if msg.content:
                        responses_input.append(
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": msg.content,
                                    }
                                ],
                            }
                        )
                else:
                    for item in msg.content:
                        if isinstance(item, TextBlock):
                            responses_input.append(
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": item.text,
                                        }
                                    ],
                                }
                            )
                        elif isinstance(item, ToolCall):
                            responses_input.append(
                                {
                                    "type": "function_call",
                                    "call_id": item.id,
                                    "name": item.name,
                                    "arguments": json.dumps(item.arguments),
                                }
                            )
                if msg.tool_calls:
                    seen = {
                        item.id
                        for item in (msg.content if not isinstance(msg.content, str) else [])
                        if isinstance(item, ToolCall)
                    }
                    for tc in msg.tool_calls:
                        if tc.id in seen:
                            continue
                        responses_input.append(
                            {
                                "type": "function_call",
                                "call_id": tc.id,
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            }
                        )
            elif msg.role == "tool":
                responses_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id,
                        "output": msg.content,
                    }
                )

        system_str = "\n".join(system_parts) if system_parts else None

        responses_tools: list[dict[str, Any]] | None = None
        if tools:
            responses_tools = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in tools
            ]

        client = self._get_client()
        use_stream = kwargs.pop("stream", self._stream)
        on_stream_text = kwargs.pop("_on_stream_text", None)
        on_stream_thinking = kwargs.pop("_on_stream_thinking", None)
        request_kwargs = dict(
            model=model,
            instructions=system_str,
            input=responses_input,
            tools=responses_tools if responses_tools else None,
            **kwargs,
        )

        if use_stream:
            async with client.responses.stream(**request_kwargs) as s:
                if on_stream_text is not None or on_stream_thinking is not None:
                    async for event in s:
                        if (
                            event.type == "response.output_text.delta"
                            and on_stream_text is not None
                        ):
                            await on_stream_text(event.delta)
                        elif (
                            event.type == "response.reasoning_summary_text.delta"
                            and on_stream_thinking is not None
                        ):
                            await on_stream_thinking(event.delta)
                resp = await s.get_final_response()
        else:
            resp = await client.responses.create(**request_kwargs)

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for item in resp.output:
            if getattr(item, "type", None) == "message":
                for part in getattr(item, "content", []) or []:
                    if getattr(part, "type", None) == "output_text":
                        text_parts.append(getattr(part, "text", "") or "")
            elif getattr(item, "type", None) == "reasoning":
                for summary in getattr(item, "summary", []) or []:
                    text = getattr(summary, "text", None)
                    if text:
                        thinking_parts.append(text)
            elif getattr(item, "type", None) == "function_call":
                tool_calls.append(
                    ToolCall(
                        id=getattr(item, "call_id", ""),
                        name=getattr(item, "name", ""),
                        arguments=json.loads(getattr(item, "arguments", "{}")),
                    )
                )

        usage = None
        if hasattr(resp, "usage") and resp.usage is not None:
            prompt_tokens = getattr(resp.usage, "input_tokens", 0)
            completion_tokens = getattr(resp.usage, "output_tokens", 0)
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        status = getattr(resp, "status", None) or "completed"
        if status == "incomplete":
            incomplete = getattr(resp, "incomplete_details", None)
            reason = getattr(incomplete, "reason", None) if incomplete else None
            stop_reason = f"incomplete:{reason}" if reason else "incomplete"
        else:
            stop_reason = "completed"

        return LLMResponse(
            content="".join(text_parts) if text_parts else "",
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=stop_reason,
            thinking="".join(thinking_parts) if thinking_parts else None,
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

    def format_tool_result(self, tool_call: ToolCall, result: Any) -> Message:
        if not isinstance(result, ToolResult):
            return Message(
                role="tool",
                content=str(result),
                tool_call_id=tool_call.id,
                name=tool_call.name,
            )

        if result.content is not None:
            return Message(
                role="tool",
                content=result.content,
                tool_call_id=tool_call.id,
                name=tool_call.name,
            )

        output: list[dict[str, Any]] = []
        for block in result.blocks or []:
            if isinstance(block, TextBlock):
                output.append({"type": "input_text", "text": block.text})
            elif isinstance(block, ImageBlock):
                if block.data is not None:
                    b64 = base64.b64encode(block.data).decode("ascii")
                    output.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:{block.mime_type};base64,{b64}",
                        }
                    )
                else:
                    output.append({"type": "input_image", "image_url": block.url})
            elif isinstance(block, DocumentBlock):
                if block.data is not None:
                    b64 = base64.b64encode(block.data).decode("ascii")
                    filename = f"document.{_doc_extension(block.mime_type)}"
                    output.append(
                        {
                            "type": "input_file",
                            "filename": filename,
                            "file_data": b64,
                        }
                    )
                else:
                    output.append({"type": "input_file", "file_url": block.url})
            elif isinstance(block, (VideoBlock, AudioBlock)):
                raise ValueError(
                    "OpenAI Responses does not support video/audio blocks in "
                    "tool results. Use Anthropic or Google for that."
                )
            else:
                raise TypeError(f"unsupported content block: {type(block).__name__}")

        return Message.model_construct(
            role="tool",
            content=output,
            tool_call_id=tool_call.id,
            name=tool_call.name,
        )


def register(
    api_key: str | None = None,
    base_url: str | None = None,
    name: str = "openai-response",
) -> None:
    register_provider(name, OpenAIResponseProvider(api_key=api_key, base_url=base_url))


try:
    provider = OpenAIResponseProvider()
except Exception:
    provider = None
