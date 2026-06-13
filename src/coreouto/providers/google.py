from __future__ import annotations

import uuid
from typing import Any

from coreouto._types import (
    LLMResponse,
    Message,
    TextBlock,
    ToolCall,
    ToolResult,
    Usage,
)
from coreouto.providers import register_provider
from coreouto.tools import Tool

try:
    from google import genai
    from google.genai import types
except ImportError as exc:
    raise ImportError(
        "The google-genai package is required. Install it with: pip install coreouto[google]"
    ) from exc


def _is_text_block(block: Any) -> bool:
    return isinstance(block, TextBlock)


def _text_blocks(blocks: list[Any]) -> list[TextBlock]:
    return [b for b in blocks if _is_text_block(b)]


def _build_function_response_part(block: Any) -> Any:
    mime_type = getattr(block, "mime_type", None)
    data = getattr(block, "data", None)
    url = getattr(block, "url", None)
    if data is not None:
        if mime_type is None:
            raise ValueError(f"{type(block).__name__} requires 'mime_type' when 'data' is set")
        return types.FunctionResponsePart(
            inline_data=types.FunctionResponseBlob(mime_type=mime_type, data=data)
        )
    if url is not None:
        return types.FunctionResponsePart(
            file_data=types.FunctionResponseFileData(file_uri=url, mime_type=mime_type)
        )
    raise ValueError(f"{type(block).__name__} must have either 'data' or 'url' set")


class GoogleProvider:
    def __init__(
        self,
        api_key: str | None = None,
        client: Any | None = None,
        http_options: dict | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            kwargs: dict[str, Any] = {}
            if api_key is not None:
                kwargs["api_key"] = api_key
            if http_options is not None:
                kwargs["http_options"] = http_options
            self._client = genai.Client(**kwargs)
        self._aio = self._client.aio

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
        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
        system_instruction = "\n".join(system_parts) if system_parts else None

        conversation: list[Any] = []

        for msg in messages:
            if msg.role == "system":
                continue
            if msg.role == "user":
                if isinstance(msg.content, list):
                    parts = [self._content_block_to_part(b) for b in msg.content]
                else:
                    parts = [types.Part.from_text(text=msg.content)]
                conversation.append(types.Content(role="user", parts=parts))
                continue
            if msg.role == "assistant":
                parts: list[Any] = []
                if msg.content:
                    if isinstance(msg.content, list):
                        for item in msg.content:
                            if isinstance(item, ToolCall):
                                parts.append(
                                    types.Part.from_function_call(
                                        name=item.name, args=item.arguments
                                    )
                                )
                            else:
                                parts.append(self._content_block_to_part(item))
                    else:
                        parts.append(types.Part.from_text(text=msg.content))
                if msg.tool_calls:
                    seen = {
                        item.id
                        for item in (msg.content if not isinstance(msg.content, str) else [])
                        if isinstance(item, ToolCall)
                    }
                    for tc in msg.tool_calls:
                        if tc.id in seen:
                            continue
                        parts.append(types.Part.from_function_call(name=tc.name, args=tc.arguments))
                conversation.append(types.Content(role="model", parts=parts))
                continue
            if msg.role == "tool":
                if isinstance(msg.content, list):
                    text_blocks = _text_blocks(msg.content)
                    output_text = " ".join(b.text for b in text_blocks)
                    response_payload: dict[str, Any] = {"output": output_text}
                    response_parts = [
                        _build_function_response_part(b)
                        for b in msg.content
                        if not _is_text_block(b)
                    ]
                    part_kwargs: dict[str, Any] = {
                        "name": msg.name or "tool",
                        "response": response_payload,
                    }
                    if response_parts:
                        part_kwargs["parts"] = response_parts
                    part = types.Part.from_function_response(**part_kwargs)
                else:
                    part = types.Part.from_function_response(
                        name=msg.name or "tool",
                        response={"output": msg.content},
                    )
                conversation.append(types.Content(role="user", parts=[part]))
                continue

        config_kwargs: dict[str, Any] = {}
        if system_instruction is not None:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            declarations = [
                types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters_json_schema=t.parameters,
                )
                for t in tools
            ]
            config_kwargs["tools"] = [types.Tool(function_declarations=declarations)]
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        call_kwargs: dict[str, Any] = {"model": model, "contents": conversation}
        if config is not None:
            call_kwargs["config"] = config
        call_kwargs.update(kwargs)

        response = await self._aio.models.generate_content(**call_kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        if getattr(response, "candidates", None):
            candidate = response.candidates[0]
            for part in getattr(candidate.content, "parts", []) or []:
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    call_id = getattr(fc, "id", None) or self._gen_call_id()
                    tool_calls.append(
                        ToolCall(
                            id=call_id,
                            name=fc.name or "",
                            arguments=dict(fc.args) if fc.args else {},
                        )
                    )
        if not text_parts and not tool_calls:
            if getattr(response, "function_calls", None):
                for fc in response.function_calls:
                    call_id = getattr(fc, "id", None) or self._gen_call_id()
                    tool_calls.append(
                        ToolCall(
                            id=call_id,
                            name=fc.name or "",
                            arguments=dict(fc.args) if fc.args else {},
                        )
                    )
            elif getattr(response, "text", None):
                text_parts.append(response.text)

        usage = None
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is not None:
            prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        return LLMResponse(
            content="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            usage=usage,
            raw=response,
        )

    def format_assistant_message(self, response: LLMResponse) -> Message:
        tool_calls = None
        if response.tool_calls:
            tool_calls = [
                ToolCall(id=tc.id, name=tc.name, arguments=dict(tc.arguments))
                for tc in response.tool_calls
            ]
        return Message(
            role="assistant",
            content=response.content or "",
            tool_calls=tool_calls,
        )

    def format_tool_result(self, tool_call: ToolCall, result: ToolResult) -> Message:
        if result.content is not None:
            return Message(
                role="tool",
                content=result.content,
                tool_call_id=result.tool_call_id,
                name=tool_call.name,
            )
        return Message(
            role="tool",
            content=result.blocks,
            tool_call_id=result.tool_call_id,
            name=tool_call.name,
        )

    @staticmethod
    def _gen_call_id() -> str:
        return f"call_{uuid.uuid4().hex[:24]}"

    @staticmethod
    def _content_block_to_part(block: Any) -> Any:
        if isinstance(block, TextBlock):
            return types.Part.from_text(text=block.text)
        mime_type = getattr(block, "mime_type", None)
        data = getattr(block, "data", None)
        url = getattr(block, "url", None)
        if data is not None:
            if mime_type is None:
                raise ValueError(f"{type(block).__name__} requires 'mime_type' when 'data' is set")
            return types.Part.from_bytes(data=data, mime_type=mime_type)
        if url is not None:
            return types.Part.from_uri(file_uri=url, mime_type=mime_type)
        raise ValueError(f"{type(block).__name__} must have either 'data' or 'url' set")


try:
    provider = GoogleProvider()
except Exception:
    provider = None


def register(
    api_key: str | None = None,
    name: str = "google",
    http_options: dict | None = None,
) -> None:
    register_provider(name, GoogleProvider(api_key=api_key, http_options=http_options))
