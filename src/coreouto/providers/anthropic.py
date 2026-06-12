from __future__ import annotations

import base64
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


def _import_anthropic():
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:
        raise ImportError(
            "Anthropic SDK not installed. Install with: pip install coreouto[anthropic]"
        ) from exc
    return AsyncAnthropic


class AnthropicProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            client_cls = _import_anthropic()
            self._client = client_cls(api_key=api_key, base_url=base_url)

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
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
                continue
            if msg.role == "user":
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": msg.content}],
                    }
                )
                continue
            if msg.role == "assistant":
                content: list[dict[str, Any]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content.append(
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.arguments,
                            }
                        )
                anthropic_messages.append({"role": "assistant", "content": content})
                continue
            if msg.role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content,
                }
                if (
                    anthropic_messages
                    and anthropic_messages[-1]["role"] == "user"
                    and anthropic_messages[-1]["content"]
                    and anthropic_messages[-1]["content"][0].get("type") == "tool_result"
                ):
                    anthropic_messages[-1]["content"].append(block)
                else:
                    anthropic_messages.append({"role": "user", "content": [block]})
                continue

        system_str = system_prompt
        if system_str is None and system_parts:
            system_str = "\n".join(system_parts)

        anthropic_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in (tools or [])
        ]

        resp = await self._client.messages.create(
            model=model,
            system=system_str,
            messages=anthropic_messages,
            tools=anthropic_tools if anthropic_tools else None,
            max_tokens=kwargs.pop("max_tokens", 1024),
            **kwargs,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text or "")
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id or "",
                        name=block.name or "",
                        arguments=dict(block.input) if block.input else {},
                    )
                )

        usage = Usage(
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            total_tokens=resp.usage.input_tokens + resp.usage.output_tokens,
        )

        return LLMResponse(
            content="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            usage=usage,
            raw=resp,
        )

    def format_assistant_message(self, response: LLMResponse) -> Message:
        tool_calls = list(response.tool_calls) if response.tool_calls else None
        return Message(
            role="assistant",
            content=response.content or "",
            tool_calls=tool_calls,
        )

    def format_tool_result(self, tool_call: ToolCall, result: ToolResult) -> Message:
        if result.blocks is not None:
            wire_blocks: list[dict[str, Any]] = []
            for block in result.blocks:
                if isinstance(block, TextBlock):
                    wire_blocks.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    if block.data is not None:
                        wire_blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": block.mime_type,
                                    "data": base64.b64encode(block.data).decode("ascii"),
                                },
                            }
                        )
                    else:
                        wire_blocks.append(
                            {
                                "type": "image",
                                "source": {"type": "url", "url": block.url},
                            }
                        )
                elif isinstance(block, DocumentBlock):
                    if block.data is not None:
                        wire_blocks.append(
                            {
                                "type": "document",
                                "source": {
                                    "type": "text",
                                    "media_type": block.mime_type,
                                    "data": base64.b64encode(block.data).decode("ascii"),
                                },
                            }
                        )
                    else:
                        wire_blocks.append(
                            {
                                "type": "document",
                                "source": {"type": "url", "url": block.url},
                            }
                        )
                elif isinstance(block, VideoBlock):
                    if block.data is not None:
                        wire_blocks.append(
                            {
                                "type": "video",
                                "source": {
                                    "type": "base64",
                                    "media_type": block.mime_type,
                                    "data": base64.b64encode(block.data).decode("ascii"),
                                },
                            }
                        )
                    else:
                        wire_blocks.append(
                            {
                                "type": "video",
                                "source": {"type": "url", "url": block.url},
                            }
                        )
                elif isinstance(block, AudioBlock):
                    if block.data is not None:
                        wire_blocks.append(
                            {
                                "type": "audio",
                                "source": {
                                    "type": "base64",
                                    "media_type": block.mime_type,
                                    "data": base64.b64encode(block.data).decode("ascii"),
                                },
                            }
                        )
                    else:
                        wire_blocks.append(
                            {
                                "type": "audio",
                                "source": {"type": "url", "url": block.url},
                            }
                        )
            return Message.model_construct(
                role="tool",
                content=wire_blocks,
                tool_call_id=tool_call.id,
                name=tool_call.name,
            )
        return Message(
            role="tool",
            content=result.content,
            tool_call_id=tool_call.id,
            name=tool_call.name,
        )


def register(
    api_key: str | None = None,
    base_url: str | None = None,
    name: str = "anthropic",
) -> None:
    register_provider(name, AnthropicProvider(api_key=api_key, base_url=base_url))
