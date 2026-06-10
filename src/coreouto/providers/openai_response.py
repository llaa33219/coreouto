from __future__ import annotations

import json
from typing import Any

from coreouto._types import LLMResponse, Message, ToolCall, Usage
from coreouto.providers import register_provider
from coreouto.tools import Tool


class OpenAIResponseProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._client = client
            self._api_key = None
            self._base_url = None
        else:
            self._client = None
            self._api_key = api_key
            self._base_url = base_url

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
                if msg.tool_calls:
                    for tc in msg.tool_calls:
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
        resp = await client.responses.create(
            model=model,
            instructions=system_str,
            input=responses_input,
            tools=responses_tools if responses_tools else None,
            **kwargs,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for item in resp.output:
            if getattr(item, "type", None) == "message":
                for part in getattr(item, "content", []) or []:
                    if part.get("type") == "output_text":
                        text_parts.append(part.get("text", ""))
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

        return LLMResponse(
            content="".join(text_parts) if text_parts else "",
            tool_calls=tool_calls,
            usage=usage,
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
        from coreouto._types import ToolResult

        content = str(result.content) if isinstance(result, ToolResult) else str(result)
        return Message(
            role="tool",
            content=content,
            tool_call_id=tool_call.id,
            name=tool_call.name,
        )


def register(
    api_key: str | None = None,
    base_url: str | None = None,
    name: str = "openai-response",
) -> None:
    register_provider(name, OpenAIResponseProvider(api_key=api_key, base_url=base_url))


provider = OpenAIResponseProvider()
