from __future__ import annotations

import uuid
from typing import Any

from coreouto._types import LLMResponse, Message, ToolCall, ToolResult, Usage
from coreouto.providers import register_provider
from coreouto.tools import Tool

try:
    import google.generativeai as genai
except ImportError as exc:
    raise ImportError(
        "The google-generativeai package is required. Install it with: pip install coreouto[google]"
    ) from exc


class GoogleProvider:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client_options: dict | None = None,
    ) -> None:
        if api_key is not None:
            genai.configure(api_key=api_key)
        self._model_name = model
        self._client_options = client_options
        if model:
            self._model = genai.GenerativeModel(model, client_options=client_options)
        else:
            self._model = None

    async def create(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        target_model = model or self._model_name
        if not target_model:
            raise ValueError("model is required")

        system_parts: list[str] = []
        google_contents: list[Any] = []
        current_tool_parts: list[Any] = []

        for message in messages:
            if message.role == "system":
                system_parts.append(message.content)
            elif message.role == "user":
                if current_tool_parts:
                    google_contents.append(genai.Content(role="user", parts=current_tool_parts))
                    current_tool_parts = []
                google_contents.append(
                    genai.Content(
                        role="user",
                        parts=[genai.Part.from_text(text=message.content)],
                    )
                )
            elif message.role == "assistant":
                if current_tool_parts:
                    google_contents.append(genai.Content(role="user", parts=current_tool_parts))
                    current_tool_parts = []
                parts: list[Any] = []
                if message.content:
                    parts.append(genai.Part.from_text(text=message.content))
                if message.tool_calls:
                    for tc in message.tool_calls:
                        parts.append(genai.Part.from_function_call(name=tc.name, args=tc.arguments))
                google_contents.append(genai.Content(role="model", parts=parts))
            elif message.role == "tool":
                current_tool_parts.append(
                    genai.Part.from_function_response(
                        name=message.name or "tool",
                        response={"result": message.content},
                    )
                )

        if current_tool_parts:
            google_contents.append(genai.Content(role="user", parts=current_tool_parts))

        system_instruction = system_prompt
        if system_parts:
            joined = "\n".join(system_parts)
            if system_instruction is None:
                system_instruction = joined
            else:
                system_instruction = f"{system_instruction}\n{joined}"

        google_tools: list[dict[str, Any]] | None = None
        if tools:
            declarations = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in tools
            ]
            google_tools = [{"function_declarations": declarations}]

        if (
            self._model is not None
            and self._model_name == target_model
            and system_instruction is None
        ):
            model_obj = self._model
        else:
            model_obj = genai.GenerativeModel(
                model_name=target_model,
                system_instruction=system_instruction,
                client_options=self._client_options,
            )

        response = model_obj.generate_content(
            contents=google_contents,
            tools=google_tools,
            **kwargs,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        candidate = response.candidates[0]
        for part in candidate.content.parts:
            if part.text:
                text_parts.append(part.text)
            if part.function_call:
                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:24]}",
                        name=part.function_call.name,
                        arguments=dict(part.function_call.args),
                    )
                )

        usage = None
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
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
        tool_calls = [tc for tc in response.tool_calls] if response.tool_calls else None
        return Message(
            role="assistant",
            content=response.content or "",
            tool_calls=tool_calls,
        )

    def format_tool_result(self, tool_call: ToolCall, result: ToolResult) -> Message:
        return Message(
            role="tool",
            content=str(result.content) if hasattr(result, "content") else str(result),
            tool_call_id=tool_call.id,
            name=tool_call.name,
        )


provider = GoogleProvider()


def register(
    api_key: str | None = None,
    client_options: dict | None = None,
    name: str = "google",
) -> None:
    register_provider(name, GoogleProvider(api_key=api_key, client_options=client_options))
