from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolCall(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_call_id: str
    content: str
    is_error: bool = False


class Usage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @model_validator(mode="after")
    def _total_matches_sum(self) -> Usage:
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError(
                f"total_tokens ({self.total_tokens}) must equal "
                f"prompt_tokens ({self.prompt_tokens}) + "
                f"completion_tokens ({self.completion_tokens})"
            )
        return self


class LLMResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage | None = None
    raw: Any | None = None


class Message(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class AgentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    model: str
    provider: str
    system_prompt: str | None = None
    tools: list[str] = Field(default_factory=list)
    max_iterations: int = 50
    provider_config: dict[str, Any] = Field(default_factory=dict)
    provider_passthrough: dict[str, Any] = Field(default_factory=dict)


class Response(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str
    messages: list[Message]
    iterations: int
    usage: list[Usage] = Field(default_factory=list)
    finish_called: bool = True
    warnings: list[str] = Field(default_factory=list)
