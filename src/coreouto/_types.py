from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolCall(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class TextBlock(BaseModel):
    """A plain text content block."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    """An image content block.

    Exactly one of `data` (raw bytes) or `url` must be set. `mime_type` is
    required when `data` is set; the URL form may omit it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["image"] = "image"
    data: bytes | None = None
    url: str | None = None
    mime_type: str | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> ImageBlock:
        if (self.data is None) == (self.url is None):
            raise ValueError("ImageBlock requires exactly one of 'data' or 'url'")
        if self.data is not None and self.mime_type is None:
            raise ValueError("ImageBlock requires 'mime_type' when 'data' is set")
        return self


class DocumentBlock(BaseModel):
    """A document content block (PDF, text, etc.).

    Exactly one of `data` (raw bytes) or `url` must be set. `mime_type` is
    required when `data` is set.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["document"] = "document"
    data: bytes | None = None
    url: str | None = None
    mime_type: str | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> DocumentBlock:
        if (self.data is None) == (self.url is None):
            raise ValueError("DocumentBlock requires exactly one of 'data' or 'url'")
        if self.data is not None and self.mime_type is None:
            raise ValueError("DocumentBlock requires 'mime_type' when 'data' is set")
        return self


class VideoBlock(BaseModel):
    """A video content block."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["video"] = "video"
    data: bytes | None = None
    url: str | None = None
    mime_type: str | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> VideoBlock:
        if (self.data is None) == (self.url is None):
            raise ValueError("VideoBlock requires exactly one of 'data' or 'url'")
        if self.data is not None and self.mime_type is None:
            raise ValueError("VideoBlock requires 'mime_type' when 'data' is set")
        return self


class AudioBlock(BaseModel):
    """An audio content block."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["audio"] = "audio"
    data: bytes | None = None
    url: str | None = None
    mime_type: str | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> AudioBlock:
        if (self.data is None) == (self.url is None):
            raise ValueError("AudioBlock requires exactly one of 'data' or 'url'")
        if self.data is not None and self.mime_type is None:
            raise ValueError("AudioBlock requires 'mime_type' when 'data' is set")
        return self


ContentBlock = Annotated[
    TextBlock | ImageBlock | DocumentBlock | VideoBlock | AudioBlock,
    Field(discriminator="type"),
]


class ToolResult(BaseModel):
    """The result of a tool invocation, returned by tool handlers.

    Provide either `content` (plain text) or `blocks` (multimodal), not both.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_call_id: str
    content: str | None = None
    blocks: list[ContentBlock] | None = None
    is_error: bool = False

    @model_validator(mode="after")
    def _exactly_one_form(self) -> ToolResult:
        if self.content is not None and self.blocks is not None:
            raise ValueError("ToolResult accepts 'content' or 'blocks', not both")
        if self.content is None and self.blocks is None:
            raise ValueError("ToolResult requires 'content' or 'blocks'")
        return self

    def flatten_text(self) -> str:
        """Concatenate all text content from `content`/`blocks` (best-effort)."""
        if self.content is not None:
            return self.content
        if self.blocks is not None:
            return "".join(block.text for block in self.blocks if isinstance(block, TextBlock))
        return ""


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
    stop_reason: str | None = None
    raw: Any | None = None


class Message(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentBlock | ToolCall]
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
    parallel_tool_calls: bool = False


class Response(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str
    messages: list[Message]
    iterations: int
    usage: list[Usage] = Field(default_factory=list)
    finish_called: bool = True
    warnings: list[str] = Field(default_factory=list)
