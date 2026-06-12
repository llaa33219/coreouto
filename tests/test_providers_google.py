from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from coreouto._types import (
    DocumentBlock,
    ImageBlock,
    LLMResponse,
    Message,
    TextBlock,
    ToolCall,
    ToolResult,
    Usage,
)
from coreouto.providers.base import Provider
from coreouto.tools import Tool


@dataclass
class FakeFunctionCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass
class FakeFunctionResponse:
    name: str | None = None
    response: dict[str, Any] | None = None
    parts: list[Any] | None = None
    id: str | None = None
    scheduling: Any = None
    will_continue: bool | None = None


@dataclass
class FakePart:
    text: str | None = None
    function_call: FakeFunctionCall | None = None
    function_response: FakeFunctionResponse | None = None
    inline_data: Any = None

    @classmethod
    def from_text(cls, text: str) -> FakePart:
        return cls(text=text)

    @classmethod
    def from_function_call(cls, name: str, args: dict[str, Any]) -> FakePart:
        return cls(function_call=FakeFunctionCall(name=name, args=args))

    @classmethod
    def from_function_response(
        cls,
        name: str,
        response: dict[str, Any],
        parts: list[Any] | None = None,
    ) -> FakePart:
        return cls(
            function_response=FakeFunctionResponse(name=name, response=response, parts=parts or [])
        )


@dataclass
class FakeContent:
    role: str
    parts: list[FakePart] = field(default_factory=list)


@dataclass
class FakeUsageMetadata:
    prompt_token_count: int
    candidates_token_count: int
    total_token_count: int = 0


@dataclass
class FakeCandidate:
    content: FakeContent


@dataclass
class FakeGenerateContentResponse:
    candidates: list[FakeCandidate] = field(default_factory=list)
    usage_metadata: FakeUsageMetadata | None = None
    function_calls: list[FakeFunctionCall] | None = None
    _text: str | None = None

    @property
    def text(self) -> str | None:
        if self._text is not None:
            return self._text
        if self.candidates and self.candidates[0].content.parts:
            for part in self.candidates[0].content.parts:
                if part.text:
                    return part.text
        return None


@dataclass
class FakeModels:
    response: FakeGenerateContentResponse | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def generate_content(
        self,
        *,
        model: str,
        contents: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> FakeGenerateContentResponse:
        self.calls.append(
            {
                "model": model,
                "contents": contents,
                "config": config,
                "kwargs": kwargs,
            }
        )
        if self.response is None:
            raise AssertionError("FakeModels.generate_content called without a queued response")
        return self.response

    def queue(self, response: FakeGenerateContentResponse) -> None:
        self.response = response


@dataclass
class FakeAio:
    models: FakeModels = field(default_factory=FakeModels)


@dataclass
class FakeAsyncClient:
    aio: FakeAio = field(default_factory=FakeAio)
    recorded_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeClientRecorder:
    recorded_args: dict[str, Any] = field(default_factory=dict)
    aio: FakeAio = field(default_factory=FakeAio)


class _ClientFactory:
    def __init__(self) -> None:
        self.constructed: list[FakeClientRecorder] = []

    def __call__(self, **kwargs: Any) -> FakeClientRecorder:
        rec = FakeClientRecorder(recorded_args=kwargs)
        self.constructed.append(rec)
        return rec


@pytest.fixture
def fake_client() -> FakeAsyncClient:
    return FakeAsyncClient()


@pytest.fixture
def provider(fake_client: FakeAsyncClient):
    from coreouto.providers.google import GoogleProvider

    return GoogleProvider(client=fake_client)


def test_provider_satisfies_protocol(provider):
    assert isinstance(provider, Provider)


@pytest.mark.asyncio
async def test_create_simple_user_message(provider, fake_client):
    fake_client.aio.models.queue(
        FakeGenerateContentResponse(
            candidates=[
                FakeCandidate(
                    content=FakeContent(
                        role="model",
                        parts=[FakePart.from_text("Hello, world!")],
                    )
                )
            ],
            usage_metadata=FakeUsageMetadata(prompt_token_count=3, candidates_token_count=5),
        )
    )
    result = await provider.create(
        messages=[Message(role="user", content="Say hello")],
        model="gemini-2.5-flash",
    )
    assert result.content == "Hello, world!"
    assert result.tool_calls == []
    assert result.usage is not None
    assert result.usage.prompt_tokens == 3
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 8
    assert result.raw is not None

    call = fake_client.aio.models.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["config"] is None
    contents = call["contents"]
    assert len(contents) == 1
    assert contents[0].role == "user"
    assert len(contents[0].parts) == 1
    assert contents[0].parts[0].text == "Say hello"


@pytest.mark.asyncio
async def test_create_with_system_prompt_arg(provider, fake_client):
    fake_client.aio.models.queue(
        FakeGenerateContentResponse(
            candidates=[
                FakeCandidate(
                    content=FakeContent(
                        role="model",
                        parts=[FakePart.from_text("OK")],
                    )
                )
            ],
            usage_metadata=FakeUsageMetadata(prompt_token_count=1, candidates_token_count=1),
        )
    )
    await provider.create(
        messages=[Message(role="user", content="Hi")],
        model="gemini-2.5-flash",
        system_prompt="You are a tester.",
    )
    call = fake_client.aio.models.calls[0]
    assert call["config"] is not None
    assert call["config"].system_instruction == "You are a tester."


@pytest.mark.asyncio
async def test_create_with_system_role_messages(provider, fake_client):
    fake_client.aio.models.queue(
        FakeGenerateContentResponse(
            candidates=[
                FakeCandidate(
                    content=FakeContent(
                        role="model",
                        parts=[FakePart.from_text("OK")],
                    )
                )
            ],
            usage_metadata=FakeUsageMetadata(prompt_token_count=1, candidates_token_count=1),
        )
    )
    await provider.create(
        messages=[
            Message(role="system", content="Be concise."),
            Message(role="system", content="Use JSON."),
            Message(role="user", content="Hi"),
        ],
        model="gemini-2.5-flash",
    )
    call = fake_client.aio.models.calls[0]
    assert call["config"].system_instruction == "Be concise.\nUse JSON."


@pytest.mark.asyncio
async def test_create_system_prompt_and_system_messages_joined(provider, fake_client):
    fake_client.aio.models.queue(
        FakeGenerateContentResponse(
            candidates=[
                FakeCandidate(
                    content=FakeContent(
                        role="model",
                        parts=[FakePart.from_text("OK")],
                    )
                )
            ],
            usage_metadata=FakeUsageMetadata(prompt_token_count=1, candidates_token_count=1),
        )
    )
    await provider.create(
        messages=[
            Message(role="system", content="from-messages"),
            Message(role="user", content="hi"),
        ],
        model="gemini-2.5-flash",
        system_prompt="from-arg",
    )
    call = fake_client.aio.models.calls[0]
    assert call["config"].system_instruction == "from-arg\nfrom-messages"


@pytest.mark.asyncio
async def test_create_with_tool_result_text(provider, fake_client):
    fake_client.aio.models.queue(
        FakeGenerateContentResponse(
            candidates=[
                FakeCandidate(
                    content=FakeContent(
                        role="model",
                        parts=[FakePart.from_text("Got it.")],
                    )
                )
            ],
            usage_metadata=FakeUsageMetadata(prompt_token_count=2, candidates_token_count=1),
        )
    )
    await provider.create(
        messages=[
            Message(role="user", content="What is 2+2?"),
            Message(
                role="tool",
                content="4",
                tool_call_id="call_123",
                name="calculator",
            ),
        ],
        model="gemini-2.5-flash",
    )
    call = fake_client.aio.models.calls[0]
    contents = call["contents"]
    assert len(contents) == 2
    assert contents[0].role == "user"
    assert len(contents[0].parts) == 1
    assert contents[0].parts[0].text == "What is 2+2?"
    assert contents[1].role == "user"
    fr = contents[1].parts[0]
    assert fr.function_response.name == "calculator"
    assert fr.function_response.response == {"output": "4"}
    assert not (fr.function_response.parts or [])


@pytest.mark.asyncio
async def test_create_assistant_with_tool_calls(provider, fake_client):
    fake_client.aio.models.queue(
        FakeGenerateContentResponse(
            candidates=[
                FakeCandidate(
                    content=FakeContent(
                        role="model",
                        parts=[FakePart.from_function_call(name="search", args={"query": "cats"})],
                    )
                )
            ],
            usage_metadata=FakeUsageMetadata(prompt_token_count=4, candidates_token_count=6),
        )
    )
    await provider.create(
        messages=[
            Message(
                role="assistant",
                content="Let me search.",
                tool_calls=[ToolCall(id="call_abc", name="search", arguments={"query": "cats"})],
            )
        ],
        model="gemini-2.5-flash",
    )
    call = fake_client.aio.models.calls[0]
    contents = call["contents"]
    assert len(contents) == 1
    assert contents[0].role == "model"
    parts = contents[0].parts
    assert parts[0].text == "Let me search."
    assert parts[1].function_call is not None
    assert parts[1].function_call.name == "search"
    assert parts[1].function_call.args == {"query": "cats"}


@pytest.mark.asyncio
async def test_create_tools_translated_to_function_declarations(provider, fake_client):
    fake_client.aio.models.queue(
        FakeGenerateContentResponse(
            candidates=[
                FakeCandidate(
                    content=FakeContent(
                        role="model",
                        parts=[FakePart.from_text("OK")],
                    )
                )
            ],
            usage_metadata=FakeUsageMetadata(prompt_token_count=1, candidates_token_count=1),
        )
    )
    tool = Tool(
        name="search",
        description="Search the web",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=lambda query: query,
    )
    await provider.create(
        messages=[Message(role="user", content="Search for cats")],
        model="gemini-2.5-flash",
        tools=[tool],
    )
    call = fake_client.aio.models.calls[0]
    assert call["config"] is not None
    assert len(call["config"].tools) == 1
    assert call["config"].tools[0].function_declarations[0].name == "search"
    assert call["config"].tools[0].function_declarations[0].description == "Search the web"
    assert call["config"].tools[0].function_declarations[0].parameters_json_schema == {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }


@pytest.mark.asyncio
async def test_create_tool_result_image_data(provider, fake_client):
    fake_client.aio.models.queue(
        FakeGenerateContentResponse(
            candidates=[
                FakeCandidate(
                    content=FakeContent(
                        role="model",
                        parts=[FakePart.from_text("ok")],
                    )
                )
            ],
            usage_metadata=FakeUsageMetadata(prompt_token_count=1, candidates_token_count=1),
        )
    )
    tc = ToolCall(id="call_img", name="snap", arguments={})
    tr = ToolResult(
        tool_call_id="call_img",
        blocks=[ImageBlock(data=b"PNGDATA", mime_type="image/png")],
    )
    await provider.create(
        messages=[
            Message(role="user", content="snap"),
            Message(
                role="assistant",
                content="",
                tool_calls=[tc],
            ),
            provider.format_tool_result(tc, tr),
        ],
        model="gemini-2.5-flash",
    )
    call = fake_client.aio.models.calls[0]
    contents = call["contents"]
    assert contents[-1].role == "user"
    fr = contents[-1].parts[0].function_response
    assert fr.name == "snap"
    assert fr.response == {"output": ""}
    assert len(fr.parts) == 1
    blob = fr.parts[0].inline_data
    assert blob.mime_type == "image/png"
    assert blob.data == b"PNGDATA"


@pytest.mark.asyncio
async def test_create_tool_result_image_url(provider, fake_client):
    fake_client.aio.models.queue(
        FakeGenerateContentResponse(
            candidates=[
                FakeCandidate(
                    content=FakeContent(
                        role="model",
                        parts=[FakePart.from_text("ok")],
                    )
                )
            ],
            usage_metadata=FakeUsageMetadata(prompt_token_count=1, candidates_token_count=1),
        )
    )
    tc = ToolCall(id="call_img", name="snap", arguments={})
    tr = ToolResult(
        tool_call_id="call_img",
        blocks=[ImageBlock(url="https://example.com/cat.png")],
    )
    await provider.create(
        messages=[
            Message(role="user", content="snap"),
            Message(role="assistant", content="", tool_calls=[tc]),
            provider.format_tool_result(tc, tr),
        ],
        model="gemini-2.5-flash",
    )
    call = fake_client.aio.models.calls[0]
    fr = call["contents"][-1].parts[0].function_response
    assert fr.name == "snap"
    file_data = fr.parts[0].file_data
    assert file_data.file_uri == "https://example.com/cat.png"


@pytest.mark.asyncio
async def test_create_tool_result_document_data(provider, fake_client):
    fake_client.aio.models.queue(
        FakeGenerateContentResponse(
            candidates=[
                FakeCandidate(
                    content=FakeContent(
                        role="model",
                        parts=[FakePart.from_text("ok")],
                    )
                )
            ],
            usage_metadata=FakeUsageMetadata(prompt_token_count=1, candidates_token_count=1),
        )
    )
    tc = ToolCall(id="call_doc", name="read", arguments={})
    tr = ToolResult(
        tool_call_id="call_doc",
        blocks=[DocumentBlock(data=b"PDFDATA", mime_type="application/pdf")],
    )
    await provider.create(
        messages=[
            Message(role="user", content="read"),
            Message(role="assistant", content="", tool_calls=[tc]),
            provider.format_tool_result(tc, tr),
        ],
        model="gemini-2.5-flash",
    )
    call = fake_client.aio.models.calls[0]
    fr = call["contents"][-1].parts[0].function_response
    assert fr.name == "read"
    assert fr.parts[0].inline_data.mime_type == "application/pdf"
    assert fr.parts[0].inline_data.data == b"PDFDATA"


@pytest.mark.asyncio
async def test_create_tool_result_mixed_text_and_image(provider, fake_client):
    fake_client.aio.models.queue(
        FakeGenerateContentResponse(
            candidates=[
                FakeCandidate(
                    content=FakeContent(
                        role="model",
                        parts=[FakePart.from_text("ok")],
                    )
                )
            ],
            usage_metadata=FakeUsageMetadata(prompt_token_count=1, candidates_token_count=1),
        )
    )
    tc = ToolCall(id="call_mix", name="snap", arguments={})
    tr = ToolResult(
        tool_call_id="call_mix",
        blocks=[
            TextBlock(text="Here is a picture:"),
            ImageBlock(data=b"PNGDATA", mime_type="image/png"),
        ],
    )
    await provider.create(
        messages=[
            Message(role="user", content="snap"),
            Message(role="assistant", content="", tool_calls=[tc]),
            provider.format_tool_result(tc, tr),
        ],
        model="gemini-2.5-flash",
    )
    call = fake_client.aio.models.calls[0]
    fr = call["contents"][-1].parts[0].function_response
    assert fr.name == "snap"
    assert fr.response == {"output": "Here is a picture:"}
    assert len(fr.parts) == 1
    assert fr.parts[0].inline_data.mime_type == "image/png"
    assert fr.parts[0].inline_data.data == b"PNGDATA"


@pytest.mark.asyncio
async def test_create_function_call_response_parsed(provider, fake_client):
    fake_client.aio.models.queue(
        FakeGenerateContentResponse(
            candidates=[
                FakeCandidate(
                    content=FakeContent(
                        role="model",
                        parts=[
                            FakePart.from_function_call(
                                name="get_weather", args={"location": "NYC"}
                            ),
                            FakePart.from_function_call(name="get_time", args={"zone": "EST"}),
                        ],
                    )
                )
            ],
            usage_metadata=FakeUsageMetadata(prompt_token_count=5, candidates_token_count=7),
        )
    )
    result = await provider.create(
        messages=[Message(role="user", content="What's the weather and time?")],
        model="gemini-2.5-flash",
    )
    assert len(result.tool_calls) == 2
    names = [tc.name for tc in result.tool_calls]
    assert "get_weather" in names
    assert "get_time" in names
    for tc in result.tool_calls:
        assert tc.id.startswith("call_")


@pytest.mark.asyncio
async def test_create_mixed_text_and_function_call(provider, fake_client):
    fake_client.aio.models.queue(
        FakeGenerateContentResponse(
            candidates=[
                FakeCandidate(
                    content=FakeContent(
                        role="model",
                        parts=[
                            FakePart.from_text("I'll search for you."),
                            FakePart.from_function_call(name="search", args={"query": "cats"}),
                        ],
                    )
                )
            ],
            usage_metadata=FakeUsageMetadata(prompt_token_count=2, candidates_token_count=4),
        )
    )
    result = await provider.create(
        messages=[Message(role="user", content="Tell me about cats")],
        model="gemini-2.5-flash",
    )
    assert result.content == "I'll search for you."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search"


@pytest.mark.asyncio
async def test_create_is_error_field_does_not_break(provider, fake_client):
    fake_client.aio.models.queue(
        FakeGenerateContentResponse(
            candidates=[
                FakeCandidate(
                    content=FakeContent(
                        role="model",
                        parts=[FakePart.from_text("ok")],
                    )
                )
            ],
            usage_metadata=FakeUsageMetadata(prompt_token_count=1, candidates_token_count=1),
        )
    )
    tc = ToolCall(id="call_err", name="boom", arguments={})
    tr = ToolResult(
        tool_call_id="call_err",
        content="Tool exploded",
        is_error=True,
    )
    await provider.create(
        messages=[
            Message(role="user", content="do it"),
            Message(role="assistant", content="", tool_calls=[tc]),
            provider.format_tool_result(tc, tr),
        ],
        model="gemini-2.5-flash",
    )
    call = fake_client.aio.models.calls[0]
    fr = call["contents"][-1].parts[0].function_response
    assert fr.response == {"output": "Tool exploded"}


def test_format_assistant_message_text_only(provider):
    response = LLMResponse(
        content="Hello!",
        usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    msg = provider.format_assistant_message(response)
    assert msg.role == "assistant"
    assert msg.content == "Hello!"
    assert msg.tool_calls is None


def test_format_assistant_message_with_tool_calls(provider):
    response = LLMResponse(
        content="Let me help.",
        tool_calls=[ToolCall(id="call_abc", name="search", arguments={"q": "x"})],
        usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    msg = provider.format_assistant_message(response)
    assert msg.role == "assistant"
    assert msg.content == "Let me help."
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].id == "call_abc"
    assert msg.tool_calls[0].name == "search"
    assert msg.tool_calls[0].arguments == {"q": "x"}


def test_format_tool_result_text(provider):
    tc = ToolCall(id="call_123", name="calc", arguments={"a": 1})
    tr = ToolResult(tool_call_id="call_123", content="42")
    msg = provider.format_tool_result(tc, tr)
    assert msg.role == "tool"
    assert msg.content == "42"
    assert msg.tool_call_id == "call_123"
    assert msg.name == "calc"


def test_format_tool_result_blocks(provider):
    tc = ToolCall(id="call_123", name="snap", arguments={})
    tr = ToolResult(
        tool_call_id="call_123",
        blocks=[ImageBlock(data=b"x", mime_type="image/png")],
    )
    msg = provider.format_tool_result(tc, tr)
    assert msg.role == "tool"
    assert msg.tool_call_id == "call_123"
    assert msg.name == "snap"
    assert isinstance(msg.content, list)
    assert isinstance(msg.content[0], ImageBlock)


def test_format_tool_result_uses_tool_call_name_when_name_missing(provider):
    tc = ToolCall(id="call_1", name="lookup", arguments={})
    tr = ToolResult(tool_call_id="call_1", content="hi")
    msg = provider.format_tool_result(tc, tr)
    assert msg.name == "lookup"


def test_register(fake_client):
    from coreouto.providers import clear_providers, get_provider
    from coreouto.providers.google import register

    clear_providers()
    register(api_key="my-key", name="google-test")
    provider = get_provider("google-test")
    assert provider is not None
    clear_providers()


def test_module_level_provider_instance(fake_client):
    from coreouto.providers.google import provider

    assert provider is None or provider is not None


def test_client_injection_used_directly(provider, fake_client):
    assert provider._client is fake_client
    assert provider._aio is fake_client.aio


def test_constructor_uses_client_factory(monkeypatch: pytest.MonkeyPatch):
    from coreouto.providers.google import GoogleProvider

    factory = _ClientFactory()
    monkeypatch.setattr("coreouto.providers.google.genai.Client", factory)
    p = GoogleProvider(api_key="abc123", http_options={"base_url": "https://proxy.example.com"})
    assert len(factory.constructed) == 1
    recorded = factory.constructed[0]
    assert recorded.recorded_args["api_key"] == "abc123"
    assert recorded.recorded_args["http_options"] == {"base_url": "https://proxy.example.com"}
    assert p._client is recorded
    assert p._aio is recorded.aio


def test_constructor_without_http_options(monkeypatch: pytest.MonkeyPatch):
    from coreouto.providers.google import GoogleProvider

    factory = _ClientFactory()
    monkeypatch.setattr("coreouto.providers.google.genai.Client", factory)
    p = GoogleProvider(api_key="k")
    assert "http_options" not in factory.constructed[0].recorded_args
    assert p._client is factory.constructed[0]


def test_register_function_passes_http_options(monkeypatch: pytest.MonkeyPatch):
    from coreouto.providers import clear_providers, get_provider
    from coreouto.providers.google import register

    factory = _ClientFactory()
    monkeypatch.setattr("coreouto.providers.google.genai.Client", factory)
    clear_providers()
    register(
        api_key="x",
        http_options={"base_url": "https://proxy.example.com"},
        name="google-proxy",
    )
    p = get_provider("google-proxy")
    assert factory.constructed[0].recorded_args["api_key"] == "x"
    assert factory.constructed[0].recorded_args["http_options"] == {
        "base_url": "https://proxy.example.com"
    }
    assert p is not None
    clear_providers()


def test_import_error_without_sdk(monkeypatch: pytest.MonkeyPatch):
    import sys

    saved = sys.modules.pop("coreouto.providers.google", None)
    saved_genai = sys.modules.pop("google", None)
    saved_google_genai = sys.modules.pop("google.genai", None)
    saved_google_genai_types = sys.modules.pop("google.genai.types", None)
    sys.modules["google"] = None
    sys.modules["google.genai"] = None
    try:
        with pytest.raises(ImportError, match=r"pip install coreouto\[google\]"):
            from coreouto.providers.google import GoogleProvider  # noqa: F401
    finally:
        sys.modules.pop("coreouto.providers.google", None)
        if saved_genai is not None:
            sys.modules["google"] = saved_genai
        else:
            sys.modules.pop("google", None)
        if saved_google_genai is not None:
            sys.modules["google.genai"] = saved_google_genai
        else:
            sys.modules.pop("google.genai", None)
        if saved_google_genai_types is not None:
            sys.modules["google.genai.types"] = saved_google_genai_types
        if saved is not None:
            sys.modules["coreouto.providers.google"] = saved


@pytest.mark.asyncio
async def test_create_passes_extra_kwargs_to_generate_content(provider, fake_client):
    fake_client.aio.models.queue(
        FakeGenerateContentResponse(
            candidates=[
                FakeCandidate(
                    content=FakeContent(
                        role="model",
                        parts=[FakePart.from_text("ok")],
                    )
                )
            ],
            usage_metadata=FakeUsageMetadata(prompt_token_count=1, candidates_token_count=1),
        )
    )
    await provider.create(
        messages=[Message(role="user", content="hi")],
        model="gemini-2.5-flash",
        temperature=0.3,
    )
    call = fake_client.aio.models.calls[0]
    assert call["kwargs"] == {"temperature": 0.3}


@pytest.mark.asyncio
async def test_create_tool_result_blocks_only_text(provider, fake_client):
    fake_client.aio.models.queue(
        FakeGenerateContentResponse(
            candidates=[
                FakeCandidate(
                    content=FakeContent(
                        role="model",
                        parts=[FakePart.from_text("ok")],
                    )
                )
            ],
            usage_metadata=FakeUsageMetadata(prompt_token_count=1, candidates_token_count=1),
        )
    )
    tc = ToolCall(id="call_t", name="echo", arguments={})
    tr = ToolResult(
        tool_call_id="call_t",
        blocks=[TextBlock(text="just text from blocks")],
    )
    await provider.create(
        messages=[
            Message(role="user", content="echo"),
            Message(role="assistant", content="", tool_calls=[tc]),
            provider.format_tool_result(tc, tr),
        ],
        model="gemini-2.5-flash",
    )
    call = fake_client.aio.models.calls[0]
    fr = call["contents"][-1].parts[0].function_response
    assert fr.name == "echo"
    assert fr.response == {"output": "just text from blocks"}
    assert not (fr.parts or [])
