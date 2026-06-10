from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import pytest

from coreouto._types import LLMResponse, Message, ToolCall, ToolResult, Usage
from coreouto.tools import Tool


@dataclass
class FakeFunctionCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeFunctionResponse:
    name: str
    response: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakePart:
    text: str | None = None
    function_call: FakeFunctionCall | None = None
    function_response: FakeFunctionResponse | None = None

    @classmethod
    def from_text(cls, text: str) -> FakePart:
        return cls(text=text)

    @classmethod
    def from_function_call(cls, name: str, args: dict[str, Any]) -> FakePart:
        return cls(function_call=FakeFunctionCall(name=name, args=args))

    @classmethod
    def from_function_response(cls, name: str, response: dict[str, Any]) -> FakePart:
        return cls(function_response=FakeFunctionResponse(name=name, response=response))


@dataclass
class FakeContent:
    role: str
    parts: list[FakePart] = field(default_factory=list)


@dataclass
class FakeUsageMetadata:
    prompt_token_count: int
    candidates_token_count: int


@dataclass
class FakeCandidate:
    content: FakeContent


@dataclass
class FakeGenerateContentResponse:
    candidates: list[FakeCandidate]
    usage_metadata: FakeUsageMetadata | None = None

    @property
    def text(self) -> str | None:
        if self.candidates and self.candidates[0].content.parts:
            return self.candidates[0].content.parts[0].text
        return None

    @property
    def parts(self) -> list[FakePart]:
        if self.candidates:
            return self.candidates[0].content.parts
        return []


@dataclass
class FakeGenerativeModel:
    model_name: str
    system_instruction: str | None = None
    client_options: dict[str, Any] | None = None
    _response: FakeGenerateContentResponse | None = None

    def generate_content(
        self,
        contents: Any,
        *,
        tools: Any = None,
        **kwargs: Any,
    ) -> FakeGenerateContentResponse:
        self.last_contents = contents
        self.last_tools = tools
        self.last_kwargs = kwargs
        if self._response is None:
            return FakeGenerateContentResponse(
                candidates=[FakeCandidate(content=FakeContent(role="model", parts=[]))],
                usage_metadata=FakeUsageMetadata(prompt_token_count=0, candidates_token_count=0),
            )
        return self._response


class FakeGenaiModule:
    def __init__(self) -> None:
        self.configured_key: str | None = None
        self.Part = FakePart
        self.Content = FakeContent

    def configure(self, *, api_key: str) -> None:
        self.configured_key = api_key

    def GenerativeModel(  # noqa: N802
        self,
        model_name: str,
        *,
        system_instruction: str | None = None,
        client_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> FakeGenerativeModel:
        return FakeGenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
            client_options=client_options,
        )


@pytest.fixture
def fake_genai():
    fake = FakeGenaiModule()
    sys.modules["google"] = type(sys)("google")
    sys.modules["google.generativeai"] = fake
    if "coreouto.providers.google" in sys.modules:
        del sys.modules["coreouto.providers.google"]
    yield fake
    del sys.modules["google.generativeai"]
    del sys.modules["google"]


@pytest.fixture
def google_provider(fake_genai):
    from coreouto.providers.google import GoogleProvider

    return GoogleProvider()


@pytest.mark.asyncio
async def test_create_simple_user_message(google_provider, fake_genai):
    fake_genai.Part = FakePart
    fake_genai.Content = FakeContent

    response = FakeGenerateContentResponse(
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

    original_gm = fake_genai.GenerativeModel

    def _gm(model_name, *, system_instruction=None, **kwargs):
        m = original_gm(model_name, system_instruction=system_instruction, **kwargs)
        m._response = response
        return m

    fake_genai.GenerativeModel = _gm

    messages = [Message(role="user", content="Say hello")]
    result = await google_provider.create(messages=messages, model="gemini-pro")

    assert result.content == "Hello, world!"
    assert result.tool_calls == []
    assert result.usage is not None
    assert result.usage.prompt_tokens == 3
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 8
    assert result.raw is response


@pytest.mark.asyncio
async def test_create_tool_result_message(google_provider, fake_genai):
    fake_genai.Part = FakePart
    fake_genai.Content = FakeContent

    response = FakeGenerateContentResponse(
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

    original_gm = fake_genai.GenerativeModel

    def _gm(model_name, *, system_instruction=None, **kwargs):
        m = original_gm(model_name, system_instruction=system_instruction, **kwargs)
        m._response = response
        return m

    fake_genai.GenerativeModel = _gm

    messages = [
        Message(role="user", content="What is 2+2?"),
        Message(
            role="tool",
            content="4",
            tool_call_id="call_123",
            name="calculator",
        ),
    ]
    result = await google_provider.create(messages=messages, model="gemini-pro")

    assert result.content == "Got it."


@pytest.mark.asyncio
async def test_create_assistant_with_tool_calls(google_provider, fake_genai):
    fake_genai.Part = FakePart
    fake_genai.Content = FakeContent

    response = FakeGenerateContentResponse(
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

    original_gm = fake_genai.GenerativeModel

    def _gm(model_name, *, system_instruction=None, **kwargs):
        m = original_gm(model_name, system_instruction=system_instruction, **kwargs)
        m._response = response
        return m

    fake_genai.GenerativeModel = _gm

    messages = [
        Message(
            role="assistant",
            content="Let me search.",
            tool_calls=[ToolCall(id="call_abc", name="search", arguments={"query": "cats"})],
        )
    ]
    result = await google_provider.create(messages=messages, model="gemini-pro")

    assert result.content is None or result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].arguments == {"query": "cats"}
    assert result.tool_calls[0].id.startswith("call_")


@pytest.mark.asyncio
async def test_create_system_prompt(google_provider, fake_genai):
    fake_genai.Part = FakePart
    fake_genai.Content = FakeContent

    created_models = []
    original_gm = fake_genai.GenerativeModel

    def _gm(model_name, *, system_instruction=None, **kwargs):
        created_models.append(system_instruction)
        m = original_gm(model_name, system_instruction=system_instruction, **kwargs)
        m._response = FakeGenerateContentResponse(
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
        return m

    fake_genai.GenerativeModel = _gm

    messages = [Message(role="user", content="Hi")]
    await google_provider.create(
        messages=messages,
        model="gemini-pro",
        system_prompt="You are a tester.",
    )

    assert created_models == ["You are a tester."]


@pytest.mark.asyncio
async def test_create_system_role_in_messages(google_provider, fake_genai):
    fake_genai.Part = FakePart
    fake_genai.Content = FakeContent

    created_models = []
    original_gm = fake_genai.GenerativeModel

    def _gm(model_name, *, system_instruction=None, **kwargs):
        created_models.append(system_instruction)
        m = original_gm(model_name, system_instruction=system_instruction, **kwargs)
        m._response = FakeGenerateContentResponse(
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
        return m

    fake_genai.GenerativeModel = _gm

    messages = [
        Message(role="system", content="Be concise."),
        Message(role="system", content="Use JSON."),
        Message(role="user", content="Hi"),
    ]
    await google_provider.create(messages=messages, model="gemini-pro")

    assert created_models == ["Be concise.\nUse JSON."]


@pytest.mark.asyncio
async def test_create_tools_sent(google_provider, fake_genai):
    fake_genai.Part = FakePart
    fake_genai.Content = FakeContent

    original_gm = fake_genai.GenerativeModel

    def _gm(model_name, *, system_instruction=None, **kwargs):
        m = original_gm(model_name, system_instruction=system_instruction, **kwargs)
        m._response = FakeGenerateContentResponse(
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
        return m

    fake_genai.GenerativeModel = _gm

    tools = [
        Tool(
            name="search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=None,
        )
    ]
    messages = [Message(role="user", content="Search for cats")]
    result = await google_provider.create(messages=messages, model="gemini-pro", tools=tools)

    assert result.content == "OK"


@pytest.mark.asyncio
async def test_create_function_call_response_parsed(google_provider, fake_genai):
    fake_genai.Part = FakePart
    fake_genai.Content = FakeContent

    response = FakeGenerateContentResponse(
        candidates=[
            FakeCandidate(
                content=FakeContent(
                    role="model",
                    parts=[
                        FakePart.from_function_call(name="get_weather", args={"location": "NYC"}),
                        FakePart.from_function_call(name="get_time", args={"zone": "EST"}),
                    ],
                )
            )
        ],
        usage_metadata=FakeUsageMetadata(prompt_token_count=5, candidates_token_count=7),
    )

    original_gm = fake_genai.GenerativeModel

    def _gm(model_name, *, system_instruction=None, **kwargs):
        m = original_gm(model_name, system_instruction=system_instruction, **kwargs)
        m._response = response
        return m

    fake_genai.GenerativeModel = _gm

    messages = [Message(role="user", content="What's the weather and time?")]
    result = await google_provider.create(messages=messages, model="gemini-pro")

    assert result.content is None or result.content == ""
    assert len(result.tool_calls) == 2
    names = [tc.name for tc in result.tool_calls]
    assert "get_weather" in names
    assert "get_time" in names
    for tc in result.tool_calls:
        assert tc.id.startswith("call_")
        assert len(tc.id) > len("call_")


def test_format_assistant_message_text_only(google_provider):
    response = LLMResponse(
        content="Hello!",
        usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    msg = google_provider.format_assistant_message(response)

    assert msg.role == "assistant"
    assert msg.content == "Hello!"
    assert msg.tool_calls is None or msg.tool_calls == []


def test_format_assistant_message_with_tool_calls(google_provider):
    response = LLMResponse(
        content="Let me help.",
        tool_calls=[ToolCall(id="call_abc", name="search", arguments={"q": "x"})],
        usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    msg = google_provider.format_assistant_message(response)

    assert msg.role == "assistant"
    assert msg.content == "Let me help."
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].id == "call_abc"
    assert msg.tool_calls[0].name == "search"


def test_format_tool_result(google_provider):
    tool_call = ToolCall(id="call_123", name="calc", arguments={"a": 1})
    result = ToolResult(tool_call_id="call_123", content="42")
    msg = google_provider.format_tool_result(tool_call, result)

    assert msg.role == "tool"
    assert msg.content == "42"
    assert msg.tool_call_id == "call_123"
    assert msg.name == "calc"


def test_import_error_without_sdk():
    mod_name = "coreouto.providers.google"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    had_google = "google.generativeai" in sys.modules
    had_google_pkg = "google" in sys.modules
    if had_google:
        old_google_genai = sys.modules.pop("google.generativeai")
    if had_google_pkg:
        old_google = sys.modules.pop("google")

    try:
        with pytest.raises(ImportError, match=r"pip install coreouto\[google\]"):
            from coreouto.providers.google import GoogleProvider

            _ = GoogleProvider()
    finally:
        if had_google:
            sys.modules["google.generativeai"] = old_google_genai
        if had_google_pkg:
            sys.modules["google"] = old_google


def test_api_key_configured(fake_genai):
    from coreouto.providers.google import GoogleProvider

    GoogleProvider(api_key="secret-key")
    assert fake_genai.configured_key == "secret-key"


def test_model_stored(fake_genai):
    from coreouto.providers.google import GoogleProvider

    provider = GoogleProvider(model="gemini-ultra")
    assert provider._model_name == "gemini-ultra"


@pytest.mark.asyncio
async def test_create_uses_stored_model(google_provider, fake_genai):
    from coreouto.providers.google import GoogleProvider

    fake_genai.Part = FakePart
    fake_genai.Content = FakeContent

    created_with = []
    original_gm = fake_genai.GenerativeModel

    def _gm(model_name, *, system_instruction=None, **kwargs):
        created_with.append(model_name)
        m = original_gm(model_name, system_instruction=system_instruction, **kwargs)
        m._response = FakeGenerateContentResponse(
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
        return m

    fake_genai.GenerativeModel = _gm

    provider = GoogleProvider(model="gemini-ultra")
    messages = [Message(role="user", content="Hi")]
    await provider.create(messages=messages, model="gemini-ultra")

    assert created_with == ["gemini-ultra"]


@pytest.mark.asyncio
async def test_create_mixed_text_and_function_call(google_provider, fake_genai):
    fake_genai.Part = FakePart
    fake_genai.Content = FakeContent

    response = FakeGenerateContentResponse(
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

    original_gm = fake_genai.GenerativeModel

    def _gm(model_name, *, system_instruction=None, **kwargs):
        m = original_gm(model_name, system_instruction=system_instruction, **kwargs)
        m._response = response
        return m

    fake_genai.GenerativeModel = _gm

    messages = [Message(role="user", content="Tell me about cats")]
    result = await google_provider.create(messages=messages, model="gemini-pro")

    assert result.content == "I'll search for you."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search"


def test_register(fake_genai):
    from coreouto.providers import clear_providers, get_provider
    from coreouto.providers.google import register

    clear_providers()
    register(api_key="my-key", name="google-test")
    provider = get_provider("google-test")
    assert provider is not None
    clear_providers()


def test_module_level_provider_instance(fake_genai):
    from coreouto.providers.google import provider

    assert provider is not None


def test_google_provider_accepts_client_options_construction(fake_genai):
    from coreouto.providers.google import GoogleProvider

    provider = GoogleProvider(
        api_key="test",
        client_options={"api_endpoint": "https://proxy.example.com"},
    )
    assert provider._client_options == {"api_endpoint": "https://proxy.example.com"}


@pytest.mark.asyncio
async def test_google_provider_passes_client_options_to_model(fake_genai):
    from coreouto.providers.google import GoogleProvider

    fake_genai.Part = FakePart
    fake_genai.Content = FakeContent

    created_models: list[FakeGenerativeModel] = []
    original_gm = fake_genai.GenerativeModel

    def _gm(model_name, *, system_instruction=None, client_options=None, **kwargs):
        m = original_gm(
            model_name,
            system_instruction=system_instruction,
            client_options=client_options,
        )
        m._response = FakeGenerateContentResponse(
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
        created_models.append(m)
        return m

    fake_genai.GenerativeModel = _gm

    provider = GoogleProvider(
        api_key="test",
        client_options={"api_endpoint": "https://proxy.example.com"},
    )
    messages = [Message(role="user", content="Hi")]
    await provider.create(messages=messages, model="gemini-pro")

    assert len(created_models) >= 1
    assert created_models[0].client_options == {"api_endpoint": "https://proxy.example.com"}


def test_google_provider_works_without_client_options(fake_genai):
    from coreouto.providers.google import GoogleProvider

    provider = GoogleProvider(api_key="test")
    assert provider._client_options is None


def test_google_register_function_accepts_client_options(fake_genai):
    from coreouto.providers import clear_providers, get_provider
    from coreouto.providers.google import register

    clear_providers()
    register(
        api_key="x",
        client_options={"api_endpoint": "https://proxy.example.com"},
        name="google-proxy",
    )
    provider = get_provider("google-proxy")
    assert provider is not None
    assert provider._client_options == {"api_endpoint": "https://proxy.example.com"}
    clear_providers()
