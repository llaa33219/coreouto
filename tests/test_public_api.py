import coreouto

_EXPECTED_NAMES = [
    "AFTER_LLM_CALL",
    "AFTER_TOOL_CALL",
    "Agent",
    "AgentConfig",
    "AgentPreset",
    "AudioBlock",
    "BEFORE_LLM_CALL",
    "BEFORE_TOOL_CALL",
    "CANONICAL_SETTINGS",
    "ContentBlock",
    "DocumentBlock",
    "ImageBlock",
    "LLMResponse",
    "MaxIterationsError",
    "Message",
    "ON_FINISH",
    "ON_ITERATION",
    "ON_RETRY",
    "ON_STREAM_TEXT",
    "ON_STREAM_THINKING",
    "ON_THINKING",
    "ON_USER_INJECTION",
    "RESERVED_TOOL_NAMES",
    "Response",
    "StopReason",
    "TextBlock",
    "Tool",
    "ToolCall",
    "ToolResult",
    "Usage",
    "VideoBlock",
    "__version__",
    "agent_as_tool",
    "available_providers",
    "call_sync",
    "clear_agent_presets",
    "clear_hooks",
    "clear_providers",
    "clear_tools",
    "get_agent_preset",
    "get_hooks",
    "get_provider",
    "get_tool",
    "list_agent_presets",
    "list_tools",
    "make_delegate_tool",
    "normalize_provider_config",
    "register_agent_preset",
    "register_hook",
    "register_provider",
    "register_tool",
    "register_tool_class",
]


def test_all_expected_names_exported():
    for name in _EXPECTED_NAMES:
        assert hasattr(coreouto, name), f"coreouto missing export: {name!r}"


def test_all_matches_dunder():
    assert sorted(coreouto.__all__) == sorted(_EXPECTED_NAMES)


def test_version_is_string():
    assert isinstance(coreouto.__version__, str)


def test_agent_is_class():
    assert isinstance(coreouto.Agent, type)


def test_message_is_class():
    assert isinstance(coreouto.Message, type)


def test_max_iterations_error_is_exception():
    assert issubclass(coreouto.MaxIterationsError, Exception)


def test_register_tool_is_callable():
    assert callable(coreouto.register_tool)


def test_call_sync_is_callable():
    assert callable(coreouto.call_sync)


def test_hook_constants_are_strings():
    for const in (
        coreouto.BEFORE_LLM_CALL,
        coreouto.AFTER_LLM_CALL,
        coreouto.BEFORE_TOOL_CALL,
        coreouto.AFTER_TOOL_CALL,
        coreouto.ON_ITERATION,
        coreouto.ON_FINISH,
        coreouto.ON_USER_INJECTION,
        coreouto.ON_RETRY,
    ):
        assert isinstance(const, str)


def test_make_delegate_tool_is_callable():
    assert callable(coreouto.make_delegate_tool)
