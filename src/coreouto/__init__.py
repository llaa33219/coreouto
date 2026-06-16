"""coreouto — a minimal, extensible Python agent library.

See README.md and docs/ for usage. The public API is re-exported here.
"""

from coreouto._types import (
    AgentConfig,
    AudioBlock,
    ContentBlock,
    DocumentBlock,
    ImageBlock,
    LLMResponse,
    Message,
    Response,
    TextBlock,
    ToolCall,
    ToolResult,
    Usage,
    VideoBlock,
)
from coreouto._version import __version__
from coreouto.agent import Agent, MaxIterationsError
from coreouto.hooks import (
    AFTER_LLM_CALL,
    AFTER_TOOL_CALL,
    BEFORE_LLM_CALL,
    BEFORE_TOOL_CALL,
    ON_FINISH,
    ON_ITERATION,
    ON_USER_INJECTION,
    clear_hooks,
    get_hooks,
    register_hook,
)
from coreouto.multi_agent import agent_as_tool, make_delegate_tool
from coreouto.presets import (
    AgentPreset,
    clear_agent_presets,
    get_agent_preset,
    list_agent_presets,
    register_agent_preset,
)
from coreouto.providers import (
    available_providers,
    clear_providers,
    get_provider,
    register_provider,
)
from coreouto.settings import CANONICAL_SETTINGS, normalize_provider_config
from coreouto.sync import call_sync
from coreouto.tools import (
    RESERVED_TOOL_NAMES,
    Tool,
    clear_tools,
    get_tool,
    list_tools,
    register_tool,
    register_tool_class,
)

__all__ = [
    "AFTER_LLM_CALL",
    "AFTER_TOOL_CALL",
    "BEFORE_LLM_CALL",
    "BEFORE_TOOL_CALL",
    "CANONICAL_SETTINGS",
    "ON_FINISH",
    "ON_ITERATION",
    "ON_USER_INJECTION",
    "RESERVED_TOOL_NAMES",
    "Agent",
    "AgentConfig",
    "AgentPreset",
    "AudioBlock",
    "ContentBlock",
    "DocumentBlock",
    "ImageBlock",
    "LLMResponse",
    "MaxIterationsError",
    "Message",
    "Response",
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
