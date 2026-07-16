"""Predefined error-handling rule presets for common providers.

Each preset is a plain ``list[ErrorRule]`` — import it, extend it, or write
your own from scratch. Pass it to a provider's ``error_handling`` parameter.

    from coreouto.contrib.error_presets import COMMON_HTTP_ERRORS
    from coreouto.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key=..., error_handling=COMMON_HTTP_ERRORS)

Compose and extend:

    my_rules = COMMON_HTTP_ERRORS + [
        ErrorRule(status_code=400, content_contains="context_length",
                  reaction="terminate", message="Context too long."),
    ]
"""

from __future__ import annotations

from coreouto._types import ErrorRule

COMMON_HTTP_ERRORS: list[ErrorRule] = [
    ErrorRule(
        status_code=429,
        reaction="retry",
        message="Rate limited — retrying with exponential backoff.",
        retry_after=1.0,
        retry_backoff=2.0,
        retry_max=5,
    ),
    ErrorRule(
        status_code=401,
        reaction="terminate",
        message="Authentication failed. Check your API key.",
    ),
    ErrorRule(
        status_code=403,
        reaction="terminate",
        message="Permission denied. Check your API key permissions.",
    ),
    ErrorRule(
        status_code=500,
        reaction="retry",
        message="Internal server error — retrying.",
        retry_after=2.0,
        retry_backoff=2.0,
        retry_max=3,
    ),
    ErrorRule(
        status_code=503,
        reaction="retry",
        message="Service unavailable — retrying.",
        retry_after=2.0,
        retry_backoff=2.0,
        retry_max=3,
    ),
]

INVALID_TOOL_ERRORS: list[ErrorRule] = [
    ErrorRule(
        status_code=400,
        content_contains="tool",
        reaction="tool_result",
        message="The tool call was rejected by the provider. Check the tool name and arguments.",
    ),
    ErrorRule(
        status_code=400,
        content_contains="invalid",
        reaction="tool_result",
        message="The request was invalid. Please check your tool arguments and format.",
    ),
]

__all__ = ["COMMON_HTTP_ERRORS", "INVALID_TOOL_ERRORS"]
