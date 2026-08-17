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

# "tool_result" is the right reaction only when the 400 comes from a
# *dangling* tool call (a call with no result in history — the appended
# error result completes the pair). When the 400 comes from a *malformed*
# tool call, the offending assistant message stays in history, so every
# retry 400s again and duplicate results pile up. For that shape, pair a
# "retry" rule with a history-repair hook — see
# examples/29_malformed_tool_call.py.
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

# Timeout exceptions carry no HTTP status code and their messages vary by
# SDK, so these rules match by exception class name (`exc_type`) instead:
#   - "APITimeoutError": openai and anthropic SDKs
#   - "TimeoutError": builtin / asyncio / concurrent.futures timeouts
#   - "TimeoutException": httpx timeouts (google-genai's transport)
# Pair with a provider-level `timeout` (seconds) so the SDK actually raises
# instead of hanging forever.
TIMEOUT_ERRORS: list[ErrorRule] = [
    ErrorRule(
        exc_type="APITimeoutError",
        reaction="retry",
        message="API request timed out — retrying.",
        retry_after=1.0,
        retry_backoff=2.0,
        retry_max=3,
    ),
    ErrorRule(
        exc_type="TimeoutError",
        reaction="retry",
        message="API request timed out — retrying.",
        retry_after=1.0,
        retry_backoff=2.0,
        retry_max=3,
    ),
    ErrorRule(
        exc_type="TimeoutException",
        reaction="retry",
        message="API request timed out — retrying.",
        retry_after=1.0,
        retry_backoff=2.0,
        retry_max=3,
    ),
]

__all__ = ["COMMON_HTTP_ERRORS", "INVALID_TOOL_ERRORS", "TIMEOUT_ERRORS"]
