"""Provider error handling: overview and preset usage.

coreouto does not retry or classify provider errors by default. When
`provider.create()` raises, the exception propagates to the caller.

To handle errors, pass `error_handling` (a list of `ErrorRule`) to a
provider constructor. Rules match by HTTP status code and/or error content,
then react: retry, terminate, inject a user message, or return as a tool
result.

The `error_handling` parameter is optional (defaults to None -> propagate).

For per-provider details with comprehensive rule lists, see:
  examples/18_openai_errors.py
  examples/19_anthropic_errors.py
  examples/20_google_errors.py

Run: python examples/17_error_handling.py
"""

import coreouto as co
from coreouto import ErrorRule
from coreouto.contrib.error_presets import COMMON_HTTP_ERRORS, INVALID_TOOL_ERRORS
from coreouto.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    api_key="sk-...",
    error_handling=COMMON_HTTP_ERRORS,
)
co.register_provider("openai-defaults", provider)

custom_rules = (
    COMMON_HTTP_ERRORS
    + INVALID_TOOL_ERRORS
    + [
        ErrorRule(
            status_code=400,
            content_contains="context_length",
            reaction="terminate",
            message="Context too long. Can't recover.",
        ),
        ErrorRule(
            status_code=403,
            content_contains="rate",
            reaction="retry",
            message="Proxy rate-limiting via 403. Retrying.",
            retry_after=2.0,
            retry_backoff=2.0,
            retry_max=3,
        ),
    ]
)

provider_custom = OpenAIProvider(
    api_key="sk-...",
    error_handling=custom_rules,
)
co.register_provider("openai-custom", provider_custom)


def log_error(*, status_code, error_message, reaction, reaction_message, **kwargs):
    print(f"HTTP {status_code}: {error_message}")
    print(f"  -> {reaction}: {reaction_message}")


co.register_hook(co.ON_PROVIDER_ERROR, log_error)


if __name__ == "__main__":
    print("COMMON_HTTP_ERRORS:")
    for rule in COMMON_HTTP_ERRORS:
        print(f"  {rule.status_code} -> {rule.reaction}")

    print("\nINVALID_TOOL_ERRORS:")
    for rule in INVALID_TOOL_ERRORS:
        print(f"  {rule.status_code} + '{rule.content_contains}' -> {rule.reaction}")

    print(f"\nCustom rules: {len(custom_rules)} total")
    print("\nFor per-provider details, see examples/18-20.")
