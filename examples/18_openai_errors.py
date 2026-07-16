"""OpenAI provider error handling — comprehensive per-error-type rules.

The openai Python SDK raises typed exceptions for each HTTP status:

  openai.BadRequestError           400  invalid request, bad tool args
  openai.AuthenticationError       401  bad API key (includes OAuthError)
  openai.PermissionDeniedError     403  key lacks permissions
  openai.NotFoundError             404  model/endpoint not found
  openai.ConflictError             409  version conflict
  openai.UnprocessableEntityError  422  schema validation failure
  openai.RateLimitError            429  rate limited
  openai.InternalServerError       5xx  server-side fault
  openai.APIConnectionError        —    network/DNS (no status_code)
  openai.APITimeoutError           —    request timed out (no status_code)

All status-code exceptions expose `.status_code: int`.
Network exceptions have NO `.status_code` — they won't match any
`status_code` rule. Match them by `content_contains` instead, or let them
propagate (the loop catches them but no rule matches → unhandled).

Run: python examples/18_openai_errors.py
"""

import coreouto as co
from coreouto import ErrorRule
from coreouto.contrib.error_presets import COMMON_HTTP_ERRORS
from coreouto.providers.openai import OpenAIProvider

# ---------------------------------------------------------------------------
# Full rule set tuned for OpenAI's exception types.
#
# Start from the shared preset, then add OpenAI-specific cases.
# ---------------------------------------------------------------------------

OPENAI_ERROR_RULES: list[ErrorRule] = [
    *COMMON_HTTP_ERRORS,
    # --- 400: same code, different content → different reactions ----------
    # Context window exceeded — can't recover, tell the caller.
    ErrorRule(
        status_code=400,
        content_contains="context_length_exceeded",
        reaction="terminate",
        message="Context window exceeded. Reduce the conversation length or clear history.",
    ),
    # The model called a tool with invalid JSON arguments. Feed the error
    # back as a tool result so the model can fix its arguments and retry.
    ErrorRule(
        status_code=400,
        content_contains="invalid_schema",
        reaction="tool_result",
        message="Tool arguments failed schema validation. Check parameter types and required fields.",
    ),
    # The model referenced a tool that doesn't exist.
    ErrorRule(
        status_code=400,
        content_contains="tool",
        reaction="tool_result",
        message="Invalid tool call. Verify the tool name exists and try again.",
    ),
    # --- 404: model not found -----------------------------------------
    ErrorRule(
        status_code=404,
        content_contains="model",
        reaction="terminate",
        message="Model not found. Check the model name in your config.",
    ),
    # --- 422: schema mismatch -----------------------------------------
    ErrorRule(
        status_code=422,
        reaction="tool_result",
        message="Request schema validation failed. The tool parameters don't match the declared schema.",
    ),
]


# ---------------------------------------------------------------------------
# Register and use
# ---------------------------------------------------------------------------

provider = OpenAIProvider(
    api_key="sk-...",
    error_handling=OPENAI_ERROR_RULES,
)
co.register_provider("openai-resilient", provider)

config = co.AgentConfig(
    name="resilient-agent",
    model="gpt-4o",
    provider="openai-resilient",
)


# ---------------------------------------------------------------------------
# Observe errors
# ---------------------------------------------------------------------------


def log_openai_errors(*, status_code, error_message, reaction, reaction_message, **kwargs):
    print(f"[OpenAI] HTTP {status_code}: {error_message[:80]}")
    print(f"  -> {reaction}: {reaction_message[:80]}")


co.register_hook(co.ON_PROVIDER_ERROR, log_openai_errors)


if __name__ == "__main__":
    print(f"OPENAI_ERROR_RULES: {len(OPENAI_ERROR_RULES)} rules")
    print(f"  base preset: {len(COMMON_HTTP_ERRORS)} rules")
    print(f"  OpenAI-specific: {len(OPENAI_ERROR_RULES) - len(COMMON_HTTP_ERRORS)} rules")
    print()
    for rule in OPENAI_ERROR_RULES:
        match = f"{rule.status_code}"
        if rule.content_contains:
            match += f" + '{rule.content_contains}'"
        extra = ""
        if rule.reaction == "retry":
            extra = (
                f" (after={rule.retry_after}s, backoff={rule.retry_backoff}x, max={rule.retry_max})"
            )
        print(f"  {match:>20s} -> {rule.reaction}{extra}")
