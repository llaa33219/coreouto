"""Anthropic provider error handling — comprehensive per-error-type rules.

The anthropic Python SDK raises typed exceptions:

  anthropic.BadRequestError           400  invalid request
  anthropic.AuthenticationError       401  bad API key
  anthropic.PermissionDeniedError     403  key lacks permissions
  anthropic.NotFoundError             404  model not found
  anthropic.ConflictError             409  version conflict
  anthropic.RequestTooLargeError      413  request body too large
  anthropic.UnprocessableEntityError  422  schema validation failure
  anthropic.RateLimitError            429  rate limited
  anthropic.InternalServerError       5xx  server-side fault
  anthropic.ServiceUnavailableError   503  service down
  anthropic.DeadlineExceededError     504  server-side timeout
  anthropic.OverloadedError           529  Anthropic-specific overload

Key differences from OpenAI:
  - 413 RequestTooLargeError: Anthropic enforces request size limits.
  - 529 OverloadedError: Anthropic-specific, indicates capacity issues.
    Worth retrying with backoff.
  - 503/504: distinct classes, both retried.

Run: python examples/19_anthropic_errors.py
"""

import coreouto as co
from coreouto import ErrorRule
from coreouto.providers.anthropic import AnthropicProvider

# ---------------------------------------------------------------------------
# Full rule set tuned for Anthropic's exception types.
# ---------------------------------------------------------------------------

ANTHROPIC_ERROR_RULES: list[ErrorRule] = [
    # --- Retryable: rate limit, overload, server errors ----------------
    ErrorRule(
        status_code=429,
        reaction="retry",
        message="Anthropic rate limit — retrying.",
        retry_after=1.0,
        retry_backoff=2.0,
        retry_max=5,
    ),
    ErrorRule(
        status_code=529,
        reaction="retry",
        message="Anthropic overloaded — retrying with longer backoff.",
        retry_after=5.0,
        retry_backoff=2.0,
        retry_max=3,
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
    # --- Terminal: auth, permissions ------------------------------------
    ErrorRule(
        status_code=401,
        reaction="terminate",
        message="Authentication failed. Check your Anthropic API key.",
    ),
    ErrorRule(
        status_code=403,
        reaction="terminate",
        message="Permission denied. Your API key may not have access to this model.",
    ),
    # --- 400: same code, different content → different reactions --------
    # Context window: Anthropic returns this when input + max_tokens exceeds
    # the model's context limit. Not recoverable mid-conversation.
    ErrorRule(
        status_code=400,
        content_contains="context",
        reaction="terminate",
        message="Context too long. Reduce conversation history or lower max_tokens.",
    ),
    # Invalid tool definition or arguments — let the model self-correct.
    ErrorRule(
        status_code=400,
        content_contains="tool",
        reaction="tool_result",
        message="Invalid tool call. Check the tool name and argument schema.",
    ),
    # --- 413: request too large -----------------------------------------
    ErrorRule(
        status_code=413,
        reaction="terminate",
        message="Request body too large. Reduce the number of messages or content size.",
    ),
    # --- 422: schema validation -----------------------------------------
    ErrorRule(
        status_code=422,
        reaction="tool_result",
        message="Schema validation failed. The tool parameters don't match the declared input_schema.",
    ),
]


provider = AnthropicProvider(
    api_key="sk-ant-...",
    error_handling=ANTHROPIC_ERROR_RULES,
)
co.register_provider("anthropic-resilient", provider)

config = co.AgentConfig(
    name="resilient-agent",
    model="claude-sonnet-4-6",
    provider="anthropic-resilient",
)


def log_anthropic_errors(*, status_code, error_message, reaction, reaction_message, **kwargs):
    print(f"[Anthropic] HTTP {status_code}: {error_message[:80]}")
    print(f"  -> {reaction}: {reaction_message[:80]}")


co.register_hook(co.ON_PROVIDER_ERROR, log_anthropic_errors)


if __name__ == "__main__":
    print(f"ANTHROPIC_ERROR_RULES: {len(ANTHROPIC_ERROR_RULES)} rules")
    print()
    for rule in ANTHROPIC_ERROR_RULES:
        match = f"{rule.status_code}"
        if rule.content_contains:
            match += f" + '{rule.content_contains}'"
        extra = ""
        if rule.reaction == "retry":
            extra = (
                f" (after={rule.retry_after}s, backoff={rule.retry_backoff}x, max={rule.retry_max})"
            )
        print(f"  {match:>20s} -> {rule.reaction}{extra}")
