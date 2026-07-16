"""Google GenAI provider error handling — comprehensive per-error-type rules.

The google-genai SDK has a MUCH simpler exception hierarchy than OpenAI or
Anthropic — only two HTTP-error subclasses:

  errors.ClientError   any 4xx (400-499)
  errors.ServerError   any 5xx (500-599)

There are no per-status-code subclasses. To distinguish a 429 from a 401,
you must read `.code` (int HTTP status) and `.status` (Google API enum
string) off the exception object.

Key .status values for ClientError:
  "RESOURCE_EXHAUSTED"   429  rate limit
  "UNAUTHENTICATED"      401  no/invalid credentials
  "PERMISSION_DENIED"    403  key lacks access
  "INVALID_ARGUMENT"     400  malformed request / bad tool args
  "NOT_FOUND"            404  model not found
  "FAILED_PRECONDITION"  400  request violates API constraints

Since coreouto's ErrorRule matches by `status_code` (int) and
`content_contains` (string), and the google-genai SDK exposes `.code` as
the int HTTP status, status_code matching works directly.

Run: python examples/20_google_errors.py
"""

import coreouto as co
from coreouto import ErrorRule
from coreouto.providers.google import GoogleProvider

# ---------------------------------------------------------------------------
# Full rule set tuned for Google GenAI's error model.
#
# Note: Google collapses all 4xx into ClientError and all 5xx into
# ServerError. We distinguish by status_code (the .code int) and
# content_contains (matching the .status enum string or the error message).
# ---------------------------------------------------------------------------

GOOGLE_ERROR_RULES: list[ErrorRule] = [
    # --- Retryable: rate limit, server errors ---------------------------
    # 429 is the most common retriable error. Google's quota system can
    # be aggressive — use a longer initial backoff.
    ErrorRule(
        status_code=429,
        reaction="retry",
        message="Google API quota exceeded — retrying.",
        retry_after=2.0,
        retry_backoff=2.0,
        retry_max=5,
    ),
    # 5xx server errors — transient, retry.
    ErrorRule(
        status_code=500,
        reaction="retry",
        message="Google internal error — retrying.",
        retry_after=2.0,
        retry_backoff=2.0,
        retry_max=3,
    ),
    ErrorRule(
        status_code=503,
        reaction="retry",
        message="Google service unavailable — retrying.",
        retry_after=2.0,
        retry_backoff=2.0,
        retry_max=3,
    ),
    # --- Terminal: auth, permissions ------------------------------------
    ErrorRule(
        status_code=401,
        reaction="terminate",
        message="Google API authentication failed. Check your API key or OAuth token.",
    ),
    ErrorRule(
        status_code=403,
        reaction="terminate",
        message="Permission denied. Your Google Cloud project may lack access to this model.",
    ),
    # --- 400: same code, different content → different reactions --------
    # Google's 400 can mean many things. Match by .status enum in the
    # error message to distinguish.
    # INVALID_ARGUMENT with "tool" in the message — bad tool call.
    # Feed back as tool result so the model self-corrects.
    ErrorRule(
        status_code=400,
        content_contains="tool",
        reaction="tool_result",
        message="Invalid tool call. Check function name and argument schema.",
    ),
    # INVALID_ARGUMENT with "safety" — content was filtered.
    # Can't recover by retrying; tell the caller.
    ErrorRule(
        status_code=400,
        content_contains="safety",
        reaction="terminate",
        message="Request rejected by Google's safety filter. Adjust your input.",
    ),
    # FAILED_PRECONDITION — e.g. trying to use a feature the model doesn't
    # support (like thinking on a non-thinking model).
    ErrorRule(
        status_code=400,
        content_contains="precondition",
        reaction="terminate",
        message="Request violates a Google API precondition. Check model capabilities.",
    ),
    # Generic 400 fallback — let the model try to fix it.
    ErrorRule(
        status_code=400,
        reaction="tool_result",
        message="Invalid request (Google API). Check your tool arguments and request format.",
    ),
    # --- 404: model not found -------------------------------------------
    ErrorRule(
        status_code=404,
        reaction="terminate",
        message="Model not found. Check the model name (e.g. 'gemini-2.0-flash').",
    ),
]


provider = GoogleProvider(
    api_key="AIza...",
    error_handling=GOOGLE_ERROR_RULES,
)
co.register_provider("google-resilient", provider)

config = co.AgentConfig(
    name="resilient-agent",
    model="gemini-2.0-flash",
    provider="google-resilient",
)


def log_google_errors(*, status_code, error_message, reaction, reaction_message, **kwargs):
    print(f"[Google] HTTP {status_code}: {error_message[:80]}")
    print(f"  -> {reaction}: {reaction_message[:80]}")


co.register_hook(co.ON_PROVIDER_ERROR, log_google_errors)


if __name__ == "__main__":
    print(f"GOOGLE_ERROR_RULES: {len(GOOGLE_ERROR_RULES)} rules")
    print()
    for rule in GOOGLE_ERROR_RULES:
        match = f"{rule.status_code}"
        if rule.content_contains:
            match += f" + '{rule.content_contains}'"
        extra = ""
        if rule.reaction == "retry":
            extra = (
                f" (after={rule.retry_after}s, backoff={rule.retry_backoff}x, max={rule.retry_max})"
            )
        print(f"  {match:>20s} -> {rule.reaction}{extra}")
