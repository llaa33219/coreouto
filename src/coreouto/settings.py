from __future__ import annotations

from typing import Any

SETTING_TEMPERATURE = "temperature"
SETTING_TOP_P = "top_p"
SETTING_MAX_TOKENS = "max_tokens"
SETTING_TOP_K = "top_k"
SETTING_STOP = "stop"
SETTING_SEED = "seed"
SETTING_METADATA = "metadata"
SETTING_REASONING_EFFORT = "reasoning_effort"

CANONICAL_SETTINGS = frozenset(
    {
        SETTING_TEMPERATURE,
        SETTING_TOP_P,
        SETTING_MAX_TOKENS,
        SETTING_TOP_K,
        SETTING_STOP,
        SETTING_SEED,
        SETTING_METADATA,
        SETTING_REASONING_EFFORT,
    }
)

_PROVIDER_MAP: dict[str, dict[str, str | None]] = {
    SETTING_TEMPERATURE: {
        "openai": "temperature",
        "openai-response": "temperature",
        "anthropic": "temperature",
        "google": "temperature",
    },
    SETTING_TOP_P: {
        "openai": "top_p",
        "openai-response": "top_p",
        "anthropic": "top_p",
        "google": "top_p",
    },
    SETTING_MAX_TOKENS: {
        "openai": "max_tokens",
        "openai-response": "max_output_tokens",
        "anthropic": "max_tokens",
        "google": "max_output_tokens",
    },
    SETTING_TOP_K: {
        "anthropic": "top_k",
        "google": "top_k",
    },
    SETTING_STOP: {
        "openai": "stop",
        "anthropic": "stop_sequences",
        "google": "stop_sequences",
    },
    SETTING_SEED: {
        "openai": "seed",
    },
    SETTING_METADATA: {
        "anthropic": "metadata",
    },
    SETTING_REASONING_EFFORT: {
        "openai-response": None,
    },
}


def _supported_providers_for_setting(setting: str) -> list[str]:
    mapping = _PROVIDER_MAP.get(setting, {})
    return sorted(mapping)


def normalize_provider_config(provider_name: str, config: dict[str, Any]) -> dict[str, Any]:
    """Translate canonical settings in `config` to provider-specific kwargs.

    - Keys in the canonical 8 → translated to the provider's kwarg name.
    - Standard keys whose value is None → dropped (don't send to SDK).
    - Standard keys not supported by `provider_name` → raise `ValueError` with a helpful
      message naming the key, the provider, and the supported providers (if any).
    - Non-canonical keys → raise `ValueError` listing the unknown key and the
      canonical 8 (suggesting the user move it to `provider_passthrough`).
    - For `reasoning_effort` with `openai-response`, the value is wrapped as
      `{"reasoning": {"effort": value}}` per the OpenAI Responses API.
    """
    known_providers = {"openai", "openai-response", "anthropic", "google"}
    is_known = provider_name in known_providers

    result: dict[str, Any] = {}
    for key, value in config.items():
        if value is None:
            continue

        if key not in CANONICAL_SETTINGS:
            allowed = ", ".join(sorted(CANONICAL_SETTINGS))
            raise ValueError(
                f"unknown provider_config key '{key}'. "
                f"Allowed canonical keys: {allowed} — "
                f"use provider_passthrough for non-canonical settings."
            )

        if not is_known:
            result[key] = value
            continue

        mapping = _PROVIDER_MAP[key]
        if provider_name not in mapping:
            supported = _supported_providers_for_setting(key)
            supported_str = ", ".join(supported) if supported else "none"
            raise ValueError(
                f"setting '{key}' is not supported by provider '{provider_name}'. "
                f"Supported providers: {supported_str}"
            )

        provider_key = mapping[provider_name]
        if key == SETTING_REASONING_EFFORT and provider_name == "openai-response":
            result["reasoning"] = {"effort": value}
        else:
            result[provider_key] = value

    return result
