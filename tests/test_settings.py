from __future__ import annotations

import pytest

from coreouto.settings import (
    CANONICAL_SETTINGS,
    SETTING_MAX_TOKENS,
    SETTING_METADATA,
    SETTING_REASONING_EFFORT,
    SETTING_SEED,
    SETTING_STOP,
    SETTING_TEMPERATURE,
    SETTING_TOP_K,
    SETTING_TOP_P,
    normalize_provider_config,
)


def test_temperature_translates_to_all_providers():
    for provider in ("openai", "openai-response", "anthropic", "google"):
        result = normalize_provider_config(provider, {"temperature": 0.5})
        assert result == {"temperature": 0.5}


def test_max_tokens_translates_with_provider_specific_name():
    assert normalize_provider_config("openai", {"max_tokens": 100}) == {"max_tokens": 100}
    assert normalize_provider_config("openai-response", {"max_tokens": 100}) == {
        "max_output_tokens": 100
    }
    assert normalize_provider_config("anthropic", {"max_tokens": 100}) == {"max_tokens": 100}
    assert normalize_provider_config("google", {"max_tokens": 100}) == {"max_output_tokens": 100}


def test_top_k_unsupported_for_openai_raises():
    with pytest.raises(
        ValueError, match=r"setting 'top_k' is not supported by provider 'openai'"
    ) as exc_info:
        normalize_provider_config("openai", {"top_k": 40})
    msg = str(exc_info.value)
    assert "anthropic" in msg
    assert "google" in msg


def test_seed_unsupported_for_anthropic_raises():
    with pytest.raises(
        ValueError, match=r"setting 'seed' is not supported by provider 'anthropic'"
    ) as exc_info:
        normalize_provider_config("anthropic", {"seed": 42})
    msg = str(exc_info.value)
    assert "openai" in msg


def test_metadata_only_anthropic():
    assert normalize_provider_config("anthropic", {"metadata": {"key": "val"}}) == {
        "metadata": {"key": "val"}
    }

    for provider in ("openai", "openai-response", "google"):
        with pytest.raises(
            ValueError, match=rf"setting 'metadata' is not supported by provider '{provider}'"
        ):
            normalize_provider_config(provider, {"metadata": {"key": "val"}})


def test_stop_translates_to_stop_sequences_for_anthropic_and_google():
    assert normalize_provider_config("openai", {"stop": ["end"]}) == {"stop": ["end"]}

    with pytest.raises(
        ValueError, match=r"setting 'stop' is not supported by provider 'openai-response'"
    ):
        normalize_provider_config("openai-response", {"stop": ["end"]})

    result = normalize_provider_config("anthropic", {"stop": ["end"]})
    assert result == {"stop_sequences": ["end"]}

    result = normalize_provider_config("google", {"stop": ["end"]})
    assert result == {"stop_sequences": ["end"]}


def test_reasoning_effort_wraps_for_openai_response():
    result = normalize_provider_config("openai-response", {"reasoning_effort": "medium"})
    assert result == {"reasoning": {"effort": "medium"}}


def test_reasoning_effort_wraps_for_anthropic_adaptive_thinking():
    result = normalize_provider_config("anthropic", {"reasoning_effort": "high"})
    assert result == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }


def test_reasoning_effort_anthropic_accepts_all_effort_levels():
    for level in ("low", "medium", "high", "xhigh", "max"):
        result = normalize_provider_config("anthropic", {"reasoning_effort": level})
        assert result == {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": level},
        }


def test_reasoning_effort_anthropic_none_or_minimal_disables_thinking():
    for off_value in ("none", "minimal"):
        result = normalize_provider_config("anthropic", {"reasoning_effort": off_value})
        assert result == {}


def test_reasoning_effort_anthropic_invalid_value_raises():
    for bad in ("lowish", "default", "all", ""):
        with pytest.raises(
            ValueError, match=r"invalid reasoning_effort '[^']*' for provider 'anthropic'"
        ):
            normalize_provider_config("anthropic", {"reasoning_effort": bad})


def test_reasoning_effort_anthropic_does_not_drop_other_settings():
    result = normalize_provider_config(
        "anthropic",
        {"reasoning_effort": "medium", "temperature": 0.2, "max_tokens": 2048},
    )
    assert result == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "medium"},
        "temperature": 0.2,
        "max_tokens": 2048,
    }


def test_reasoning_effort_unsupported_for_other_providers_raises():
    for provider in ("openai", "google"):
        with pytest.raises(
            ValueError,
            match=rf"setting 'reasoning_effort' is not supported by provider '{provider}'",
        ):
            normalize_provider_config(provider, {"reasoning_effort": "medium"})


def test_unknown_key_raises_with_helpful_message():
    with pytest.raises(
        ValueError, match=r"unknown provider_config key 'response_format'"
    ) as exc_info:
        normalize_provider_config("openai", {"response_format": {"type": "json_object"}})
    msg = str(exc_info.value)
    assert "provider_passthrough" in msg
    assert "temperature" in msg


def test_none_value_dropped():
    result = normalize_provider_config("openai", {"temperature": None})
    assert result == {}


def test_empty_config_returns_empty_dict():
    assert normalize_provider_config("openai", {}) == {}


def test_canonical_settings_constant_frozenset():
    assert (
        frozenset(
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
        == CANONICAL_SETTINGS
    )
