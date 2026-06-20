from __future__ import annotations

import pytest

from coreouto._types import AgentConfig
from coreouto.presets import (
    AgentPreset,
    clear_agent_presets,
    get_agent_preset,
    list_agent_presets,
    register_agent_preset,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    clear_agent_presets()
    yield
    clear_agent_presets()


def test_register_agent_preset_creates_preset():
    preset = register_agent_preset("researcher", model="gpt-4", provider="openai")
    assert isinstance(preset, AgentPreset)
    assert preset.name == "researcher"
    assert preset.model == "gpt-4"
    assert preset.provider == "openai"


def test_get_agent_preset_returns_registered_preset():
    registered = register_agent_preset("researcher", model="gpt-4", provider="openai")
    fetched = get_agent_preset("researcher")
    assert fetched is registered


def test_reregister_overwrites_last_write_wins():
    first = register_agent_preset("researcher", model="gpt-4", provider="openai")
    second = register_agent_preset("researcher", model="claude-3", provider="anthropic")
    assert second.model == "claude-3"
    assert second.provider == "anthropic"
    assert get_agent_preset("researcher") is second
    assert first is not second


def test_get_agent_preset_missing_raises_keyerror_with_informative_message():
    register_agent_preset("alpha", model="gpt-4", provider="openai")
    register_agent_preset("beta", model="gpt-4", provider="openai")
    with pytest.raises(KeyError) as exc_info:
        get_agent_preset("missing")
    msg = str(exc_info.value)
    assert "missing" in msg
    assert "alpha" in msg
    assert "beta" in msg


def test_list_agent_presets_returns_sorted_names():
    register_agent_preset("charlie", model="gpt-4", provider="openai")
    register_agent_preset("alpha", model="gpt-4", provider="openai")
    register_agent_preset("bravo", model="gpt-4", provider="openai")
    assert list_agent_presets() == ["alpha", "bravo", "charlie"]


def test_list_agent_presets_empty_when_cleared():
    register_agent_preset("alpha", model="gpt-4", provider="openai")
    clear_agent_presets()
    assert list_agent_presets() == []


def test_clear_agent_presets_empties_registry():
    register_agent_preset("alpha", model="gpt-4", provider="openai")
    register_agent_preset("beta", model="gpt-4", provider="openai")
    assert list_agent_presets() == ["alpha", "beta"]
    clear_agent_presets()
    assert list_agent_presets() == []


def test_to_config_returns_agent_config_with_preset_fields():
    preset = register_agent_preset(
        "researcher",
        model="gpt-4",
        provider="openai",
        system_prompt="You research things.",
        tools=["search", "read"],
        max_iterations=20,
    )
    config = preset.to_config()
    assert isinstance(config, AgentConfig)
    assert config.name == "researcher"
    assert config.model == "gpt-4"
    assert config.provider == "openai"
    assert config.system_prompt == "You research things."
    assert config.tools == ["search", "read"]
    assert config.max_iterations == 20


def test_to_config_drops_description():
    preset = register_agent_preset(
        "researcher",
        model="gpt-4",
        provider="openai",
        description="A research assistant preset.",
    )
    config = preset.to_config()
    assert isinstance(config, AgentConfig)
    assert not hasattr(config, "description") or getattr(config, "description", None) is None
    assert config.name == "researcher"
    assert config.model == "gpt-4"
    assert config.provider == "openai"


def test_optional_description_stored_on_preset():
    preset = register_agent_preset(
        "researcher",
        model="gpt-4",
        provider="openai",
        description="A research assistant.",
    )
    assert preset.description == "A research assistant."


def test_optional_system_prompt_stored_on_preset():
    preset = register_agent_preset(
        "researcher",
        model="gpt-4",
        provider="openai",
        system_prompt="Be terse.",
    )
    assert preset.system_prompt == "Be terse."


def test_optional_tools_stored_on_preset():
    preset = register_agent_preset(
        "researcher",
        model="gpt-4",
        provider="openai",
        tools=["search", "calc"],
    )
    assert preset.tools == ["search", "calc"]


def test_tools_default_is_empty_list():
    preset = register_agent_preset("researcher", model="gpt-4", provider="openai")
    assert preset.tools == []


def test_max_iterations_default_is_unlimited():
    preset = register_agent_preset("researcher", model="gpt-4", provider="openai")
    assert preset.max_iterations is None
    assert preset.to_config().max_iterations is None


def test_max_iterations_explicit_none_is_unlimited():
    preset = register_agent_preset(
        "researcher", model="gpt-4", provider="openai", max_iterations=None
    )
    assert preset.max_iterations is None
    assert preset.to_config().max_iterations is None


def test_max_iterations_explicit_int_is_preserved():
    preset = register_agent_preset(
        "researcher", model="gpt-4", provider="openai", max_iterations=20
    )
    assert preset.max_iterations == 20
    assert preset.to_config().max_iterations == 20


def test_system_prompt_default_is_none():
    preset = register_agent_preset("researcher", model="gpt-4", provider="openai")
    assert preset.system_prompt is None


def test_description_default_is_none():
    preset = register_agent_preset("researcher", model="gpt-4", provider="openai")
    assert preset.description is None


def test_provider_passthrough_stored_and_converted():
    preset = register_agent_preset(
        "researcher",
        model="gpt-4",
        provider="openai",
        provider_passthrough={"response_format": {"type": "json_object"}},
    )
    assert preset.provider_passthrough == {"response_format": {"type": "json_object"}}
    config = preset.to_config()
    assert config.provider_passthrough == {"response_format": {"type": "json_object"}}


def test_provider_passthrough_default_is_empty_dict():
    preset = register_agent_preset("researcher", model="gpt-4", provider="openai")
    assert preset.provider_passthrough == {}
