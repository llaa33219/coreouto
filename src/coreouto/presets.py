"""Agent preset registry.

Presets are named bundles of agent configuration that can be converted into
an ``AgentConfig`` and passed to ``Agent(...)``. The registry is a plain
module-level dict; callers opt in to lifecycle management via the
``register_agent_preset`` / ``get_agent_preset`` / ``list_agent_presets`` /
``clear_agent_presets`` functions.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from coreouto._types import AgentConfig

_PRESETS: dict[str, AgentPreset] = {}


class AgentPreset(BaseModel):
    name: str
    model: str
    provider: str
    system_prompt: str | None = None
    tools: list[str] = Field(default_factory=list)
    max_iterations: int | None = None
    description: str | None = None
    provider_passthrough: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _name_not_empty(self) -> AgentPreset:
        if not self.name:
            raise ValueError("preset name must be a non-empty string")
        return self

    def to_config(self) -> AgentConfig:
        return AgentConfig(
            name=self.name,
            model=self.model,
            provider=self.provider,
            system_prompt=self.system_prompt,
            tools=list(self.tools),
            max_iterations=self.max_iterations,
            provider_passthrough=dict(self.provider_passthrough),
        )


def register_agent_preset(
    name: str,
    *,
    model: str,
    provider: str,
    system_prompt: str | None = None,
    tools: list[str] | tuple[str, ...] = (),
    description: str | None = None,
    max_iterations: int | None = None,
    provider_passthrough: dict[str, Any] | None = None,
) -> AgentPreset:
    preset = AgentPreset(
        name=name,
        model=model,
        provider=provider,
        system_prompt=system_prompt,
        tools=list(tools),
        max_iterations=max_iterations,
        description=description,
        provider_passthrough=provider_passthrough or {},
    )
    _PRESETS[name] = preset
    return preset


def get_agent_preset(name: str) -> AgentPreset:
    if name not in _PRESETS:
        raise KeyError(f"agent preset not registered: {name!r}. available: {sorted(_PRESETS)}")
    return _PRESETS[name]


def list_agent_presets() -> list[str]:
    return sorted(_PRESETS)


def clear_agent_presets() -> None:
    _PRESETS.clear()
