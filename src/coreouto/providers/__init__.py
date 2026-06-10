"""String-keyed provider registry.

Maps provider names to Provider instances (duck-typed, no isinstance check).
"""

from __future__ import annotations

from coreouto.providers.base import Provider

_PROVIDERS: dict[str, Provider] = {}


def register_provider(name: str, provider: Provider) -> None:
    _PROVIDERS[name] = provider


def get_provider(name: str) -> Provider:
    try:
        return _PROVIDERS[name]
    except KeyError:
        raise KeyError(
            f"provider not registered: {name!r}. available: {list(_PROVIDERS)}"
        ) from None


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


def clear_providers() -> None:
    _PROVIDERS.clear()


def __getattr__(name: str):
    if name in ("openai", "anthropic", "google", "openai_response"):
        import importlib

        try:
            return importlib.import_module(f"coreouto.providers.{name}")
        except Exception as exc:
            raise AttributeError(f"module 'coreouto.providers' has no attribute '{name}'") from exc
    raise AttributeError(f"module 'coreouto.providers' has no attribute '{name}'")
