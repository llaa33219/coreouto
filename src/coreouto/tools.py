from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Callable
from typing import Any, get_args, get_origin

from pydantic import BaseModel, ConfigDict

RESERVED_TOOL_NAMES = frozenset({"finish"})


class Tool(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Any
    parallelizable: bool = True


_TOOL_REGISTRY: dict[str, Tool] = {}


def clear_tools() -> None:
    _TOOL_REGISTRY.clear()


def get_tool(name: str) -> Tool | None:
    return _TOOL_REGISTRY.get(name)


def list_tools() -> list[str]:
    return list(_TOOL_REGISTRY.keys())


def _type_to_schema(tp: Any) -> dict[str, Any]:
    origin = get_origin(tp)
    args = get_args(tp)

    if origin is typing.Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and type(None) in args:
            return _type_to_schema(non_none[0])
        raise NotImplementedError(f"Cannot extract schema for {tp}")

    if tp is list or origin is list:
        if args:
            return {"type": "array", "items": _type_to_schema(args[0])}
        return {"type": "array"}

    if tp is dict or origin is dict:
        return {"type": "object"}

    if tp is str:
        return {"type": "string"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is bool:
        return {"type": "boolean"}

    try:
        if isinstance(tp, type) and issubclass(tp, BaseModel):
            return tp.model_json_schema()
    except TypeError:
        pass

    if origin is typing.Literal:
        return {"enum": list(args)}

    raise NotImplementedError(f"Cannot extract schema for {tp}")


def extract_schema(func: Callable[..., Any]) -> dict[str, Any]:
    underlying = getattr(func, "__func__", func)
    sig = inspect.signature(underlying)
    try:
        hints = typing.get_type_hints(underlying)
    except NameError as exc:
        raise NotImplementedError(f"Cannot extract schema for unresolved type: {exc}") from exc

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        tp = hints.get(name, Any)
        properties[name] = _type_to_schema(tp)

        if param.default is inspect.Parameter.empty:
            origin = get_origin(tp)
            args = get_args(tp)
            is_optional = (origin is typing.Union or origin is types.UnionType) and type(
                None
            ) in args
            if not is_optional:
                required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _assert_not_reserved(name: str) -> None:
    if name in RESERVED_TOOL_NAMES:
        raise ValueError(
            f"{name!r} is a reserved tool name injected automatically by the agent loop. "
            f"Use the built-in to terminate the run; do not register a user tool with this name."
        )


def register_tool(
    name: str | None = None,
    description: str | None = None,
    parallelizable: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name if name is not None else func.__name__
        tool_description = description if description is not None else (func.__doc__ or "").strip()

        _assert_not_reserved(tool_name)
        schema = extract_schema(func)

        tool = Tool(
            name=tool_name,
            description=tool_description,
            parameters=schema,
            handler=func,
            parallelizable=parallelizable,
        )
        _TOOL_REGISTRY[tool_name] = tool

        return func

    if callable(name):
        func = name
        name = None
        return decorator(func)

    return decorator


def register_tool_class(
    name: str,
    cls: type[Any],
    *,
    handler: Callable[..., Any] | None = None,
    description: str | None = None,
    parallelizable: bool = True,
) -> None:
    if handler is None:
        raise ValueError("class-based tools require explicit handler")

    _assert_not_reserved(name)

    tool_description = description if description is not None else (cls.__doc__ or "").strip()

    schema = extract_schema(handler)

    tool = Tool(
        name=name,
        description=tool_description,
        parameters=schema,
        handler=handler,
        parallelizable=parallelizable,
    )
    _TOOL_REGISTRY[name] = tool
