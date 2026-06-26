from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from coreouto.tools import (
    RESERVED_TOOL_NAMES,
    Tool,
    clear_tools,
    extract_schema,
    get_tool,
    list_tools,
    register_tool,
    register_tool_class,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    clear_tools()
    yield
    clear_tools()


class _Point(BaseModel):
    x: float
    y: float


class TestToolModel:
    def test_tool_fields(self) -> None:
        def handler(x: int) -> int:
            return x

        t = Tool(
            name="add",
            description="adds",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=handler,
        )
        assert t.name == "add"
        assert t.description == "adds"
        assert t.parameters == {"type": "object", "properties": {}, "required": []}
        assert t.handler is handler


class TestRegisterTool:
    def test_sync_handler(self) -> None:
        @register_tool("greet")
        def greet(name: str) -> str:
            return f"Hello {name}"

        tool = get_tool("greet")
        assert tool is not None
        assert tool.name == "greet"
        assert tool.description == ""
        assert tool.handler is greet
        assert tool.parameters["properties"]["name"] == {"type": "string"}
        assert tool.parameters["required"] == ["name"]
        assert greet("world") == "Hello world"

    def test_async_handler(self) -> None:
        @register_tool("async_greet")
        async def async_greet(name: str) -> str:
            return f"Hello {name}"

        tool = get_tool("async_greet")
        assert tool is not None
        assert tool.name == "async_greet"
        assert tool.handler is async_greet

    def test_uses_func_name_when_no_name(self) -> None:
        @register_tool()
        def my_tool(x: int) -> int:
            return x

        assert get_tool("my_tool") is not None

    def test_uses_docstring_when_no_description(self) -> None:
        @register_tool("doc_tool")
        def doc_tool(x: int) -> int:
            """Does a thing."""
            return x

        tool = get_tool("doc_tool")
        assert tool is not None
        assert tool.description == "Does a thing."

    def test_uses_empty_string_when_no_docstring(self) -> None:
        @register_tool("no_doc")
        def no_doc(x: int) -> int:
            return x

        tool = get_tool("no_doc")
        assert tool is not None
        assert tool.description == ""

    def test_explicit_description_overrides_docstring(self) -> None:
        @register_tool("over", description="override")
        def over(x: int) -> int:
            """Original."""
            return x

        tool = get_tool("over")
        assert tool is not None
        assert tool.description == "override"

    def test_decorator_returns_original_function(self) -> None:
        def orig(x: int) -> int:
            return x

        decorated = register_tool("orig")(orig)
        assert decorated is orig

    def test_reserved_names_empty_allows_any_name(self) -> None:
        @register_tool("finish")
        def finish_tool(x: int) -> int:
            return x

        @register_tool("continue_loop")
        def continue_loop_tool(x: int) -> int:
            return x

        assert get_tool("finish") is not None
        assert get_tool("finish").handler is finish_tool
        assert get_tool("continue_loop") is not None
        assert get_tool("continue_loop").handler is continue_loop_tool


class TestRegisterToolClass:
    def test_with_explicit_handler(self) -> None:
        class MyClass:
            def method(self, value: int) -> int:
                return value * 2

        instance = MyClass()
        handler = instance.method
        register_tool_class("my_class_tool", MyClass, handler=handler)
        tool = get_tool("my_class_tool")
        assert tool is not None
        assert tool.handler is handler
        assert tool.name == "my_class_tool"

    def test_without_handler_raises(self) -> None:
        class MyClass:
            pass

        with pytest.raises(ValueError, match="class-based tools require explicit handler"):
            register_tool_class("bad", MyClass)

    def test_register_tool_class_continue_loop_name_succeeds(self) -> None:
        class MyClass:
            def method(self, value: int) -> int:
                return value * 2

        instance = MyClass()
        register_tool_class("continue_loop", MyClass, handler=instance.method)
        tool = get_tool("continue_loop")
        assert tool is not None
        assert tool.name == "continue_loop"
        assert tool.handler.__func__ is instance.method.__func__


class TestRegistry:
    def test_reserved_tool_names_is_empty(self) -> None:
        assert frozenset() == RESERVED_TOOL_NAMES

    def test_get_tool_missing_returns_none(self) -> None:
        assert get_tool("missing") is None

    def test_list_tools(self) -> None:
        @register_tool("a")
        def a(x: int) -> int:
            return x

        @register_tool("b")
        def b(x: int) -> int:
            return x

        names = list_tools()
        assert sorted(names) == ["a", "b"]

    def test_clear_tools(self) -> None:
        @register_tool("tmp")
        def tmp(x: int) -> int:
            return x

        assert get_tool("tmp") is not None
        clear_tools()
        assert get_tool("tmp") is None
        assert list_tools() == []


class TestExtractSchema:
    def test_primitive_types(self) -> None:
        def fn(a: str, b: int, c: float, d: bool) -> None:
            pass

        schema = extract_schema(fn)
        assert schema["type"] == "object"
        assert schema["properties"]["a"] == {"type": "string"}
        assert schema["properties"]["b"] == {"type": "integer"}
        assert schema["properties"]["c"] == {"type": "number"}
        assert schema["properties"]["d"] == {"type": "boolean"}
        assert sorted(schema["required"]) == ["a", "b", "c", "d"]

    def test_list_types(self) -> None:
        def fn(a: list[str], b: list[int]) -> None:
            pass

        schema = extract_schema(fn)
        assert schema["properties"]["a"] == {"type": "array", "items": {"type": "string"}}
        assert schema["properties"]["b"] == {"type": "array", "items": {"type": "integer"}}

    def test_dict_type(self) -> None:
        def fn(a: dict[str, int]) -> None:
            pass

        schema = extract_schema(fn)
        assert schema["properties"]["a"] == {"type": "object"}

    def test_optional_not_required(self) -> None:
        def fn(a: int, b: int | None) -> None:
            pass

        schema = extract_schema(fn)
        assert schema["properties"]["a"] == {"type": "integer"}
        assert schema["properties"]["b"] == {"type": "integer"}
        assert schema["required"] == ["a"]

    def test_union_none_not_required(self) -> None:
        def fn(a: int, b: int | None) -> None:
            pass

        schema = extract_schema(fn)
        assert schema["properties"]["a"] == {"type": "integer"}
        assert schema["properties"]["b"] == {"type": "integer"}
        assert schema["required"] == ["a"]

    def test_base_model(self) -> None:
        def fn(p: _Point) -> None:
            pass

        schema = extract_schema(fn)
        assert schema["properties"]["p"]["type"] == "object"
        assert "x" in schema["properties"]["p"]["properties"]
        assert "y" in schema["properties"]["p"]["properties"]

    def test_literal(self) -> None:
        def fn(mode: Literal["a", "b"]) -> None:
            pass

        schema = extract_schema(fn)
        assert schema["properties"]["mode"] == {"enum": ["a", "b"]}

    def test_unsupported_type_raises(self) -> None:
        class Custom:
            pass

        def fn(x: Custom) -> None:
            pass

        with pytest.raises(NotImplementedError, match="Cannot extract schema for"):
            extract_schema(fn)

    def test_skips_self(self) -> None:
        class MyClass:
            def method(self, value: int) -> int:
                return value

        schema = extract_schema(MyClass.method)
        assert "self" not in schema["properties"]
        assert schema["properties"]["value"] == {"type": "integer"}

    def test_default_makes_not_required(self) -> None:
        def fn(a: int, b: str = "default") -> None:
            pass

        schema = extract_schema(fn)
        assert schema["required"] == ["a"]
