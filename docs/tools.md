# Tools

Tools let the agent call functions during its loop. Register a tool with `@register_tool` and coreouto extracts the type hints into JSON Schema so the LLM knows what arguments to pass.

## The `@register_tool` decorator

```python
import coreouto as co

@co.register_tool("search")
def search(query: str) -> str:
    """Search the web for `query`."""
    return f"Results for: {query}"
```

The tool name defaults to the function name if you omit it:

```python
@co.register_tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"
# registered as "search"
```

You can override the description:

```python
@co.register_tool("search", description="Look up information on the web.")
def search(query: str) -> str:
    return f"Results for: {query}"
```

Without an explicit description, the function's docstring is used.

## Type hints to JSON Schema

coreouto inspects your function's type hints and builds a JSON Schema from them. Supported types:

| Type hint                  | JSON Schema                          |
|----------------------------|--------------------------------------|
| `str`                      | `{"type": "string"}`                 |
| `int`                      | `{"type": "integer"}`                |
| `float`                    | `{"type": "number"}`                 |
| `bool`                     | `{"type": "boolean"}`                |
| `list[str]`                | `{"type": "array", "items": {"type": "string"}}` |
| `list` (bare)              | `{"type": "array"}`                  |
| `dict`                     | `{"type": "object"}`                 |
| `Optional[str]`            | `{"type": "string"}` (not required)  |
| `Literal["a", "b"]`        | `{"enum": ["a", "b"]}`              |
| `BaseModel` subclass       | The model's JSON Schema              |

Parameters without a default value are marked `required`. Parameters with a default value or `Optional` type are not.

### Example with mixed types

```python
from typing import Optional, Literal
from pydantic import BaseModel

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str

@co.register_tool("advanced_search")
def advanced_search(
    query: str,
    max_results: int = 10,
    source: Literal["web", "news", "images"] = "web",
    language: Optional[str] = None,
) -> str:
    """Search with filters."""
    return f"Found results for {query}"
```

This produces a schema where `query` is required, and `max_results`, `source`, and `language` are optional.

### Unsupported types

Union types other than `Optional` (e.g., `str | int`) raise `NotImplementedError` at registration time. If you need complex input, use a `BaseModel`:

```python
from pydantic import BaseModel

class SearchParams(BaseModel):
    query: str
    filters: dict[str, str]

@co.register_tool("filtered_search")
def filtered_search(params: SearchParams) -> str:
    """Search with structured filters."""
    return f"Searching: {params.query}"
```

## Async tool handlers

Tools can be `async`. The agent loop awaits them automatically:

```python
@co.register_tool("fetch_page")
async def fetch_page(url: str) -> str:
    """Fetch a web page."""
    # async HTTP call here
    return "<html>...</html>"
```

## Returning errors

If a tool raises an exception, the agent loop catches it and sends the error back to the LLM as a tool result. The agent can then decide what to do:

```python
@co.register_tool("divide")
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    return a / b
# If b=0, the LLM receives: "ZeroDivisionError: division by zero"
```

You can also return error strings explicitly:

```python
@co.register_tool("validate")
def validate(input: str) -> str:
    """Validate user input."""
    if not input.strip():
        return "Error: input cannot be empty"
    return "Valid"
```

## Class-based tools

For tools that need state or come from a class, use `register_tool_class`:

```python
class DatabaseTool:
    """Query a database."""

    def __init__(self, connection_string: str):
        self.conn = connection_string

    async def query(self, sql: str) -> str:
        """Execute a SQL query."""
        return f"Results for: {sql}"


db = DatabaseTool("postgresql://localhost/mydb")
co.register_tool_class(
    "db_query",
    DatabaseTool,
    handler=db.query,
)
```

The `handler` argument is required. It's the callable that gets invoked when the agent calls the tool.

## Listing and inspecting tools

```python
co.list_tools()       # => ["search", "advanced_search", "fetch_page"]
tool = co.get_tool("search")  # => Tool(name="search", description=..., parameters=..., handler=...)
```

Clear all tools (useful in tests):

```python
co.clear_tools()
```

## Using tools with agents

List tool names in the preset or config:

```python
from coreouto.providers.openai import OpenAIProvider

co.register_provider("minimax", OpenAIProvider(
    api_key="...",
    base_url="https://api.minimax.io/v1",
))

preset = co.register_agent_preset(
    "researcher",
    model="MiniMax-M3",
    provider="minimax",
    tools=["search", "fetch_page"],
)
```

## Multimodal tool results

A tool can return images, documents, video, or audio — not just text. The agent loop forwards these to the LLM so the model can actually see or read the returned file. Return either a list of `ContentBlock` objects or a `ToolResult` with `blocks=...`:

```python
import coreouto as co

@co.register_tool("screenshot")
async def screenshot(url: str) -> list[co.ContentBlock]:
    png_bytes = ...  # capture the screenshot
    return [
        co.TextBlock(text=f"Screenshot of {url}:"),
        co.ImageBlock(data=png_bytes, mime_type="image/png"),
    ]


@co.register_tool("fetch_pdf")
async def fetch_pdf(ticker: str) -> co.ToolResult:
    pdf_bytes = ...
    return co.ToolResult(
        tool_call_id="",  # the agent loop fills this in
        blocks=[
            co.DocumentBlock(data=pdf_bytes, mime_type="application/pdf"),
        ],
    )


@co.register_tool("classify_chart")
async def classify_chart(image_url: str) -> co.ToolResult:
    return co.ToolResult(
        tool_call_id="",
        blocks=[
            co.ImageBlock(url=image_url),
            co.TextBlock(text="Caption: daily active users, last 30 days."),
        ],
    )
```

You can also return a plain string (legacy shape) and coreouto will wrap it in a `ToolResult` for you.

### Content block types

| Block | Fields | Notes |
|---|---|---|
| `TextBlock` | `text: str` | Plain text. |
| `ImageBlock` | `data: bytes` or `url: str`; `mime_type` required when `data` is set | PNG, JPEG, GIF, WebP. |
| `DocumentBlock` | `data: bytes` or `url: str`; `mime_type` required when `data` is set | PDF, text, etc. |
| `VideoBlock` | `data: bytes` or `url: str`; `mime_type` required when `data` is set | MP4, MOV, WebM. |
| `AudioBlock` | `data: bytes` or `url: str`; `mime_type` required when `data` is set | WAV, MP3. |

### Provider support

| Provider | `image` | `document` | `video` | `audio` |
|---|---|---|---|---|
| Anthropic | yes | yes | yes | yes |
| Google (new SDK) | yes | yes | yes | yes |
| OpenAI Responses API | yes | yes | no — `ValueError` | no — `ValueError` |
| OpenAI Chat Completions | no — `ValueError` | no — `ValueError` | no — `ValueError` | no — `ValueError` |

If you need multimodal tool results, register the multimodal-capable provider. For example, switch the preset's `provider="openai"` to `provider="openai-response"` to enable image and document results on OpenAI.

For full per-provider wire-format details, see [Providers — Multimodal support](providers.md#multimodal-support).


