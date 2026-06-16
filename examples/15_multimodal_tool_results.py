"""Example 15: Multimodal tool results.

A tool can return not just a string but also images, documents, video,
or audio. The agent loop forwards these to the LLM so the model can
actually see / read the returned file — not just a stringified link.

Use the `coreouto.Image`, `Document`, `Video`, `Audio`, `Text` content
block builders (or just return raw `coreouto.ImageBlock(...)` etc.) and
return them as a list, or wrap them in a `coreouto.ToolResult`.

Provider support:

  provider           | images | documents | video | audio
  -------------------|--------|-----------|-------|------
  anthropic          |   yes  |    yes    |  yes  |  yes
  openai-response    |   yes  |    yes    |  no   |  no
  google (new SDK)   |   yes  |    yes    |  yes  |  yes
  openai (chat)      |   no   |    no     |  no   |  no   -> ValueError

Run with:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/15_multimodal_tool_results.py
"""

from __future__ import annotations

import asyncio
import os

import coreouto as co
from coreouto._types import AgentConfig


@co.register_tool("screenshot_website")
async def screenshot_website(url: str) -> list[co.ContentBlock]:
    """Capture a screenshot of `url` and return it as an image block.

    The actual capture is mocked here; the point is the return shape.
    """
    fake_png = b"\x89PNG\r\n\x1a\n" + b"faked-screenshot-bytes" * 8
    return [
        co.TextBlock(text=f"Screenshot captured for {url}:"),
        co.ImageBlock(data=fake_png, mime_type="image/png"),
    ]


@co.register_tool("fetch_pdf_report")
async def fetch_pdf_report(ticker: str) -> co.ToolResult:
    """Return a PDF report (mocked) as a document block.

    Note: ToolResult lets you return either a string (`content`) or a
    list of blocks (`blocks`). Use `ToolResult` when you want to set
    `is_error`, or when you want the explicit shape.
    """
    fake_pdf = b"%PDF-1.4\n%fake-pdf-bytes-for-ticker-" + ticker.encode() * 4
    return co.ToolResult(
        tool_call_id="",  # the agent loop fills this in
        blocks=[
            co.TextBlock(text=f"Q4 report for {ticker}:"),
            co.DocumentBlock(data=fake_pdf, mime_type="application/pdf"),
        ],
    )


@co.register_tool("classify_chart")
async def classify_chart(image_url: str) -> co.ToolResult:
    """Return a chart image by URL plus a short caption.

    The agent sees both the text and the image and can answer questions
    about the chart.
    """
    return co.ToolResult(
        tool_call_id="",
        blocks=[
            co.ImageBlock(url=image_url),
            co.TextBlock(text="Caption: daily active users, last 30 days."),
        ],
    )


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for this example")

    co.providers.anthropic.register(api_key=api_key)

    config = AgentConfig(
        name="multimodal-demo",
        model="claude-sonnet-4-6",
        provider="anthropic",
        system_prompt=(
            "You are a research analyst. When tools return images, "
            "DOCUMENTS, or other media, look at them and incorporate the "
            "information into your answer. Call the `finish` tool when you "
            "are done."
        ),
        tools=["screenshot_website", "fetch_pdf_report", "classify_chart"],
    )
    agent = co.Agent(config)

    response = await agent.call(
        "Capture a screenshot of https://example.com and tell me what you see."
    )
    print("Screenshot task:", response.content)
    print("Iterations:", response.iterations)


if __name__ == "__main__":
    asyncio.run(main())
