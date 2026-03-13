"""
mcp_server/server.py
MCP (Model Context Protocol) server exposing seller knowledge base as a tool.
Agents call this tool to retrieve relevant seller FAQ content.

Run: python mcp_server/server.py
"""

import sys
import asyncio
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from rag.retriever import retrieve_with_scores

app = Server("seller-knowledge-base")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_seller_knowledge",
            description=(
                "Search the Amazon Seller Central knowledge base for information "
                "about listings, fees, FBA, account health, returns, shipping, "
                "brand registry, inventory management, and policies."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The seller question or topic to search for"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 4)",
                        "default": 4
                    }
                },
                "required": ["query"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name != "search_seller_knowledge":
        raise ValueError(f"Unknown tool: {name}")

    query = arguments["query"]
    k = arguments.get("k", 4)

    results = retrieve_with_scores(query, k=k)

    output = []
    for doc, score in results:
        output.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "relevance_score": round(float(score), 4)
        })

    return [types.TextContent(
        type="text",
        text=json.dumps(output, indent=2)
    )]


async def main():
    print("Starting MCP server: seller-knowledge-base")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())