"""Web search tool using Tavily API.

Wraps TavilySearchResults as a standard LangChain tool so nodes can call it
directly or pass it to a tool-calling agent.
"""

from langchain_tavily import TavilySearch

from app.config import settings


def get_search_tool(max_results: int = 5) -> TavilySearch:
    """Return a configured Tavily search tool."""
    return TavilySearch(
        max_results=max_results,
        tavily_api_key=settings.tavily_api_key,
    )
