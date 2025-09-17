from tavily import TavilyClient
import os

TAVILY_API_KEY = os.getenv("tavily_api_key")
def search_news(query: str, num_results: int = 5):
    client = TavilyClient(api_key=TAVILY_API_KEY)
    results = client.search(query=query, search_depth="advanced", max_results=num_results)
    return [{"title": r["title"], "url": r["url"], "snippet": r["content"]} for r in results.get("results", [])]