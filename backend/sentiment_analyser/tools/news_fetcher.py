from smolagents.tools import tool
from utils.fetch_news import fetch_news
import requests
import os

@tool
def get_financial_news(query: str, limit: int = 5) -> list:
    """
    Fetch recent financial news headlines using NewsAPI.

    Args:
        query (str): The topic or keyword to search for.
        limit (int): Maximum number of articles to fetch.

    Returns:
        list: A list of dictionaries containing news headlines and URLs.
    """
    api_key = os.getenv("news_api_key")
    if not api_key:
        raise ValueError("news_api_key environment variable not set")
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": limit,
        "apiKey": api_key
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"NewsAPI Error: {response.status_code} - {response.text}")
    
    articles = response.json().get("articles", [])
    
    # Return simplified article data
    return [
        {
            "title": article.get("title"),
            "url": article.get("url"),
            "source": article.get("source", {}).get("name"),
            "publishedAt": article.get("publishedAt"),
            "description": article.get("description")
        }
        for article in articles
    ]