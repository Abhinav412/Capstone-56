import requests
import os

def fetch_news(query="IPO OR stock", language="en", page_size=5):
    api_key = os.getenv("news_api_key")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": language,
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"NewsAPI Error: {response.status_code} - {response.text}")
    return response.json().get("articles", [])