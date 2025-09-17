from smolagents import CodeAgent
from smolagents import InferenceClientModel
from tools.news_fetcher import get_financial_news
from tools.content_extractor import extract_article_content
from tools.finbert_tool import analyze_sentiment
import config

def run_smart_sentiment_agent(queries: list[str], articles_per_query: int = 5) -> dict:
    results = {}
    
    model = InferenceClientModel()  
    
    agent = CodeAgent(
        tools=[get_financial_news, extract_article_content, analyze_sentiment],
        model=model,
        additional_authorized_imports=["requests", "newspaper", "transformers"]
    )

    for query in queries:
        task = f"""
Search for up to {articles_per_query} recent financial news articles about "{query}".
For each article:
1. Extract the full content.
2. Analyze the sentiment with FinBERT.

Then summarize the overall sentiment (majority label + net score) and provide a list of article titles, sentiment, and URLs.
"""
        try:
            output = agent.run(task)
            results[query] = output
        except Exception as e:
            results[query] = {"error": str(e)}

    return results
