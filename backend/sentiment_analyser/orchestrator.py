from agents.search_agent import run_search_agent
from agents.extract_agent import run_extract_agent
from agents.sentiment_agent import run_sentiment_agent
import statistics
from collections import Counter

def run_full_pipeline(query: str, limit: int = 5):
    articles = run_search_agent(query=query)[:limit]

    results = []
    for article in articles:
        url = article["url"]
        content = run_extract_agent(url)
        sentiment = run_sentiment_agent(content)
        article.update({
            "content": content[:300] + "...",
            "sentiment": sentiment["sentiment"],
            "score": sentiment["confidence"][sentiment["sentiment"]]
        })
        results.append(article)

    labels = [r["sentiment"] for r in results]
    scores = [r["score"] for r in results]

    if not labels:
        majority = "N/A"
        avg_score = 0.0
        median_score = 0.0
    else:
        majority = Counter(labels).most_common(1)[0][0]
        avg_score = round(sum(scores) / len(scores), 4)
        median_score = round(statistics.median(scores), 4)

    return {
        "query": query,
        "majority_sentiment": majority,
        "average_score": avg_score,
        "median_score": median_score,
        "articles": results
    }
