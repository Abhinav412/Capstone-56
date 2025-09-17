from fastapi import FastAPI
from models.finbert import FinBERTSentiment
from utils.fetch_news import fetch_news
from utils.extract_article import extract_full_text
import config

app = FastAPI()
model = FinBERTSentiment()

@app.get("/analyze-latest")
def analyze_latest(q: str = "IPO OR market", limit: int = 5):
    articles = fetch_news(query=q, page_size=limit)
    results = []

    for article in articles:
        url = article.get("url")
        content = extract_full_text(url)
        if not content:
            continue
        sentiment = model.predict(content)

        results.append({
            "title": article.get("title"),
            "url": url,
            "source": article.get("source", {}).get("name"),
            "publishedAt": article.get("publishedAt"),
            "sentiment": sentiment["sentiment"],
            "confidence": sentiment["confidence"]
        })

    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)