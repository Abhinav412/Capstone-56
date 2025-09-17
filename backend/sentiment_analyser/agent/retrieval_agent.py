import time
from utils.fetch_news import fetch_news
from utils.extract_article import extract_full_text
from models.finbert import FinBERTSentiment

class NewsRetrievalAgent:
    def __init__(self, query="IPO OR stock market", interval=900, limit=5):
        self.query = query
        self.interval = interval  # in seconds
        self.limit = limit
        self.model = FinBERTSentiment()
        self.seen_urls = set()

    def run_once(self):
        print(f"[INFO] Fetching news for query: {self.query}")
        articles = fetch_news(query=self.query, page_size=self.limit)
        
        for article in articles:
            url = article.get("url")
            if url in self.seen_urls:
                continue  # Avoid duplicates

            content = extract_full_text(url)
            if not content:
                continue

            sentiment = self.model.predict(content)

            result = {
                "title": article.get("title"),
                "url": url,
                "source": article.get("source", {}).get("name"),
                "publishedAt": article.get("publishedAt"),
                "sentiment": sentiment["sentiment"],
                "confidence": sentiment["confidence"]
            }

            print("[Processed Article]")
            print(result)

            self.seen_urls.add(url)

    def run_forever(self):
        while True:
            try:
                self.run_once()
                time.sleep(self.interval)
            except Exception as e:
                print(f"[ERROR] during agent run: {e}")
                time.sleep(60)
