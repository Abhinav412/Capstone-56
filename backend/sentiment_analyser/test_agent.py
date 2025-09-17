import streamlit as st
import pandas as pd
import time
import config
import os
from collections import Counter, defaultdict

from utils.fetch_news import fetch_news
from utils.extract_article import extract_full_text
from models.finbert import FinBERTSentiment

model = FinBERTSentiment()

st.set_page_config(page_title="News Sentiment Dashboard", layout="wide")
st.title("Real-Time Financial News Sentiment Explorer")

query = st.text_input("Enter search term (e.g., IPO, Zomato, stock market)", value="IPO")
limit = st.slider("Number of articles", 1, 20, 5)

if os.getenv("NEWS_API_KEY") is None:
    st.error("NEWS_API_KEY not loaded. Check your .env or config setup.")

if st.button("Fetch and Analyze"):
    with st.spinner("Fetching and analyzing news..."):
        articles = fetch_news(query=query, page_size=limit)
        time.sleep(1)

    if not articles:
        st.warning("No articles found for your query.")
    else:
        results = []

        for article in articles:
            title = article.get("title")
            url = article.get("url")
            source = article.get("source", {}).get("name")
            published_at = article.get("publishedAt")

            try:
                content = extract_full_text(url)
            except Exception as e:
                st.warning(f"Error extracting article: {url}")
                content = ""

            if not content:
                sentiment = {"sentiment": "unavailable", "confidence": {}}
            else:
                sentiment = model.predict(content)

            results.append({
                "Title": title,
                "Source": source,
                "Published": published_at,
                "Sentiment": sentiment["sentiment"],
                "Confidence": sentiment["confidence"].get(sentiment["sentiment"], 0),
                "All_Confidences": sentiment["confidence"],
                "URL": url,
                "Content": content
            })

        df = pd.DataFrame(results)

        sentiment_labels = df["Sentiment"].tolist()
        label_counts = Counter(sentiment_labels)

        majority_sentiment = label_counts.most_common(1)[0][0]

        confidence_totals = defaultdict(float)
        confidence_counts = defaultdict(int)

        for conf_dict in df["All_Confidences"]:
            for label, score in conf_dict.items():
                confidence_totals[label] += score
                confidence_counts[label] += 1

        mean_confidences = {
            label: round(confidence_totals[label] / confidence_counts[label], 4)
            for label in confidence_totals
        }

        label_weights = {"negative": -1, "neutral": 0, "positive": 1}
        weighted_sum = 0
        total_weight = 0

        for label, score in mean_confidences.items():
            weighted_sum += label_weights[label] * score
            total_weight += score

        net_sentiment_score = round(weighted_sum, 3)

        st.subheader("Overall Sentiment Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Majority Sentiment", majority_sentiment.capitalize())
        col2.metric("Articles Analyzed", len(df))
        col3.metric("Net Sentiment Score", net_sentiment_score)

        st.markdown("**Sentiment Label Distribution**")
        st.bar_chart(pd.Series(label_counts))

        st.markdown("**Mean Confidence per Label**")
        st.dataframe(pd.DataFrame([mean_confidences]))

        st.subheader("Article Analysis")
        for _, row in df.iterrows():
            with st.expander(f"{row['Title']} — [{row['Sentiment'].upper()}]"):
                st.markdown(f"**Source:** {row['Source']}")
                st.markdown(f"**Published:** {row['Published']}")
                st.markdown(f"[Read Original]({row['URL']})")
                st.markdown("**Extracted Content:**")
                st.write(row["Content"][:1200] + ("..." if len(row["Content"]) > 1200 else ""))
                st.markdown(f"**Confidence:** {round(row['Confidence'] * 100, 2)}%")
