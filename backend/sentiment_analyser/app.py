import streamlit as st
from orchestrator import run_full_pipeline

st.set_page_config(page_title="Financial News Sentiment", layout="wide")

st.title("Financial Sentiment Analysis Dashboard")

query = st.text_input("Enter topic (e.g., 'Adani IPO', 'RBI interest rates')", value="Zomato IPO")

if st.button("Run Analysis"):
    with st.spinner("Fetching and analyzing..."):
        output = run_full_pipeline(query)

    st.subheader("Overall Summary")
    st.write(f"**Query:** {output['query']}")
    st.write(f"**Majority Sentiment:** {output['majority_sentiment']}")
    st.write(f"**Average Score:** {output['average_score']}")
    st.write(f"**Median Score:** {output['median_score']}")

    st.subheader("Articles")
    for i, article in enumerate(output["articles"], 1):
        st.markdown(f"### {i}. [{article['title']}]({article['url']})")
        st.write(f"**Sentiment:** {article['sentiment']} | **Score:** {article['score']}")
        st.write(article["snippet"])