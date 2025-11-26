import os
from crewai import Agent, LLM

from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model="ollama/qwen3:1.7b",
    base_url="http://localhost:11434",
    #api_key=os.getenv("OLLAMA_API_KEY"),
    temperature=0.1,
    #max_tokens=4000,
)

sentiment_analyser = Agent(
    role="Financial Sentiment Analyser",
    goal=(
        "Fetch company-specific and market-wide news, analyze sentiment using FinBERT, "
        "compute aggregated probabilities, and provide structured JSON output with "
        "company_sentiment, market_sentiment, and market_indices."
    ),
    backstory=(
        "You are an expert financial analyst specializing in sentiment analysis for Indian markets. "
        "You MUST follow this exact workflow:\n"
        "1. Call News fetcher with {company_name} and days_back=5 for both company and market news\n"
        "2. Extract all article summaries and call finbert_sentiment_tool to score them\n"
        "3. Call summarize_sentiment_results separately for company articles and market articles\n"
        "4. Call Indices fetcher to get current market index data\n"
        "5. Return structured JSON matching the output_json schema with all four sections populated"
    ),
    llm=llm,
    verbose=True,
    max_iter=20,
    max_retry_limit=3,
)