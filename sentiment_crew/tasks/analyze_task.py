from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from crewai import Task

from ..agents.sentiment_agent import sentiment_analyser
from ..tools.fetch_indices import fetch_indices_tool
from ..tools.fetch_news import fetch_news_tool
from ..tools.finbert_tool import finbert_sentiment_tool
from ..tools.sentiment_summary_tool import summarize_sentiment_results


class ProbabilityTriple(BaseModel):
    negative: float = Field(default=0.0, ge=0.0, le=1.0)
    neutral: float = Field(default=0.0, ge=0.0, le=1.0)
    positive: float = Field(default=0.0, ge=0.0, le=1.0)


class SentimentDistribution(BaseModel):
    negative: int = Field(default=0, ge=0)
    neutral: int = Field(default=0, ge=0)
    positive: int = Field(default=0, ge=0)


class ArticleScore(BaseModel):
    text: str = ""
    negative: float = Field(default=0.0, ge=0.0, le=1.0)
    neutral: float = Field(default=0.0, ge=0.0, le=1.0)
    positive: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None


class SentimentSection(BaseModel):
    sentiment_label: str = Field(default="neutral")
    summary: str = Field(default="")
    average_probabilities: ProbabilityTriple = Field(default_factory=ProbabilityTriple)
    sentiment_distribution: SentimentDistribution = Field(default_factory=SentimentDistribution)
    article_scores: List[ArticleScore] = Field(default_factory=list)


class IndexSnapshot(BaseModel):
    last_timestamp: str = ""
    last_close: float = 0.0
    change_pct: float = 0.0
    sentiment: str = "neutral"
    fetched_at: str = ""


class SentimentPayload(BaseModel):
    company_sentiment: SentimentSection = Field(default_factory=SentimentSection)
    market_sentiment: SentimentSection = Field(default_factory=SentimentSection)
    market_indices: Dict[str, IndexSnapshot] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


analyze_task = Task(
    description=(
        """Execute the following steps in order for {company_name}:

STEP 1: Fetch News
- Call News fetcher with company_name='{company_name}', days_back=5
- Store the returned company_news and market_news arrays

STEP 2: Score Company Articles
- Extract all 'summary' fields from company_news
- Call finbert_sentiment_tool with the list of summaries
- This returns a list of dicts with {text, negative, neutral, positive}

STEP 3: Summarize Company Sentiment
- Call summarize_sentiment_results with the FinBERT results from step 2
- Store this as company_sentiment

STEP 4: Score Market Articles  
- Extract all 'summary' fields from market_news
- Call finbert_sentiment_tool with the list of summaries

STEP 5: Summarize Market Sentiment
- Call summarize_sentiment_results with the FinBERT results from step 4
- Store this as market_sentiment

STEP 6: Fetch Market Indices
- Call Indices fetcher (no arguments needed)
- Store this as market_indices

STEP 7: Build Final JSON
- Combine company_sentiment, market_sentiment, market_indices into the output JSON structure
- Add any warnings/errors encountered during processing"""
    ),
    tools=[fetch_news_tool, fetch_indices_tool, finbert_sentiment_tool, summarize_sentiment_results],
    expected_output=(
        "Strict JSON matching SentimentPayload schema with:\n"
        "- company_sentiment: {sentiment_label, summary, average_probabilities, sentiment_distribution, article_scores}\n"
        "- market_sentiment: {sentiment_label, summary, average_probabilities, sentiment_distribution, article_scores}\n"
        "- market_indices: {index_name: {last_timestamp, last_close, change_pct, sentiment, fetched_at}}\n"
        "- warnings: [list of non-fatal issues]\n"
        "- errors: [list of fatal issues]"
    ),
    output_json=SentimentPayload,
    agent=sentiment_analyser,
    inputs=["company_name"],
)
