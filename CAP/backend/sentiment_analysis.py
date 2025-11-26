"""
Sentiment Analysis Module
Direct tool-based sentiment analysis without LLM orchestration.
Uses NewsAPI + FinBERT for zero-cost, fast sentiment scoring.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Callable
import importlib.util

import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Configure logging
LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Path Setup
# --------------------------------------------------------------------------- #
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
CREW_ROOT = PROJECT_ROOT / 'sentiment_crew'
if not CREW_ROOT.exists():
    CREW_ROOT = PROJECT_ROOT.parent / 'sentiment_crew'

# Add paths for imports
if CREW_ROOT.exists():
    crew_parent = str(CREW_ROOT.parent)
    if crew_parent not in sys.path:
        sys.path.insert(0, crew_parent)
    LOGGER.debug("Added to sys.path: %s", crew_parent)


# --------------------------------------------------------------------------- #
# Dynamic Tool Loader
# --------------------------------------------------------------------------- #
def _load_tool_function(module_path: Path, function_name: str) -> Callable | None:
    """
    Dynamically load a function from a Python file.
    Handles both direct functions and CrewAI Tool wrappers.
    """
    if not module_path.exists():
        LOGGER.debug("Module not found: %s", module_path)
        return None
    
    try:
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if not spec or not spec.loader:
            return None
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"_dynamic_{module_path.stem}"] = module
        spec.loader.exec_module(module)
        
        # Try direct function
        if hasattr(module, function_name):
            func = getattr(module, function_name)
            if callable(func):
                LOGGER.debug("✅ Found callable: %s", function_name)
                return func
        
        # Try Tool wrapper
        tool_name = f"{function_name}_tool"
        if hasattr(module, tool_name):
            tool_obj = getattr(module, tool_name)
            LOGGER.debug("Found tool: %s (type: %s)", tool_name, type(tool_obj).__name__)
            
            # Extract callable
            for attr in ['func', '_run', 'run', '_func']:
                if hasattr(tool_obj, attr):
                    func = getattr(tool_obj, attr)
                    if callable(func):
                        LOGGER.debug("✅ Extracted from tool.%s", attr)
                        return func
            
            if hasattr(tool_obj, '__call__'):
                LOGGER.debug("✅ Tool is callable")
                return lambda *args, **kwargs: tool_obj(*args, **kwargs)
        
        LOGGER.warning("No callable found in %s", module_path)
        return None
        
    except Exception as exc:
        LOGGER.debug("Failed to load %s: %s", module_path, exc)
        return None


# --------------------------------------------------------------------------- #
# Load Tools
# --------------------------------------------------------------------------- #
TOOLS_AVAILABLE = False
fetch_news: Callable | None = None
fetch_indices: Callable | None = None
analyze_sentiment: Callable | None = None

# Try package import
try:
    from sentiment_crew.tools.fetch_news import fetch_news
    from sentiment_crew.tools.fetch_indices import fetch_indices
    from sentiment_crew.tools.finbert_tool import analyze_sentiment
    TOOLS_AVAILABLE = True
    LOGGER.info("✅ All tools loaded via package import")
except ImportError as exc:
    LOGGER.debug("Package import failed: %s", exc)
    
    # Try dynamic loading
    tools_dir = CREW_ROOT / 'tools'
    if tools_dir.exists():
        LOGGER.info("Attempting dynamic tool loading from: %s", tools_dir)
        
        fetch_news = _load_tool_function(tools_dir / 'fetch_news.py', 'fetch_news')
        fetch_indices = _load_tool_function(tools_dir / 'fetch_indices.py', 'fetch_indices')
        analyze_sentiment = _load_tool_function(tools_dir / 'finbert_tool.py', 'analyze_sentiment')
        
        if fetch_news and fetch_indices and analyze_sentiment:
            TOOLS_AVAILABLE = True
            LOGGER.info("✅ All tools loaded via dynamic import")
        else:
            missing = []
            if not fetch_news:
                missing.append("fetch_news")
            if not fetch_indices:
                missing.append("fetch_indices")
            if not analyze_sentiment:
                missing.append("analyze_sentiment")
            LOGGER.error("❌ Missing tools: %s", ", ".join(missing))
    else:
        LOGGER.error("❌ Tools directory not found: %s", tools_dir)

# Final status
if TOOLS_AVAILABLE:
    LOGGER.info("✅ Sentiment analysis ready (NewsAPI + FinBERT + Index data)")
else:
    LOGGER.warning("⚠️ Tools unavailable - CSV fallback only")


# --------------------------------------------------------------------------- #
# Core Sentiment Analysis
# --------------------------------------------------------------------------- #
def _analyze_with_tools(company_name: str) -> Dict[str, Any]:
    """
    Direct sentiment analysis using tools only.
    No LLM orchestration - zero cost, fast execution.
    Includes company news, market news, and index sentiment (NIFTY 50, SENSEX).
    """
    LOGGER.info("Starting tool-based sentiment analysis for: %s", company_name)
    
    if not TOOLS_AVAILABLE or fetch_news is None or analyze_sentiment is None:
        tools_status = f"fetch_news={'✓' if fetch_news else '✗'}, fetch_indices={'✓' if fetch_indices else '✗'}, analyze_sentiment={'✓' if analyze_sentiment else '✗'}"
        error_msg = f"Tools not available ({tools_status}). Check sentiment_crew/tools/ directory."
        LOGGER.error("%s", error_msg)
        
        tools_dir = CREW_ROOT / 'tools'
        if tools_dir.exists():
            tool_files = list(tools_dir.glob('*.py'))
            LOGGER.error("   Available files: %s", [f.name for f in tool_files])
        else:
            LOGGER.error("   Tools directory missing: %s", tools_dir)
        
        return {
            "company_name": company_name,
            "prediction": None,
            "error": error_msg,
        }
    
    try:
        # Fetch company and market news
        LOGGER.info("Fetching company and market news...")
        news_data = fetch_news(company_name, days_back=5)
        
        company_articles = news_data.get("company_news", [])
        market_articles = news_data.get("market_news", [])
        
        # Fetch index news (NIFTY 50, SENSEX)
        index_articles = []
        if fetch_indices:
            LOGGER.info("Fetching index news (NIFTY 50, SENSEX)...")
            try:
                indices_data = fetch_indices(days_back=5)
                index_articles = indices_data.get("indices_news", [])
            except Exception as exc:
                LOGGER.warning("Failed to fetch index news: %s", exc)
        
        LOGGER.info(
            "Fetched %d company + %d market + %d index articles",
            len(company_articles),
            len(market_articles),
            len(index_articles)
        )
        
        # Extract text for analysis
        company_texts = [
            article.get("summary", "") or article.get("title", "")
            for article in company_articles
            if article.get("summary") or article.get("title")
        ]
        
        market_texts = [
            article.get("summary", "") or article.get("title", "")
            for article in market_articles
            if article.get("summary") or article.get("title")
        ]
        
        index_texts = [
            article.get("summary", "") or article.get("title", "")
            for article in index_articles
            if article.get("summary") or article.get("title")
        ]
        
        LOGGER.debug(
            "Extracted %d company + %d market + %d index texts",
            len(company_texts),
            len(market_texts),
            len(index_texts)
        )
        
        # Analyze with FinBERT
        company_scores = []
        market_scores = []
        index_scores = []
        
        if company_texts:
            LOGGER.info("Analyzing %d company texts with FinBERT...", len(company_texts))
            company_scores = analyze_sentiment(company_texts, temperature=1.0)
            LOGGER.info("Company analysis complete")
        
        if market_texts:
            LOGGER.info("Analyzing %d market texts with FinBERT...", len(market_texts))
            market_scores = analyze_sentiment(market_texts, temperature=1.0)
            LOGGER.info("Market analysis complete")
        
        if index_texts:
            LOGGER.info("Analyzing %d index texts with FinBERT...", len(index_texts))
            index_scores = analyze_sentiment(index_texts, temperature=1.0)
            LOGGER.info("Index analysis complete")
        
        # Calculate averaged probabilities
        company_probs = _average_scores(company_scores)
        market_probs = _average_scores(market_scores)
        index_probs = _average_scores(index_scores)
        
        # Blend sentiments (70% company, 20% market, 10% indices)
        blended_probs = {
            "negative": (
                0.70 * company_probs["negative"]
                + 0.20 * market_probs["negative"]
                + 0.10 * index_probs["negative"]
            ),
            "neutral": (
                0.70 * company_probs["neutral"]
                + 0.20 * market_probs["neutral"]
                + 0.10 * index_probs["neutral"]
            ),
            "positive": (
                0.70 * company_probs["positive"]
                + 0.20 * market_probs["positive"]
                + 0.10 * index_probs["positive"]
            ),
        }
        
        # Determine final sentiment
        sentiment_label = max(blended_probs, key=blended_probs.get)
        prediction = _label_to_prediction(sentiment_label)
        
        LOGGER.info(
            "Sentiment analysis complete: %s (prediction=%d)",
            sentiment_label,
            prediction
        )
        
        return {
            "company_name": company_name,
            "prediction": prediction,
            "sentiment_label": sentiment_label,
            "avg_positive": round(blended_probs["positive"], 4),
            "avg_negative": round(blended_probs["negative"], 4),
            "avg_neutral": round(blended_probs["neutral"], 4),
            "company_probabilities": company_probs,
            "market_probabilities": market_probs,
            "index_probabilities": index_probs,
            "total_articles": len(company_scores) + len(market_scores) + len(index_scores),
            "source": "tools",
            "blended_sentiment": {
                "blended_string": sentiment_label,
                "blended_numeric": blended_probs["positive"] - blended_probs["negative"],
                "blended_probabilities": blended_probs,
                "company_weight": 0.70,
                "market_weight": 0.20,
                "indices_weight": 0.10,
                "components": {
                    "company": {
                        "label": max(company_probs, key=company_probs.get) if company_scores else None,
                        "probabilities": company_probs,
                        "article_count": len(company_scores),
                    },
                    "market": {
                        "label": max(market_probs, key=market_probs.get) if market_scores else None,
                        "probabilities": market_probs,
                        "article_count": len(market_scores),
                    },
                    "indices": {
                        "label": max(index_probs, key=index_probs.get) if index_scores else None,
                        "probabilities": index_probs,
                        "article_count": len(index_scores),
                    },
                },
            },
        }
        
    except Exception as exc:
        LOGGER.exception("Sentiment analysis failed: %s", exc)
        return {
            "company_name": company_name,
            "prediction": None,
            "error": f"Sentiment analysis failed: {exc}",
        }


# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #
def _average_scores(scores: List[Dict[str, float]]) -> Dict[str, float]:
    """Average sentiment scores across multiple texts."""
    if not scores:
        LOGGER.debug("No scores to average, using neutral baseline")
        return {"negative": 0.33, "neutral": 0.34, "positive": 0.33}
    
    totals = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    
    for score in scores:
        totals["negative"] += score.get("negative", 0.0)
        totals["neutral"] += score.get("neutral", 0.0)
        totals["positive"] += score.get("positive", 0.0)
    
    count = len(scores)
    return {
        "negative": round(totals["negative"] / count, 4),
        "neutral": round(totals["neutral"] / count, 4),
        "positive": round(totals["positive"] / count, 4),
    }


def _label_to_prediction(label: str) -> int:
    """
    Map sentiment label to prediction integer.
    
    Returns:
        1 = positive/bullish
        0 = neutral
        -1 = negative/bearish
    """
    normalized = (label or "").lower()
    
    if normalized in ("positive", "bullish"):
        return 1
    elif normalized in ("negative", "bearish"):
        return -1
    else:
        return 0


# --------------------------------------------------------------------------- #
# Main Interface
# --------------------------------------------------------------------------- #
def sentiment_analysis(
    company_name: str | None = None,
    news_query: str | None = None,
    news_api_key: str | None = None,
    sentiment_csv_path: str | None = None,
    days: int = 30,
) -> Dict[str, Any]:
    """
    Perform sentiment analysis using direct tools (no LLM).
    
    Priority order:
    1. CSV fallback (if provided)
    2. Tool-based analysis (NewsAPI + FinBERT + Index data)
    
    Args:
        company_name: Target company name
        news_query: Alternative query (fallback to company_name)
        news_api_key: Unused (kept for compatibility)
        sentiment_csv_path: Path to CSV fallback
        days: Unused (fixed 5-day window)
    
    Returns:
        Dict with sentiment prediction and probabilities including index data
    """
    company = company_name or news_query or "Unknown"
    LOGGER.info("Starting sentiment analysis for: %s", company)

    # CSV fallback (legacy support)
    if sentiment_csv_path and os.path.exists(sentiment_csv_path):
        LOGGER.info("Using CSV sentiment source: %s", sentiment_csv_path)
        try:
            df = pd.read_csv(sentiment_csv_path)
            if {"Positive", "Negative"}.issubset(df.columns):
                avg_positive = float(df["Positive"].mean())
                avg_negative = float(df["Negative"].mean())
                avg_neutral = float(df["Neutral"].mean()) if "Neutral" in df.columns else 0.0
                
                if avg_positive > avg_negative and avg_positive > avg_neutral:
                    label, prediction = "positive", 1
                elif avg_negative > avg_positive and avg_negative > avg_neutral:
                    label, prediction = "negative", -1
                else:
                    label, prediction = "neutral", 0
                
                LOGGER.info("CSV sentiment: %s (prediction=%d)", label, prediction)
                
                return {
                    "company_name": company,
                    "prediction": prediction,
                    "sentiment_label": label,
                    "avg_positive": avg_positive,
                    "avg_negative": avg_negative,
                    "avg_neutral": avg_neutral,
                    "company_probabilities": {
                        "positive": avg_positive,
                        "neutral": avg_neutral,
                        "negative": avg_negative,
                    },
                    "market_probabilities": {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
                    "index_probabilities": {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
                    "total_articles": len(df),
                    "source": "csv",
                }
        except Exception as exc:
            LOGGER.exception("Failed to read CSV: %s", exc)

    # Tool-based analysis
    return _analyze_with_tools(company)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    )
    
    print("\n" + "="*80)
    print("Direct Tool-Based Sentiment Analysis")
    print("No LLM Required - Uses NewsAPI + FinBERT + Index Data")
    print("="*80 + "\n")
    
    result = sentiment_analysis(company_name="Reliance Industries")
    
    print("\n" + "="*80)
    print("Sentiment Analysis Result:")
    print("="*80)
    
    for key, value in result.items():
        if key != "blended_sentiment":
            print(f"{key}: {value}")
        else:
            print(f"\n{key}:")
            blended = value
            print(f"  blended_string: {blended.get('blended_string')}")
            print(f"  blended_numeric: {blended.get('blended_numeric')}")
            print(f"  company_weight: {blended.get('company_weight')}")
            print(f"  market_weight: {blended.get('market_weight')}")
            print(f"  indices_weight: {blended.get('indices_weight')}")
            
            components = blended.get('components', {})
            if components.get('company'):
                comp = components['company']
                print(f"\n  Company Sentiment:")
                print(f"    label: {comp.get('label')}")
                print(f"    articles: {comp.get('article_count')}")
            
            if components.get('market'):
                mkt = components['market']
                print(f"\n  Market Sentiment:")
                print(f"    label: {mkt.get('label')}")
                print(f"    articles: {mkt.get('article_count')}")
            
            if components.get('indices'):
                idx = components['indices']
                print(f"\n  Index Sentiment:")
                print(f"    label: {idx.get('label')}")
                print(f"    articles: {idx.get('article_count')}")

