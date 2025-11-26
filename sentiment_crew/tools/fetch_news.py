import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Union
from newsapi import NewsApiClient
from tavily import TavilyClient
from crewai.tools import tool
from dotenv import load_dotenv
import json
import logging

load_dotenv()

# Simple in-memory caches for the current process/run
NEWS_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
MARKET_CACHE: Dict[str, Dict[str, Any]] = {}

# Simple retry/backoff helper
def _with_retries(func, *args, retries: int = 3, backoff_sec: float = 1.0, **kwargs):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            sleep = backoff_sec * (2 ** (attempt - 1))
            time.sleep(sleep)
    raise last_exc

def _normalize_article(source: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    # Attempt to extract canonical fields from different providers
    title = (raw.get("title") or raw.get("headline") or "").strip()
    url = raw.get("url") or raw.get("link") or ""
    desc = raw.get("description") or raw.get("content") or raw.get("snippet") or ""
    published = (
        raw.get("publishedAt")
        or raw.get("published_at")
        or raw.get("published")
        or raw.get("date")
        or raw.get("time")
        or None
    )
    # Trim/clean summary for FinBERT; keep first 400 chars
    summary = " ".join(desc.split())[:400].strip()
    return {
        "source": source,
        "title": title,
        "summary": summary,
        "url": url,
        "published_at": published,
        "raw": raw,
    }

def _dedupe_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_urls = set()
    seen_titles = set()
    out: List[Dict[str, Any]] = []
    for a in articles:
        url = (a.get("url") or "").strip()
        title = (a.get("title") or "").strip()
        key = url or title
        if not key:
            continue
        if url and url in seen_urls:
            continue
        if title and title in seen_titles:
            continue
        if url:
            seen_urls.add(url)
        if title:
            seen_titles.add(title)
        out.append(a)
    return out

def _normalize_request(
    payload: Union[str, Dict[str, Any], List[Any]],
    fallback_days: int,
) -> tuple[str, int]:
    """
    Accepts the flexible payload coming from the agent/tooling layer and returns a clean
    (company_name, days_back) tuple.
    """
    original = payload
    company = ""
    days_back = fallback_days

    # Allow the LLM to send JSON as a string
    if isinstance(payload, str):
        stripped = payload.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                # Treat it as a literal company name
                company = stripped
        else:
            company = stripped

    if isinstance(payload, dict):
        company = payload.get("company_name") or payload.get("name") or company
        days_back = payload.get("days_back", days_back)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and item.get("company_name"):
                company = item["company_name"]
                days_back = item.get("days_back", days_back)
                break
    elif not isinstance(payload, str):
        company = str(payload or "")

    if not company:
        company = str(original or "")

    return company.strip(), int(days_back or fallback_days)


LOGGER = logging.getLogger(__name__)


@tool("News fetcher")
def fetch_news_tool(company_name: Union[str, Dict[str, Any], List[Any]], days_back: int = 2) -> Dict[str, Any]:
    """
    Fetches company-specific and market-wide news articles using Tavily and NewsAPI.

    Accepts flexible input (raw string, dict, or list of dicts) to match LLM output.
    - Respects days_back window.
    - Caches results in-process to avoid repeated API hits.
    - Normalizes articles to include: title, summary, url, published_at, source.
    - Deduplicates by URL and title.
    - Handles missing API keys gracefully and returns structured warnings/errors.
    """
    # Normalize the incoming request payload first
    normalized_company, normalized_days = _normalize_request(company_name, days_back)
    LOGGER.info("News fetcher invoked: company=%s, days_back=%d", normalized_company, normalized_days)
    company_name = normalized_company or "Unknown"
    days_back = normalized_days if normalized_days > 0 else days_back

    warnings: List[str] = []
    errors: List[str] = []

    news_api_key = os.getenv("NEWS_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not news_api_key and not tavily_api_key:
        return {
            "company_news": [],
            "market_news": [],
            "n_company_articles": 0,
            "n_market_articles": 0,
            "warnings": ["No NEWS_API_KEY or TAVILY_API_KEY found in environment; no external fetch attempted."],
            "errors": [],
        }

    news_api = NewsApiClient(api_key=news_api_key) if news_api_key else None
    tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None

    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    # Disambiguation and relevance boosting for Indian finance context
    company_query = (
        f'"{company_name}" AND (India OR Indian OR NSE OR BSE OR stock OR share OR results OR company)'
    )

    # --- Company-specific fetch (cached) ---
    cache_key = (company_query, from_date)
    company_articles: List[Dict[str, Any]] = []
    if cache_key in NEWS_CACHE:
        company_articles = NEWS_CACHE[cache_key]["company_articles"]
    else:
        raw_articles: List[Dict[str, Any]] = []

        # Tavily
        if tavily_client:
            try:
                tavily_results = _with_retries(
                    tavily_client.search, query=company_query, include_domains=None
                )
                for r in tavily_results.get("results", []):
                    raw_articles.append(r)
            except Exception as exc:
                warnings.append(f"Tavily fetch failed: {exc}")

        # NewsAPI
        if news_api:
            try:
                resp = _with_retries(
                    news_api.get_everything,
                    q=company_query,
                    from_param=from_date,
                    language="en",
                    sort_by="relevancy",
                    page_size=25,
                    domains="moneycontrol.com,economictimes.indiatimes.com,livemint.com,businesstoday.in,reuters.com,bloomberg.com",
                )
                raw_articles.extend(resp.get("articles", []) if isinstance(resp, dict) else [])
            except Exception as exc:
                warnings.append(f"NewsAPI company fetch failed: {exc}")

        # Normalize & dedupe
        normalized = [
            _normalize_article(
                "Tavily"
                if r.get("source") is None and r.get("url") and "tavily" in (r.get("url") or "")
                else ("NewsAPI" if r.get("url") else "Tavily"),
                r,
            )
            for r in raw_articles
        ]
        deduped = _dedupe_articles(normalized)
        # Keep only those within days_back window if published_at looks parseable (best-effort)
        filtered: List[Dict[str, Any]] = []
        cutoff = datetime.utcnow() - timedelta(days=days_back)
        for a in deduped:
            p = a.get("published_at")
            keep = True
            if p:
                try:
                    parsed = (
                        datetime.fromisoformat(p.replace("Z", "+00:00"))
                        if "T" in p
                        else datetime.strptime(p, "%Y-%m-%d %H:%M:%S")
                    )
                    if parsed < cutoff:
                        keep = False
                except Exception:
                    keep = True
            if keep:
                filtered.append(a)

        company_articles = filtered
        NEWS_CACHE[cache_key] = {
            "company_articles": company_articles,
            "fetched_at": datetime.utcnow().isoformat(),
        }

    # --- Market-wide fetch (cached) ---
    market_cache_key = from_date
    market_articles: List[Dict[str, Any]] = []
    if market_cache_key in MARKET_CACHE:
        market_articles = MARKET_CACHE[market_cache_key]["market_articles"]
    else:
        raw_market: List[Dict[str, Any]] = []
        market_query = "Indian stock market OR NIFTY OR SENSEX OR India VIX OR India MMI OR NSE OR BSE"

        if news_api:
            try:
                resp = _with_retries(
                    news_api.get_everything,
                    q=market_query,
                    from_param=from_date,
                    language="en",
                    sort_by="relevancy",
                    page_size=25,
                    domains="moneycontrol.com,economictimes.indiatimes.com,livemint.com,businesstoday.in,reuters.com,bloomberg.com",
                )
                raw_market.extend(resp.get("articles", []) if isinstance(resp, dict) else [])
            except Exception as exc:
                warnings.append(f"NewsAPI market fetch failed: {exc}")

        if tavily_client:
            try:
                tavily_results = _with_retries(
                    tavily_client.search, query=market_query, include_domains=None
                )
                raw_market.extend(
                    tavily_results.get("results", []) if isinstance(tavily_results, dict) else []
                )
            except Exception as exc:
                warnings.append(f"Tavily market fetch failed: {exc}")

        normalized_market = [
            _normalize_article("NewsAPI" if r.get("url") else "Tavily", r)
            for r in raw_market
        ]
        deduped_market = _dedupe_articles(normalized_market)
        MARKET_CACHE[market_cache_key] = {
            "market_articles": deduped_market,
            "fetched_at": datetime.utcnow().isoformat(),
        }
        market_articles = deduped_market

    result = {
        "company_news": company_articles,
        "company_fetched_at": NEWS_CACHE[cache_key]["fetched_at"],
        "market_news": market_articles,
        "market_fetched_at": MARKET_CACHE[market_cache_key]["fetched_at"],
        "n_company_articles": len(company_articles),
        "n_market_articles": len(market_articles),
        "warnings": warnings,
        "errors": errors,
    }
    LOGGER.info(
        "News fetcher completed: company_articles=%d, market_articles=%d",
        len(company_articles),
        len(market_articles)
    )
    return result
