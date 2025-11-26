import yfinance as yf
from datetime import datetime
from typing import Dict, Any
from crewai.tools import tool

CACHE: Dict[str, Dict[str, Any]] = {}

def _cached_history(ticker: str, period: str = "5d"):
    key = f"{ticker}:{period}"
    if key in CACHE:
        return CACHE[key]["data"]
    data = yf.Ticker(ticker).history(period=period)
    CACHE[key] = {"data":data, "fetched_at": datetime.utcnow().isoformat()}
    return data

def _sentiment_from_change(change_pct: float) -> str:
    if change_pct >= 0.5:
        return "bullish"
    if change_pct <= -0.5:
        return "bearish"
    return "neutral"

def _index_sentiment_to_probabilities(change_pct: float) -> Dict[str, float]:
    """
    Convert index percentage change to sentiment probabilities

    Logic:
        - Strong positive (>1.5%): 10% neg, 20% neu, 70% pos
        - Moderate positive (0.5-1.5%): 15% neg, 35% neu, 50% pos
        - Neutral (-0.5 to 0.5%): 25% neg, 50% neu, 25% pos
        - Moderate negative (-1.5 to -0.5%): 50% neg, 35% neu, 15% pos
        - Strong negative (<1.5%): 70% neg, 20% neu, 10% pos
    """
    if change_pct > 1.5:
        return {"negative": 0.10, "neutral": 0.20, "positive": 0.70}
    elif change_pct > 0.5:
        return {"negative": 0.15, "neutral": 0.35, "positive": 0.50}
    elif change_pct >= -0.5:
        return {"negative": 0.25, "neutral": 0.50, "positive": 0.25}
    elif change_pct >= -1.5:
        return {"negative": 0.50, "neutral": 0.35, "positive": 0.15}
    else:
        return {"negative": 0.70, "neutral": 0.20, "positive": 10}

@tool("Indices fetcher")
def fetch_indices_tool():
    """Fetch key Indian market indices with sentiment labels and probabilities."""
    indices = {
        "S&P BSE SENSEX" : "^BSESN",
        "NSE Nifty 50" : "^NSEI",
        "India VIX" : "^INDIAVIX",
        "Nifty Bank" : "^NSEBANK",
        "Nifty IT" : "^CNXIT",
        "Nifty Pharma" : "^CNXPHARMA",
        "Nifty Auto" : "^CNXAUTO",
        "Nifty Financial Services" : "NIFTY_FIN_SERVICE.NS",
        "S&P BSE 200 Index" : "BSE-200.BO",
        "S&P BSE 500 Index" : "BSE-500.BO"
    }

    result: Dict[str, Dict[str, Any]] = {}

    for name, ticker in indices.items():
        try:
            data = _cached_history(ticker)
            if data.empty or data["Close"].shape[0] < 2:
                result[name] = {"Warning": "Insufficient close prices for sentiment"}
                continue

            last = float(data["Close"].iloc[-1])
            prev = float(data["Close"].iloc[-2])
            change_pct = ((last-prev) / prev) * 100
            sentiment = _sentiment_from_change(change_pct)
            result[name] = {
                "last_timestamp": data.index[-1].isoformat(),
                "last_close" : round(last, 2),
                "change_pct" : round(change_pct, 2),
                "sentiment": sentiment,
                "probabilities": _index_sentiment_to_probabilities(change_pct),
                "fetched_at": CACHE[f"{ticker}:5d"]["fetched_at"],
            }
        except Exception as e:
            result[name] = {"error" : str(e)}
    
    return result