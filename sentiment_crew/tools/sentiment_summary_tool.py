from typing import Any, Dict, List
import logging
from crewai.tools import tool

LABELS = ("negative", "neutral", "positive")


@tool("summarize_sentiment_results")
def summarize_sentiment_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate FinBERT sentiment scores into summary statistics.
    
    Args:
        results: List of dicts from finbert_sentiment_tool, each with:
            {text, negative, neutral, positive, [optional_metadata]}
    
    Returns:
        Dict with:
            - sentiment_label: "negative" | "neutral" | "positive"
            - summary: Human-readable interpretation
            - average_probabilities: {negative, neutral, positive} averaged across articles
            - sentiment_distrbution: {negative, neutral, positive} count of articles per label
            - article_scores: Original input with validated probabilities
    """

    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Summarizer invoked with result %d results", len(results) if results else 0)

    summary_payload: Dict[str, Any] = {
        "average_probabilities": {label: 0.0 for label in LABELS},
        "sentiment_distribution": {label: 0 for label in LABELS},
        "article_scores": [],
        "sentiment_label": "neutral",
        "summary": "No sentiment data available.",
    }

    if not results:
        return summary_payload

    valid_entries: List[Dict[str, Any]] = []

    for entry in results:
        if not isinstance(entry, dict):
            continue

        scores: Dict[str, float] = {}
        for label in LABELS:
            try:
                scores[label] = float(entry.get(label, 0.0))
            except (TypeError, ValueError):
                scores[label] = 0.0

        if sum(scores.values()) <= 0:
            continue

        valid_entries.append({**scores, "text": entry.get("text", ""), "metadata": entry.get("metadata")})

    if not valid_entries:
        return summary_payload

    averaged: Dict[str, float] = {label: 0.0 for label in LABELS}
    counts: Dict[str, int] = {label: 0 for label in LABELS}

    for entry in valid_entries:
        for label in LABELS:
            averaged[label] += entry[label]
        winner = max(LABELS, key=lambda lbl: entry[lbl])
        counts[winner] += 1

    total_entries = len(valid_entries)
    if total_entries:
        averaged = {label: averaged[label] / total_entries for label in LABELS}

    overall_label = max(LABELS, key=lambda lbl: averaged[lbl])
    tone_map = {"positive": "Bullish (positive)", "neutral": "Sideways (neutral)", "negative": "Bearish (negative)"}

    summary_payload.update(
        {
            "summary": f"Overall sentiment: {tone_map[overall_label]} ({overall_label}).",
            "sentiment_label": overall_label,
            "average_probabilities": averaged,
            "sentiment_distribution": counts,
            "article_scores": valid_entries,
        }
    )
    LOGGER.info(
        "Summarizer completed: label=%s, avg_probs=%s",
        summary_payload["sentiment_label"],
        summary_payload["average_probabilities"]
    )
    return summary_payload