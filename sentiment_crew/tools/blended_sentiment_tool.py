import logging
from typing import Any, Dict

LOGGER = logging.getLogger(__name__)


class BlendedSentimentTool:
    """
    CrewAI Tool for computing blended company + macro sentiment.
    Returns:
        - blended_numeric: -1/0/1 (coarse sentiment)
        - blended_string: negative/neutral/positive or bearish/neutral/bullish
        - company_weight / macro_weight
        - components: detailed company + market breakdown
    """

    def __init__(self, company_weight: float = 0.70, market_weight: float = 0.25, indices_weight: float = 0.05):
        self.company_weight = company_weight
        self.market_weight = market_weight
        self.indices_weight = indices_weight
        LOGGER.info(
            "BlendedSentimentTool initialized: company=%.2f, market=%.2f, indices=%.2f",
            company_weight, market_weight, indices_weight
        )

    def _extract_probabilities(self, section: Any) -> Dict[str, float]:
        """Extract {negative, neutral, positive} from various payload formats."""
        if not isinstance(section, dict):
            return {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        
        # Try nested paths
        probs = (
            section.get("probabilities")
            or section.get("average_probabilities")
            or {}
        )
        
        if not isinstance(probs, dict):
            return {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        
        return {
            "negative": float(probs.get("negative", 0.0)),
            "neutral": float(probs.get("neutral", 0.0)),
            "positive": float(probs.get("positive", 0.0)),
        }

    def _aggregate_index_probabilities(self, indices: Dict[str, Any]) -> Dict[str, float]:
        """Average probabilities across all market indices."""
        if not indices:
            return {"negative": 0.33, "neutral": 0.34, "positive": 0.33}
        
        totals = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        count = 0
        
        for idx_data in indices.values():
            if isinstance(idx_data, dict) and "probabilities" in idx_data:
                probs = idx_data["probabilities"]
                totals["negative"] += probs.get("negative", 0.0)
                totals["neutral"] += probs.get("neutral", 0.0)
                totals["positive"] += probs.get("positive", 0.0)
                count += 1
        
        if count == 0:
            return {"negative": 0.33, "neutral": 0.34, "positive": 0.33}
        
        return {k: v / count for k, v in totals.items()}

    # Crew tool entry point
    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        LOGGER.info("[BlendedSentimentTool] Running blended sentiment computation...")
        
        # Extract components
        company_sentiment = payload.get("company_sentiment") or payload.get("components", {}).get("company", {})
        market_sentiment = payload.get("market_sentiment") or payload.get("components", {}).get("market", {})
        market_indices = payload.get("market_indices", {})
        
        # Get probabilities
        company_probs = self._extract_probabilities(company_sentiment)
        market_probs = self._extract_probabilities(market_sentiment)
        indices_probs = self._aggregate_index_probabilities(market_indices)
        
        LOGGER.debug("Company probs: %s", company_probs)
        LOGGER.debug("Market probs: %s", market_probs)
        LOGGER.debug("Indices probs: %s", indices_probs)
        
        # Weighted blend
        blended_probs = {
            "negative": (
                self.company_weight * company_probs["negative"]
                + self.market_weight * market_probs["negative"]
                + self.indices_weight * indices_probs["negative"]
            ),
            "neutral": (
                self.company_weight * company_probs["neutral"]
                + self.market_weight * market_probs["neutral"]
                + self.indices_weight * indices_probs["neutral"]
            ),
            "positive": (
                self.company_weight * company_probs["positive"]
                + self.market_weight * market_probs["positive"]
                + self.indices_weight * indices_probs["positive"]
            ),
        }
        
        # Determine label
        blended_label = max(blended_probs, key=blended_probs.get)
        blended_numeric = blended_probs["positive"] - blended_probs["negative"]
        
        LOGGER.info(
            "[BlendedSentimentTool] Blended: label=%s, numeric=%.4f, probs=%s",
            blended_label, blended_numeric, blended_probs
        )
        
        return {
            "blended_string": blended_label,
            "blended_numeric": round(blended_numeric, 4),
            "blended_probabilities": blended_probs,
            "company_weight": self.company_weight,
            "market_weight": self.market_weight,
            "indices_weight": self.indices_weight,
            "components": {
                "company": {"label": company_sentiment.get("sentiment_label"), "probabilities": company_probs},
                "market": {"label": market_sentiment.get("sentiment_label"), "probabilities": market_probs},
                "indices": {"probabilities": indices_probs},
            },
        }
