import logging
import time
from typing import Any, Dict, List

from crewai import Crew

from .agents.sentiment_agent import sentiment_analyser
from .tasks.analyze_task import analyze_task
from .tools.blended_sentiment_tool import BlendedSentimentTool

LOGGER = logging.getLogger(__name__)

crew = Crew(
    agents=[sentiment_analyser],
    tasks=[analyze_task],
    verbose=True,
)

blender = BlendedSentimentTool(company_weight=0.70, market_weight=0.25, indices_weight=0.05)


def _normalize_result_payload(crew_result: Any) -> Dict[str, Any]:
    """Extract dict from CrewAI result object."""
    if isinstance(crew_result, dict):
        return crew_result

    if hasattr(crew_result, "json") and callable(crew_result.json):
        try:
            data = crew_result.json()
            if isinstance(data, dict):
                LOGGER.debug("Extracted JSON from crew_result.json()")
                return data
        except Exception as exc:
            LOGGER.warning("Failed to parse crew_result.json(): %s", exc)

    if hasattr(crew_result, "raw_output") and isinstance(crew_result.raw_output, dict):
        LOGGER.debug("Using crew_result.raw_output")
        return crew_result.raw_output

    if hasattr(crew_result, "output") and isinstance(crew_result.output, dict):
        LOGGER.debug("Using crew_result.output")
        return crew_result.output

    LOGGER.warning("Could not extract dict from crew_result, returning empty dict")
    return {}


def run_sentiment(company_name: str) -> Dict[str, Any]:
    """Execute crew workflow and blend sentiment results."""
    LOGGER.info("=== Starting crew sentiment analysis for: %s ===", company_name)
    time.sleep(2)

    # Run crew
    crew_result = crew.kickoff(inputs={"company_name": company_name})
    structured_payload = _normalize_result_payload(crew_result)
    
    LOGGER.info("Crew execution completed. Payload keys: %s", list(structured_payload.keys()))
    
    # Log raw crew output for debugging
    LOGGER.debug("Raw crew payload: %s", structured_payload)
    
    # Blend sentiment
    blended_result = blender.run(structured_payload)
    
    LOGGER.info(
        "=== Crew sentiment complete: label=%s, blended_numeric=%.4f ===",
        blended_result.get("blended_string"),
        blended_result.get("blended_numeric", 0.0)
    )
    
    return blended_result


if __name__ == "__main__":
    company_name = input("Enter a company name: ").strip() or "Mahindra & Mahindra"
    blended_result = run_sentiment(company_name)
    print("\nBlended Sentiment Output")
    print(blended_result)