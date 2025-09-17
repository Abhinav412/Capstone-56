from smolagents.tools import tool
from models.finbert import FinBERTSentiment

@tool
def analyze_sentiment(text: str) -> dict:
    """
    Analyze the sentiment of a given text using the FinBERT model.

    Args:
        text (str): The text content to analyze.

    Returns:
        dict: A dictionary with sentiment label ('positive', 'neutral', 'negative') and confidence score.
    """
    return FinBERTSentiment(text)
