import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import torch
from dotenv import load_dotenv
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from crewai.tools import tool

load_dotenv()
LOGGER = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
LABELS = ["negative", "neutral", "positive"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def _load_artifacts(model_name: str, checkpoint_path: Optional[str]):
    LOGGER.debug("FinBERT: loading model=%s checkpoint=%s", model_name, checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(LABELS))

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            LOGGER.info("Loaded FinBERT checkpoint from %s", checkpoint_path)
        except Exception as exc:
            LOGGER.warning("Failed to load checkpoint %s: %s. Using base weights.", checkpoint_path, exc)
    elif checkpoint_path:
        LOGGER.warning("FinBERT checkpoint not found at %s; using base weights.", checkpoint_path)

    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def _softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    temperature = max(float(temperature), 1e-3)
    return nn.functional.softmax(logits / temperature, dim=-1)


def analyze_sentiment(texts: List[str], temperature: float = 1.0) -> List[Dict[str, float]]:
    if not texts:
        LOGGER.debug("FinBERT: received empty text batch.")
        return []

    start_time = time.time()
    LOGGER.debug(
        "FinBERT: scoring %d texts (temperature=%s). Preview=%s",
        len(texts),
        temperature,
        texts[0][:120] if texts else "",
    )

    artifact_dir = os.getenv("ARTIFACT_DIR") or os.getenv("artifact_dir")
    default_checkpoint = (
        Path(artifact_dir) / "finbert.pth"
        if artifact_dir
        else Path(r"c:\PES\Projects\Capstone-56\artifacts\finbert\finbert.pth")
    )
    checkpoint_path = os.getenv("FINBERT_CHECKPOINT_PATH") or str(default_checkpoint)

    tokenizer, model = _load_artifacts(MODEL_NAME, checkpoint_path)

    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(DEVICE)

    try:
        with torch.no_grad():
            logits = model(**encodings).logits
            probabilities = _softmax_with_temperature(logits, temperature).cpu().tolist()
    except Exception as exc:
        LOGGER.exception("FinBERT forward pass failed: %s", exc)
        raise

    results: List[Dict[str, float]] = []
    for text, score_vector in zip(texts, probabilities):
        sentiment = {"text": text}
        for label, score in zip(LABELS, score_vector):
            sentiment[label] = round(float(score), 4)
        results.append(sentiment)

    LOGGER.debug(
        "FinBERT: completed batch in %.2fs (first scores=%s)",
        time.time() - start_time,
        results[0] if results else {},
    )
    return results


@tool("finbert_sentiment_tool")
def finbert_sentiment_tool(texts: List[str], temperature: float = 1.0) -> List[Dict[str, float]]:
    """
    Score sentiment for a batch of text snippets using FinBERT.

    Args:
        texts: List of article summaries or headlines to analyze.
        temperature: Softmax temperature (default 1.0)
    
    Returns:
        List of dicts with keys: text, negative, neutral, positive (probabilities sum to 1.0)
    
    Example:
        Input: ["Stock market rallies on GDP data", "Company misses earnings"]
        Output: [
            {"text": "Stock market rallies...", "negative": 0.05, "neutral": 0.15, "positive": 0.80},
            {"text": "Company misses...", "negative": 0.85, "neutral": 0.10, "positive": 0.05}
        ]
    """
    if not isinstance(texts, list):
        LOGGER.warning("FinBERT tool received non-list input: %s. Wrapping in list.", type(texts))
        texts = [str(texts)]
    
    texts = [str(t).strip() for t in texts if t and str(t).strip()]

    if not texts:
        LOGGER.warning("FinBERT tool recieved empty text list after filtering.")
        return []
    
    LOGGER.info("FinBERT tool invoked with %d texts.", len(texts))

    try:
        results = analyze_sentiment(texts, temperature=temperature)
        LOGGER.info("FinBERT tool completed successfully: %d results", len(results))
        return results
    except Exception as exc:
        LOGGER.exception("FinBERT tool failed: %s", exc)
        return [
            {"text": text, "negative": 0.33, "neutral": 0.34, "positve": 0.33}
            for text in texts
        ]