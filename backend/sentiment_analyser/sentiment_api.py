from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    scores: dict

# Get model path from environment variable with fallback
ARTIFACT_DIR = os.getenv(
    "artifact_dir",
    os.path.join(os.path.dirname(__file__), "artifacts", "finbert-augmented-v1")
)

# Resolve and validate path
model_path = Path(ARTIFACT_DIR).resolve()
if not model_path.exists():
    raise FileNotFoundError(f"Model artifact directory not found at: {model_path}")

# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your FinBERT model once at startup
print(f"Loading FinBERT model from: {model_path}")
print(f"Using device: {device}")

try:
    # Load base model structure
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert",
        num_labels=3
    )
    
    # Load fine-tuned weights if available
    weights_file = model_path / "finbert.pth"
    if weights_file.exists():
        state_dict = torch.load(weights_file, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"✅ Fine-tuned weights loaded from: {weights_file}")
    else:
        print("⚠️ Using base ProsusAI/finbert model (fine-tuned weights not found)")
    
    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    print("✅ FinBERT model loaded successfully")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

@app.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze sentiment of financial text using FinBERT.
    
    Args:
        request: SentimentRequest containing text to analyze
        
    Returns:
        SentimentResponse with sentiment label, confidence, and scores
    """
    try:
        # Tokenize input
        inputs = tokenizer(
            request.text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        # Move inputs to same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = F.softmax(outputs.logits, dim=-1)
        
        # Move predictions back to CPU for processing
        predictions = predictions.cpu()
        
        # FinBERT label order: [positive, negative, neutral]
        labels = ["negative", "neutral", "positive"]
        scores_dict = {labels[i]: float(predictions[0][i]) for i in range(len(labels))}
        
        # Get highest confidence prediction
        max_idx = torch.argmax(predictions[0]).item()
        sentiment = labels[max_idx]
        confidence = float(predictions[0][max_idx])
        
        return SentimentResponse(
            sentiment=sentiment,
            confidence=confidence,
            scores=scores_dict
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API and model status"""
    return {
        "status": "healthy",
        "model_path": str(model_path),
        "device": str(device),
        "model_loaded": model is not None
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "FinBERT Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze-sentiment": "Analyze sentiment of financial text",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)