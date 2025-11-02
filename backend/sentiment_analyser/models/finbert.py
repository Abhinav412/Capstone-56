from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from typing import Dict, Union

class FinBERTSentiment:
    def __init__(self, artifact_dir: str = None):
        """
        Initialize FinBERT sentiment analyzer.
        
        Args:
            artifact_dir: Optional path to artifact directory containing finbert.pth
        """
        # Use ProsusAI/finbert as the base model
        model_name = "ProsusAI/finbert"
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        try:
            # Load the base model structure
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=3
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Try to load fine-tuned weights if artifact_dir is provided
            if artifact_dir and os.path.exists(artifact_dir):
                model_path = os.path.join(artifact_dir, "finbert.pth")
                
                if os.path.exists(model_path):
                    state_dict = torch.load(
                        model_path, 
                        map_location=self.device,
                        weights_only=True
                    )
                    self.model.load_state_dict(state_dict)
                    print(f"✓ Fine-tuned model loaded from: {model_path}")
                else:
                    print(f"⚠ Artifact not found at {model_path}")
                    print(f"Using base ProsusAI/finbert model...")
            else:
                print("⚠ No artifact_dir provided, using base ProsusAI/finbert model...")
            
            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            print("✓ FinBERT model initialized successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict(self, text: str) -> Dict[str, Union[str, Dict[str, float]]]:
        """
        Predict sentiment for the given text.
        
        Args:
            text: Financial text to analyze
            
        Returns:
            Dictionary containing sentiment and confidence scores
        """
        if not text or not text.strip():
            return {
                "sentiment": "neutral",
                "confidence": {"negative": 0.0, "neutral": 0.0, "positive": 0.0},
                "error": "Empty text provided"
            }
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()
            
            # Move results back to CPU for processing
            probs = probs.cpu()
            
            # Define labels (must match model training order)
            labels = ['positive', 'negative', 'neutral']
            
            # Calculate scores
            scores = {
                label: round(float(probs[i]), 4) 
                for i, label in enumerate(labels)
            }
            
            # Get predicted sentiment
            predicted_idx = torch.argmax(probs).item()
            predicted = labels[predicted_idx]
            
            return {
                "sentiment": predicted,
                "confidence": scores
            }
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return {
                "sentiment": "neutral",
                "confidence": {"negative": 0.33, "neutral": 0.34, "positive": 0.33},
                "error": str(e)
            }
    
    def predict_batch(self, texts: list) -> list:
        """
        Predict sentiment for multiple texts in batch.
        
        Args:
            texts: List of financial texts to analyze
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text) for text in texts]
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'model') and self.device.type == 'cuda':
            # Clear CUDA cache if using GPU
            torch.cuda.empty_cache()