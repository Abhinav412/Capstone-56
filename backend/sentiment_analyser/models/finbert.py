from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

class FinBERTSentiment:
    def __init__(self):
        # Use ProsusAI/finbert as the base model
        model_name = "ProsusAI/finbert"
        
        try:
            # Load the base model structure
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
            
            # Load tokenizer (do this early so we can use it in fallback too)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load artifact directory and model path
            artifact_dir = os.getenv("artifact_dir")
            if not artifact_dir:
                raise ValueError("artifact_dir environment variable not set")
            
            model_path = os.path.join(artifact_dir, "finbert.pth")
            
            # Load the fine-tuned weights
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                print(f"Model loaded from artifact: {model_path}")
            else:
                print(f"Artifact not found at {model_path}, using base ProsusAI/finbert model...")
            
            # Set to evaluation mode
            self.model.eval()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Use a reliable fallback model (ProsusAI/finbert is reliable)
            fallback_model = "ProsusAI/finbert"
            print(f"Falling back to: {fallback_model}")
            self.model = AutoModelForSequenceClassification.from_pretrained(fallback_model, num_labels=3)
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model.eval()
    
    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()
        labels = ['negative', 'neutral', 'positive']
        scores = {label: round(float(probs[i]), 4) for i, label in enumerate(labels)}
        predicted = labels[torch.argmax(probs).item()]
        return {"sentiment": predicted, "confidence": scores}