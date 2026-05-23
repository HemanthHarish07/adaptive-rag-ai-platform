import os
import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Any, Union

from .dataset import ID_MAPPING, LABEL_MAPPING


class IntentClassifier:
    """
    A high-level wrapper around the fine-tuned intent classifier
    for production inference.
    """
    def __init__(self, model_dir: str = "models/intent_classifier"):
        # Keep model_dir as provided by caller, but normalize for safety.
        self.model_dir = os.path.abspath(model_dir)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not os.path.exists(self.model_dir):

            raise FileNotFoundError(
                f"Model directory not found at: {self.model_dir}. Please train the model first."
            )

            
        print(
            f"Loading intent classifier from: {self.model_dir} on device: {self.device.upper()}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)

        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predicts the intent of a single input query.
        """
        if not text.strip():
            return {
                "query": text,
                "label": "unknown",
                "label_id": -1,
                "confidence": 0.0,
                "probabilities": {name: 0.0 for name in LABEL_MAPPING.keys()}
            }
            
        # Tokenize query
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Move tensors to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            
        # Get prediction details
        pred_id = torch.argmax(probs).item()
        confidence = probs[pred_id].item()
        pred_label = ID_MAPPING[pred_id]
        
        # Map all probabilities
        probs_dict = {ID_MAPPING[i]: probs[i].item() for i in range(len(ID_MAPPING))}
        
        return {
            "query": text,
            "label": pred_label,
            "label_id": pred_id,
            "confidence": confidence,
            "probabilities": probs_dict
        }
        
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predicts the intents for a batch of input queries.
        """
        cleaned_texts = [t.strip() if t.strip() else "" for t in texts]
        
        # Tokenize batch
        inputs = self.tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
        results = []
        for idx, text in enumerate(texts):
            if not text.strip():
                results.append({
                    "query": text,
                    "label": "unknown",
                    "label_id": -1,
                    "confidence": 0.0,
                    "probabilities": {name: 0.0 for name in LABEL_MAPPING.keys()}
                })
                continue
                
            item_probs = probs[idx]
            pred_id = torch.argmax(item_probs).item()
            confidence = item_probs[pred_id].item()
            pred_label = ID_MAPPING[pred_id]
            probs_dict = {ID_MAPPING[i]: item_probs[i].item() for i in range(len(ID_MAPPING))}
            
            results.append({
                "query": text,
                "label": pred_label,
                "label_id": pred_id,
                "confidence": confidence,
                "probabilities": probs_dict
            })
            
        return results

def main():
    parser = argparse.ArgumentParser(description="Test Intent Classifier on a custom query")
    parser.add_argument("query", type=str, nargs="?", default=None, help="The query text to classify")
    parser.add_argument("--model_dir", type=str, default="models/intent_classifier", help="Directory where trained model is saved")
    
    args = parser.parse_args()
    
    # If no query is provided, run interactive prompt
    if not args.query:
        print("Starting interactive testing mode. Type 'exit' to quit.")
        try:
            classifier = IntentClassifier(args.model_dir)
            while True:
                q = input("\nEnter query > ").strip()
                if q.lower() == 'exit':
                    break
                if not q:
                    continue
                res = classifier.predict(q)
                print(f"Predicted Intent: {res['label']} (Confidence: {res['confidence']:.4f})")
                print("Probabilities:")
                # Sort probabilities in descending order
                sorted_probs = sorted(res["probabilities"].items(), key=lambda x: x[1], reverse=True)
                for label, prob in sorted_probs:
                    print(f"  - {label}: {prob:.4f}")
        except KeyboardInterrupt:
            print("\nExiting interactive testing.")
        except Exception as e:
            print(f"Error: {e}")
    else:
        # Predict single query
        try:
            classifier = IntentClassifier(args.model_dir)
            res = classifier.predict(args.query)
            print("\n" + "="*40)
            print(f"Query: {res['query']}")
            print(f"Intent: {res['label']}")
            print(f"Confidence: {res['confidence']:.4f}")
            print("-"*40)
            print("All probabilities:")
            sorted_probs = sorted(res["probabilities"].items(), key=lambda x: x[1], reverse=True)
            for label, prob in sorted_probs:
                print(f"  {label:<20}: {prob:.4f}")
            print("="*40)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
