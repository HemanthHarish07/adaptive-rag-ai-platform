import os
import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Any

try:
    from .dataset import ID_MAPPING, LABEL_MAPPING
except ImportError:
    from dataset import ID_MAPPING, LABEL_MAPPING


class IntentClassifier:
    """
    A high-level wrapper around the fine-tuned intent classifier.
    Falls back to a lightweight rule-based model when artifacts are missing.
    """
    def __init__(self, model_dir: str = "models/intent_classifier"):
        self.model_dir = os.path.abspath(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fallback = False
        self.model = None
        self.tokenizer = None

        if os.path.exists(self.model_dir) and os.path.isdir(self.model_dir):
            try:
                print(f"Loading intent classifier from: {self.model_dir} on device: {self.device.upper()}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
                self.model.to(self.device)
                self.model.eval()
            except Exception as exc:
                print(f"Warning: Failed to load fine-tuned model, switching to fallback: {exc}")
                self.use_fallback = True
        else:
            print(f"Model directory not found at {self.model_dir}. Using fallback classifier.")
            self.use_fallback = True

    def _fallback_predict(self, text: str) -> Dict[str, Any]:
        normalized = text.strip().lower()
        if not normalized:
            return {
                "query": text,
                "label": "unknown",
                "label_id": -1,
                "confidence": 0.0,
                "probabilities": {name: 0.0 for name in LABEL_MAPPING.keys()},
            }

        debugging_keywords = ["debug", "error", "exception", "traceback", "bug", "fail", "stack trace", "crash"]
        if any(word in normalized for word in debugging_keywords):
            label = "debugging"
            confidence = 0.92
        else:
            label = "theory"
            confidence = 0.84

        probs = {name: 0.0 for name in LABEL_MAPPING.keys()}
        probs[label] = confidence
        other = [k for k in LABEL_MAPPING.keys() if k != label]
        if other:
            probs[other[0]] = 1.0 - confidence

        return {
            "query": text,
            "predicted_intent": label,
            "label": label,
            "label_id": LABEL_MAPPING.get(label, -1),
            "confidence": confidence,
            "confidence_score": confidence,
            "probabilities": probs,
        }

    def predict(self, text: str) -> Dict[str, Any]:
        if self.use_fallback or self.model is None or self.tokenizer is None:
            return self._fallback_predict(text)

        if not text.strip():
            return self._fallback_predict(text)

        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]

        pred_id = torch.argmax(probs).item()
        confidence = probs[pred_id].item()
        pred_label = ID_MAPPING[pred_id]
        probs_dict = {ID_MAPPING[i]: probs[i].item() for i in range(len(ID_MAPPING))}

        return {
            "query": text,
            "predicted_intent": pred_label,
            "label": pred_label,
            "label_id": pred_id,
            "confidence": confidence,
            "confidence_score": confidence,
            "probabilities": probs_dict,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        if self.use_fallback or self.model is None or self.tokenizer is None:
            return [self._fallback_predict(text) for text in texts]

        cleaned_texts = [t.strip() if t.strip() else "" for t in texts]
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
            probs = torch.softmax(outputs.logits, dim=-1)

        results = []
        for idx, text in enumerate(texts):
            if not text.strip():
                results.append(self._fallback_predict(text))
                continue

            item_probs = probs[idx]
            pred_id = torch.argmax(item_probs).item()
            confidence = item_probs[pred_id].item()
            pred_label = ID_MAPPING[pred_id]
            probs_dict = {ID_MAPPING[i]: item_probs[i].item() for i in range(len(ID_MAPPING))}

            results.append({
                "query": text,
                "predicted_intent": pred_label,
                "label": pred_label,
                "label_id": pred_id,
                "confidence": confidence,
                "confidence_score": confidence,
                "probabilities": probs_dict,
            })

        return results


def main():
    parser = argparse.ArgumentParser(description="Test Intent Classifier on a custom query")
    parser.add_argument("query", type=str, nargs="?", default=None, help="The query text to classify")
    parser.add_argument("--model_dir", type=str, default="models/intent_classifier", help="Directory where trained model is saved")

    args = parser.parse_args()

    classifier = IntentClassifier(args.model_dir)

    if not args.query:
        print("Starting interactive testing mode. Type 'exit' to quit.")
        while True:
            q = input("\nEnter query > ").strip()
            if q.lower() == 'exit':
                break
            if not q:
                continue
            res = classifier.predict(q)
            print(f"Predicted Intent: {res['label']} (Confidence: {res['confidence']:.4f})")
            print("Probabilities:")
            sorted_probs = sorted(res["probabilities"].items(), key=lambda x: x[1], reverse=True)
            for label, prob in sorted_probs:
                print(f"  - {label}: {prob:.4f}")
    else:
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


if __name__ == "__main__":
    main()
