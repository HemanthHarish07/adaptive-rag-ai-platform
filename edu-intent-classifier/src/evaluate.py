import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader

from .dataset import load_and_prepare_dataset, LABEL_MAPPING, ID_MAPPING


def plot_confusion_matrix(cm, classes, save_path):
    """
    Plots and saves a beautiful, publication-quality confusion matrix using Seaborn.
    """
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    
    # Custom harmonious color palette
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=classes, 
        yticklabels=classes,
        cbar=True,
        square=True,
        annot_kws={"size": 12, "weight": "bold"},
        linewidths=0.5,
        linecolor="gray"
    )
    
    plt.title("Confusion Matrix - Intent Classifier", fontsize=16, pad=20, weight="bold")
    plt.ylabel("True Intent Class", fontsize=12, labelpad=10, weight="bold")
    plt.xlabel("Predicted Intent Class", fontsize=12, labelpad=10, weight="bold")
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix visualization saved to: {os.path.abspath(save_path)}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Educational Intent Classifier")
    parser.add_argument("--model_dir", type=str, default="models/intent_classifier", help="Directory where trained model is saved")
    parser.add_argument("--dataset_path", type=str, default="data/raw/intent_dataset.csv", help="Path to raw CSV dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("="*60)
    print("EDUCATIONAL INTENT CLASSIFIER - EVALUATION STAGE")
    print("="*60)
    
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Trained model directory not found at: {args.model_dir}")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    
    # Load model and tokenizer
    print(f"Loading fine-tuned model from: {args.model_dir}...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()
    
    # Load evaluation dataset
    # We use the same model configuration name to ensure tokenization compatibility
    _, val_dataset, _, _, _ = load_and_prepare_dataset(
        csv_path=args.dataset_path,
        model_name=args.model_dir,
        seed=args.seed
    )
    
    # Prepare DataLoader
    dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Generating predictions on the validation set...")
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Class names in correct order of IDs
    classes = [ID_MAPPING[i] for i in range(len(ID_MAPPING))]
    
    # Calculate global metrics
    acc = accuracy_score(all_labels, all_preds)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    
    print("\n" + "="*30 + " GLOBAL METRICS " + "="*30)
    print(f"Accuracy:        {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1-Score (Weighted): {f1_w:.4f}")
    print(f"F1-Score (Macro):    {f1_m:.4f}")
    print(f"Precision (Weighted):{precision_w:.4f}")
    print(f"Recall (Weighted):   {recall_w:.4f}")
    print("="*76)
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    print(report)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    os.makedirs(args.model_dir, exist_ok=True)
    plot_confusion_matrix(cm, classes, os.path.join(args.model_dir, "confusion_matrix.png"))
    
    # Save evaluation summary JSON
    summary = {
        "global_metrics": {
            "accuracy": acc,
            "f1_weighted": f1_w,
            "f1_macro": f1_m,
            "precision_weighted": precision_w,
            "recall_weighted": recall_w
        },
        "class_distribution": pd.Series(all_labels).value_counts().rename(index=ID_MAPPING).to_dict(),
        "confusion_matrix": cm.tolist()
    }
    
    summary_path = os.path.join(args.model_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved numerical evaluation summary to: {os.path.abspath(summary_path)}")
    print("="*60)

if __name__ == "__main__":
    main()
