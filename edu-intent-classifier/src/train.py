import os
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
import torch

try:
    from .dataset import load_and_prepare_dataset
except ImportError:
    from dataset import load_and_prepare_dataset


def compute_metrics(eval_pred):
    """
    Computes key classification metrics: accuracy, precision, recall, and F1.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate weighted and macro metrics
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1_weighted': f1_w,
        'f1_macro': f1_m,
        'precision_weighted': precision_w,
        'recall_weighted': recall_w
    }

class TrainingProgressCallback(TrainerCallback):
    """
    Simple callback to log training progress cleanly.
    """
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n--- Epoch {state.epoch:.1f}/{state.num_train_epochs} Finished ---")

def main():
    parser = argparse.ArgumentParser(description="Train Educational Intent Classifier using HuggingFace Trainer")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Base transformer model name")
    parser.add_argument("--dataset_path", type=str, default="data/raw/intent_dataset.csv", help="Path to raw CSV dataset")
    parser.add_argument("--output_dir", type=str, default="models/intent_classifier", help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=64, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("="*60)
    print("EDUCATIONAL INTENT CLASSIFIER - TRAINING STAGE")
    print("="*60)
    
    # Check device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    
    # Load and prepare datasets
    train_dataset, val_dataset, tokenizer, label_to_id, id_to_label = load_and_prepare_dataset(
        csv_path=args.dataset_path,
        model_name=args.model_name,
        seed=args.seed,
        max_length=args.max_length
    )
    
    # Load base model for sequence classification
    print(f"\nLoading base model: {args.model_name} with {len(label_to_id)} classes...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id
    )
    
    # Move model to device
    model.to(device)
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        report_to="none" # Disable integrations like wandb for local predictability
    )
    
    # Initialize HuggingFace Trainer
    print("\nInitializing HuggingFace Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[TrainingProgressCallback()]
    )
    
    # Run training
    print("\nStarting fine-tuning...")
    train_result = trainer.train()
    
    # Save the best model and tokenizer
    print(f"\nSaving fine-tuned model and tokenizer to: {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Log training stats
    print("\nTraining Metrics:")
    metrics = train_result.metrics
    for k, v in metrics.items():
        print(f"  {k}: {v}")
        
    # Evaluate final model
    print("\nRunning final validation evaluation...")
    eval_metrics = trainer.evaluate()
    print("\nValidation Metrics:")
    for k, v in eval_metrics.items():
        print(f"  {k}: {v}")
        
    print("\nFine-tuning process completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
