import os
from typing import Dict, Tuple

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Static label mapping based on requirements
LABEL_MAPPING = {
    "concept_explanation": 0,
    "coding_help": 1,
    "exam_preparation": 2,
    "debugging": 3,
    "definition": 4,
    "theory": 5,
}

ID_MAPPING = {value: key for key, value in LABEL_MAPPING.items()}

__all__ = ["LABEL_MAPPING", "ID_MAPPING", "load_and_prepare_dataset"]


def load_and_prepare_dataset(
    csv_path: str,
    model_name: str = "distilbert-base-uncased",
    test_size: float = 0.2,
    seed: int = 42,
    max_length: int = 128,
) -> Tuple[Dataset, Dataset, AutoTokenizer, Dict[str, int], Dict[int, str]]:
    """Load dataset from CSV, validate labels, split, and tokenize.

    Expected CSV columns:
      - query: the input text
      - label: one of the intent labels defined in LABEL_MAPPING
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["query", "label"]).reset_index(drop=True)

    df["label_id"] = df["label"].map(LABEL_MAPPING)
    unknown_labels = df.loc[df["label_id"].isna(), "label"].unique()
    if len(unknown_labels) > 0:
        raise ValueError(
            f"Found unknown labels in dataset: {list(unknown_labels)}. "
            f"Expected labels: {sorted(LABEL_MAPPING.keys())}"
        )

    df["label_id"] = df["label_id"].astype(int)

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label_id"],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = Dataset.from_pandas(
        train_df[["query", "label_id"]].rename(columns={"label_id": "label"})
    )
    val_dataset = Dataset.from_pandas(
        val_df[["query", "label_id"]].rename(columns={"label_id": "label"})
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["query"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    return tokenized_train, tokenized_val, tokenizer, LABEL_MAPPING, ID_MAPPING

