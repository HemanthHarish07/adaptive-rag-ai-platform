"""edu-intent-classifier package.

This package exposes the inference and dataset helpers for the intent classifier.
"""

from .dataset import ID_MAPPING, LABEL_MAPPING, load_and_prepare_dataset
from .inference import IntentClassifier

__all__ = [
    "ID_MAPPING",
    "LABEL_MAPPING",
    "IntentClassifier",
    "load_and_prepare_dataset",
]
