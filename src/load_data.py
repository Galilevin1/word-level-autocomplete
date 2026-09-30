"""
Dataset loading utilities for the Bayes autocomplete project.
"""

from pathlib import Path
from datasets import load_dataset


def load_text(path):
    """Load a local UTF-8 text file."""

    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    return text


def load_wikitext():
    """Load WikiText-2 with predefined train, validation, and test splits."""

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1"
    )

    return dataset