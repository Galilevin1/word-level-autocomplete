"""
Encoded WikiText-2 splits shared by all models.
"""

import json
from pathlib import Path

from src.load_data import load_wikitext
from src.preprocess import preprocess_wikitext
from src.vocabulary import DEFAULT_MIN_COUNT, Vocabulary, coverage_stats

SPLITS = ("train", "validation", "test")
CACHE_PATH = Path("data/processed/wikitext2_sentences.json")


def load_tokenized_splits(cache_path=CACHE_PATH):
    """Tokenized sentences per split; cached after the first run."""

    cache_path = Path(cache_path)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    dataset = load_wikitext()
    splits = {k: preprocess_wikitext(dataset[k]) for k in SPLITS}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(splits, ensure_ascii=False),
                          encoding="utf-8")
    return splits


def load_corpus(min_count=DEFAULT_MIN_COUNT, vocab=None):
    """Return (vocab, encoded) with encoded[split] = list of id arrays.

    The vocabulary is built from the training split only, unless an
    existing (saved) vocabulary is passed in.
    """

    splits = load_tokenized_splits()
    if vocab is None:
        vocab = Vocabulary.build(splits["train"], min_count=min_count)
    encoded = {k: [vocab.encode_sentence(s) for s in splits[k]]
               for k in SPLITS}
    return vocab, encoded


def corpus_stats(vocab, encoded):
    """Dataset statistics recorded for the report."""

    return {
        "min_count": vocab.min_count,
        "vocab_size": len(vocab),
        **{k: coverage_stats(vocab, encoded[k]) for k in SPLITS},
    }
