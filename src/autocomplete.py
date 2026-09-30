"""
Autocomplete interface shared by the four trained models.

    predict_next("the game was", model="gru", k=5)

Policy
  * The context is the sentence currently being typed: text up to the
    last sentence-final punctuation (. ! ?) is ignored, because every
    model was trained on independent sentences. An empty context
    (empty text, or text ending in ". ") predicts the first word of a
    sentence.
  * Input is tokenized exactly like the corpus (lowercase, numbers ->
    <NUM>); words outside the training vocabulary become <UNK> and are
    reported in ``unknown_words``.
  * The full predictive distribution is validated (normalized, zero
    probability for <PAD>/<START>). Only when building the displayed
    suggestions are <PAD>, <START>, <UNK> and <END> excluded; the shown
    probabilities are NOT renormalized. P(<END>) is reported separately.
    <NUM> can be suggested (it stands for "a number").
  * Loaded models and vocabularies are cached per process, so repeated
    predictions do not reload checkpoints.
"""

from pathlib import Path

import numpy as np

from models.bigram import BayesianBigram
from models.hmm import HMM
from models.recurrent import RecurrentLM
from src.preprocess import split_prefix
from src.vocabulary import (END_ID, NOT_SUGGESTED, PAD_ID, START_ID, UNK_ID,
                            Vocabulary)

DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
SMOKE_CHECKPOINT_DIR = Path("checkpoints/smoke")
MODEL_FILES = {"bigram": "bigram.npz", "hmm": "hmm.npz",
               "rnn": "rnn.pt", "gru": "gru.pt"}
LOADERS = {"bigram": BayesianBigram.load, "hmm": HMM.load,
           "rnn": RecurrentLM.load, "gru": RecurrentLM.load}

_cache = {}


def _model_vocab_size(name, model):
    return model.config["vocab_size"] if name in ("rnn", "gru") else model.V


def load_model(name, checkpoint_dir=DEFAULT_CHECKPOINT_DIR):
    """(vocab, model) for a saved model; cached per process."""
    if name not in MODEL_FILES:
        raise ValueError(f"unknown model {name!r}; choose from {list(MODEL_FILES)}")
    checkpoint_dir = Path(checkpoint_dir)
    key = (name, checkpoint_dir.resolve())
    if key not in _cache:
        vocab = Vocabulary.load(checkpoint_dir / "vocab.json")
        model = LOADERS[name](checkpoint_dir / MODEL_FILES[name])
        size = _model_vocab_size(name, model)
        if size != len(vocab):
            raise ValueError(f"{name}: checkpoint vocabulary size {size} does "
                             f"not match vocab.json ({len(vocab)})")
        if model.vocab_sha256 != vocab.sha256():
            raise ValueError(f"{name}: checkpoint was trained with a different "
                             f"vocabulary (sha256 {model.vocab_sha256[:12] or 'missing'}"
                             f" != {vocab.sha256()[:12]})")
        _cache[key] = (vocab, model)
    return _cache[key]


def next_distribution(text, model="gru", checkpoint_dir=DEFAULT_CHECKPOINT_DIR):
    """(context tokens, context ids, full normalized distribution)."""
    vocab, m = load_model(model, checkpoint_dir)
    tokens = split_prefix(text)
    ids = vocab.encode(tokens)
    p = np.asarray(m.predict(ids), dtype=np.float64)
    if not (p.shape == (len(vocab),) and abs(p.sum() - 1.0) < 1e-6
            and p[PAD_ID] == 0 and p[START_ID] == 0 and (p >= 0).all()):
        raise AssertionError(f"{model}: invalid predictive distribution")
    return tokens, ids, p


def predict_next(text, model="gru", k=5, checkpoint_dir=DEFAULT_CHECKPOINT_DIR):
    """Top-k next-word suggestions for ``text``.

    Returns a dict with the ranked ``suggestions`` [(word, probability)],
    ``p_end_of_sentence``, ``p_unknown_word``, the ``context`` tokens used
    and any ``unknown_words``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    vocab, _ = load_model(model, checkpoint_dir)
    tokens, ids, p = next_distribution(text, model, checkpoint_dir)
    # rank only displayable words: descending probability, ties -> lower id
    # (more frequent word); at most k, and never a reserved token.
    displayable = np.setdiff1d(np.arange(len(p)), NOT_SUGGESTED)
    top = displayable[np.argsort(-p[displayable], kind="stable")[:k]]
    return {
        "model": model,
        "context": tokens,
        "unknown_words": [t for t, i in zip(tokens, ids) if i == UNK_ID],
        "suggestions": [(vocab.itos[i], float(p[i])) for i in top],
        "p_end_of_sentence": float(p[END_ID]),
        "p_unknown_word": float(p[UNK_ID]),
    }
