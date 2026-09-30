"""Autocomplete interface, vocabulary hashing and checkpoint safeguards."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from models.bigram import BayesianBigram
from models.hmm import HMM
from models.recurrent import RecurrentLM
from run_experiments import freeze_vocabulary
from src.autocomplete import MODEL_FILES, load_model, predict_next
from src.train import set_seed
from src.vocabulary import (NOT_SUGGESTED, PAD_ID, SPECIALS, START_ID,
                            Vocabulary)

ROOT = Path(__file__).resolve().parents[1]
TRAIN = [["the", "game", "was", "released", "in", "<NUM>"],
         ["the", "game", "was", "good"],
         ["it", "was", "released", "in", "japan"],
         ["the", "band", "was", "formed", "in", "<NUM>"]]
VOCAB = Vocabulary.build(TRAIN, min_count=1)
SWAPPED = Vocabulary(VOCAB.itos[6:7] + VOCAB.itos[5:6] + VOCAB.itos[7:])
DISPLAYABLE = {w for i, w in enumerate(VOCAB.itos) if i not in NOT_SUGGESTED}


def save_all(directory, vocab=VOCAB, sha=None, models=VOCAB):
    """Train tiny versions of all four models; save with a vocab hash."""
    directory.mkdir(parents=True, exist_ok=True)
    vocab.save(directory / "vocab.json")
    sha = models.sha256() if sha is None else sha
    enc = [models.encode_sentence(s) for s in TRAIN]
    V = len(models)
    BayesianBigram(V, 0.1).fit(enc).save(directory / "bigram.npz", vocab_sha256=sha)
    h = HMM(3, V, seed=0).init_params(enc)
    h.fit(enc, max_iter=3, log=lambda *_: None)
    h.save(directory / "hmm.npz", vocab_sha256=sha)
    for cell in ("rnn", "gru"):
        set_seed(0)
        RecurrentLM(V, cell, 8, 16).save(directory / f"{cell}.pt", vocab_sha256=sha)
    return directory


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    return save_all(tmp_path_factory.mktemp("ckpt"))


# ---------------------------------------------------------- vocabulary hash
def test_hash_deterministic_and_order_sensitive():
    assert Vocabulary.build(TRAIN, min_count=1).sha256() == VOCAB.sha256()
    assert len(SWAPPED) == len(VOCAB) and SWAPPED.itos[:5] == SPECIALS
    assert SWAPPED.sha256() != VOCAB.sha256()


@pytest.mark.parametrize("name", list(MODEL_FILES))
def test_matching_hash_loads_and_is_cached(ckpt, name):
    vocab, model = load_model(name, ckpt)
    assert model.vocab_sha256 == vocab.sha256() == VOCAB.sha256()
    assert load_model(name, ckpt)[1] is model          # reused, not reloaded


@pytest.mark.parametrize("name", list(MODEL_FILES))
def test_mismatched_token_order_rejected(tmp_path, name):
    d = save_all(tmp_path / "d", vocab=SWAPPED)        # models trained on VOCAB
    with pytest.raises(ValueError, match="different vocabulary"):
        load_model(name, d)


@pytest.mark.parametrize("name", list(MODEL_FILES))
def test_missing_hash_rejected(tmp_path, name):
    d = save_all(tmp_path / "d", sha="")
    with pytest.raises(ValueError, match="missing"):
        load_model(name, d)


@pytest.mark.parametrize("name", list(MODEL_FILES))
def test_size_mismatch_rejected(tmp_path, name):
    bigger = Vocabulary(VOCAB.itos[5:] + ["extra"])
    d = save_all(tmp_path / "d", vocab=VOCAB, models=bigger, sha=VOCAB.sha256())
    with pytest.raises(ValueError, match="size"):
        load_model(name, d)


def test_freeze_vocabulary(tmp_path):
    path = tmp_path / "vocab.json"
    assert freeze_vocabulary(VOCAB, path) is True
    before = path.stat().st_mtime_ns
    # unchanged vocabulary: reused, file not rewritten
    assert freeze_vocabulary(Vocabulary.build(TRAIN, min_count=1), path) is False
    assert path.stat().st_mtime_ns == before
    # a different mapping is refused and the frozen file is untouched
    with pytest.raises(SystemExit, match="frozen"):
        freeze_vocabulary(SWAPPED, path)
    assert Vocabulary.load(path).sha256() == VOCAB.sha256()


# ---------------------------------------------------------------- inference
@pytest.mark.parametrize("name", list(MODEL_FILES))
def test_predict_next_contract(ckpt, name):
    vocab, model = load_model(name, ckpt)
    r = predict_next("The game was", name, k=5, checkpoint_dir=ckpt)
    assert r["context"] == ["the", "game", "was"] and r["unknown_words"] == []
    words = [w for w, _ in r["suggestions"]]
    probs = [p for _, p in r["suggestions"]]
    assert len(words) == 5 and probs == sorted(probs, reverse=True)
    assert set(words) <= DISPLAYABLE
    # displayed probabilities are the model's own, not renormalized
    full = model.predict(vocab.encode(["the", "game", "was"]))
    for w, p in r["suggestions"]:
        assert p == pytest.approx(full[vocab.stoi[w]])
    assert full[PAD_ID] == 0 and full[START_ID] == 0
    assert full.sum() == pytest.approx(1.0)
    assert r["p_end_of_sentence"] == pytest.approx(full[vocab.stoi["<END>"]])


@pytest.mark.parametrize("name", list(MODEL_FILES))
def test_empty_boundary_and_unknown_prefixes(ckpt, name):
    empty = predict_next("", name, checkpoint_dir=ckpt)
    assert empty["context"] == []
    # text ending a sentence -> same as an empty (new-sentence) context
    for text in ("It was good.", "It was good. ", "Really?!"):
        assert predict_next(text, name, checkpoint_dir=ckpt)["suggestions"] == empty["suggestions"]
    # only the current sentence is used
    a = predict_next("Zebras run. the game", name, checkpoint_dir=ckpt)
    b = predict_next("the game", name, checkpoint_dir=ckpt)
    assert a["context"] == ["the", "game"] and a["suggestions"] == b["suggestions"]
    # unknown words map to <UNK> and are reported
    u = predict_next("the zyzzyva", name, checkpoint_dir=ckpt)
    assert u["unknown_words"] == ["zyzzyva"]
    u2 = predict_next("the qwertyuiop", name, checkpoint_dir=ckpt)
    assert u["suggestions"] == u2["suggestions"]       # any OOV word == <UNK>
    # numbers are mapped like the corpus
    assert predict_next("in 2011", name, checkpoint_dir=ckpt)["context"] == ["in", "<NUM>"]
    # k larger than the vocabulary: exactly the displayable words
    big = predict_next("the", name, k=10_000, checkpoint_dir=ckpt)
    assert len(big["suggestions"]) == len(VOCAB) - len(NOT_SUGGESTED)
    assert {w for w, _ in big["suggestions"]} == DISPLAYABLE
    assert len(predict_next("the", name, k=3, checkpoint_dir=ckpt)["suggestions"]) == 3
    with pytest.raises(ValueError):
        predict_next("the", name, k=0, checkpoint_dir=ckpt)


FRESH = """
import json, sys
sys.path.insert(0, {root!r})
from src.autocomplete import predict_next
print(json.dumps({{m: predict_next({text!r}, m, 3, {ckpt!r})
                  for m in ("bigram", "hmm", "rnn", "gru")}}))
"""


def test_all_models_in_fresh_process(ckpt):
    code = FRESH.format(root=str(ROOT), ckpt=str(ckpt), text="the game was")
    res = subprocess.run([sys.executable, "-c", code], check=True,
                         capture_output=True, text=True, cwd=ROOT)
    for m, r in json.loads(res.stdout).items():
        here = predict_next("the game was", m, 3, ckpt)
        assert [w for w, _ in r["suggestions"]] == [w for w, _ in here["suggestions"]]
        assert np.allclose([p for _, p in r["suggestions"]],
                           [p for _, p in here["suggestions"]])


def test_cli(ckpt):
    res = subprocess.run([sys.executable, "main.py", "--model", "all",
                          "--text", "the game was", "--top-k", "3",
                          "--checkpoint-dir", str(ckpt)],
                         check=True, capture_output=True, text=True, cwd=ROOT)
    for m in MODEL_FILES:
        assert f"[{m}] context: the game was" in res.stdout
    assert res.stdout.count("P(end of sentence)") == 4
