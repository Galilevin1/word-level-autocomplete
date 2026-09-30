"""RNN / GRU language-model tests on tiny synthetic data."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from models.recurrent import RecurrentLM, pad
from src.evaluate import evaluate
from src.train import (TrainConfig, batch_tensors, mean_loss, set_seed,
                       token_loss, train_model)
from src.vocabulary import END_ID, PAD_ID, START_ID, UNK_ID

V = 12
CELLS = ["rnn", "gru"]


def sent(*w):
    return np.array([START_ID, *w, END_ID])


SENTS = [sent(5, 6, 7), sent(8, 9), sent(5, 6, 7, 8, 9, 10, 11), sent(UNK_ID)]


def make(cell, **kw):
    set_seed(0)
    return RecurrentLM(V, cell, emb_dim=8, hidden_dim=16, **kw).eval()


@pytest.mark.parametrize("cell", CELLS)
def test_shapes(cell):
    m = make(cell, num_layers=2)
    ids = torch.randint(3, V, (4, 7))
    logits, h = m(ids)
    assert logits.shape == (4, 7, V)
    assert h.shape == (2, 4, 16)
    assert torch.isinf(logits[..., [PAD_ID, START_ID]]).all()


def test_target_shifting():
    inputs, targets = batch_tensors(SENTS[:2])
    assert inputs.tolist() == [[START_ID, 5, 6, 7], [START_ID, 8, 9, PAD_ID]]
    assert targets.tolist() == [[5, 6, 7, END_ID], [8, 9, END_ID, PAD_ID]]


@pytest.mark.parametrize("cell", CELLS)
def test_padding_does_not_change_real_positions_or_loss(cell):
    m = make(cell)
    inputs, targets = batch_tensors(SENTS)
    with torch.no_grad():
        batch_logits, _ = m(inputs)
        batch_loss, n = token_loss(m, inputs, targets)
        single_total, single_n = 0.0, 0
        for i, s in enumerate(SENTS):
            x, y = batch_tensors([s])
            logits, _ = m(x)
            T = len(s) - 1
            assert torch.allclose(logits[0, :T], batch_logits[i, :T], atol=1e-6)
            loss, k = token_loss(m, x, y)
            single_total += float(loss)
            single_n += k
    assert n == single_n == sum(len(s) - 1 for s in SENTS)
    assert float(batch_loss) == pytest.approx(single_total, rel=1e-6)


@pytest.mark.parametrize("cell", CELLS)
def test_gradient_flow(cell):
    m = make(cell).train()
    inputs, targets = batch_tensors(SENTS)
    loss, n = token_loss(m, inputs, targets)
    (loss / n).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    # <PAD> embedding receives no gradient (padding_idx)
    assert (m.embedding.weight.grad[PAD_ID] == 0).all()


@pytest.mark.parametrize("cell", CELLS)
def test_prediction_normalization_and_prefix_only(cell):
    m = make(cell)
    rows = m.batch_log_probs(SENTS)
    for s, r in zip(SENTS, rows):
        assert r.shape == (len(s) - 1, V)
        assert np.allclose(np.logaddexp.reduce(r, axis=1), 0, atol=1e-5)
    s = SENTS[2]
    for t in range(len(s) - 1):
        p = m.predict(s[1:t + 1])
        assert p.sum() == pytest.approx(1.0)
        assert p[PAD_ID] == 0 and p[START_ID] == 0
        assert np.allclose(np.log(p[4:]), rows[2][t, 4:], atol=1e-5)
    # changing future tokens leaves earlier rows unchanged
    s2 = s.copy()
    s2[4:-1] = 5
    rows2 = m.batch_log_probs([s2])[0]
    assert np.allclose(rows[2][:4], rows2[:4], atol=1e-6)
    assert np.isfinite(evaluate(m, SENTS)["all"]["nll"])


def synthetic_corpus(n, rng):
    """Deterministic grammar: 5 -> 6 -> 7 ..., so a model can learn it."""
    out = []
    for _ in range(n):
        start = rng.integers(5, V)
        length = rng.integers(2, 6)
        out.append(sent(*[5 + (start - 5 + k) % (V - 5) for k in range(length)]))
    return out


@pytest.mark.parametrize("cell", CELLS)
def test_training_reduces_loss_and_saves_best(cell, tmp_path):
    rng = np.random.default_rng(0)
    train, val = synthetic_corpus(300, rng), synthetic_corpus(50, rng)
    m = make(cell, dropout=0.0)
    before = mean_loss(m, val)
    cfg = TrainConfig(epochs=4, batch_size=16, lr=1e-2, patience=10)
    select = lambda model: evaluate(model, val, ranks=False)["all"]["nll"]
    hist = train_model(m, train, val, cfg, tmp_path / "m.pt", select,
                       log=lambda *_: None)
    assert hist[-1]["val_loss"] < before
    best = min(hist, key=lambda r: r["val_nll"])
    loaded = RecurrentLM.load(tmp_path / "m.pt")
    assert loaded.metadata["epoch"] == best["epoch"]
    assert select(loaded) == pytest.approx(best["val_nll"], abs=1e-6)


LOAD_SCRIPT = """
import sys, numpy as np
sys.path.insert(0, {root!r})
from models.recurrent import RecurrentLM
m = RecurrentLM.load({path!r})
np.save({out!r}, m.predict([5, 6, 7]))
"""


@pytest.mark.parametrize("cell", CELLS)
def test_save_load_in_fresh_process(cell, tmp_path):
    m = make(cell)
    m.save(tmp_path / "m.pt", note="test")
    out = tmp_path / "p.npy"
    root = str(Path(__file__).resolve().parents[1])
    code = LOAD_SCRIPT.format(root=root, path=str(tmp_path / "m.pt"), out=str(out))
    subprocess.run([sys.executable, "-c", code], check=True)
    assert np.allclose(np.load(out), m.predict([5, 6, 7]), atol=1e-7)
    assert RecurrentLM.load(tmp_path / "m.pt").config == m.config
