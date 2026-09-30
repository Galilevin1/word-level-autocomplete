"""Bayesian bigram and common-evaluator tests on a toy corpus."""

import math

import numpy as np
import pytest

from models.bigram import BayesianBigram
from src.evaluate import evaluate
from src.vocabulary import (END_ID, NUM_ID, PAD_ID, START_ID, UNK_ID,
                            Vocabulary)

TRAIN = [["the", "cat", "sat"], ["the", "dog", "sat"], ["a", "cat", "ran"]]
VOCAB = Vocabulary.build(TRAIN, min_count=1)
ENC = [VOCAB.encode_sentence(s) for s in TRAIN]
V = len(VOCAB)
O = V - 2  # output space excludes <PAD> and <START>


def ids(*words):
    return [VOCAB.stoi[w] for w in words]


def test_posterior_predictive_matches_formula():
    alpha = 0.5
    m = BayesianBigram(V, alpha).fit(ENC)
    the, cat, dog, sat = ids("the", "cat", "dog", "sat")
    p = m.probs([the])[0]
    # C(the, cat) = 1, C(the, dog) = 1, n_the = 2
    assert p[cat] == pytest.approx((1 + alpha) / (2 + alpha * O))
    assert p[sat] == pytest.approx(alpha / (2 + alpha * O))  # unseen > 0
    p_start = m.probs([START_ID])[0]
    assert p_start[the] == pytest.approx((2 + alpha) / (3 + alpha * O))
    assert m.count(the, cat) == 1 and m.count(the, sat) == 0


def test_distributions_normalized_and_masked():
    m = BayesianBigram(V, 0.01).fit(ENC)
    P = m.probs(np.arange(V))  # every context, including unseen ones
    assert np.allclose(P.sum(axis=1), 1.0)
    assert (P[:, PAD_ID] == 0).all() and (P[:, START_ID] == 0).all()
    support = np.delete(np.arange(V), [PAD_ID, START_ID])
    assert (P[:, support] > 0).all()
    # A context never observed (e.g. <NUM>, <END>) gives the uniform prior
    assert np.allclose(P[NUM_ID, support], 1 / O)


def test_no_sentence_boundary_crossing():
    m = BayesianBigram(V, 1.0).fit(ENC)
    assert m.row_totals[END_ID] == 0          # <END> is never a context
    assert m.count(ids("sat")[0], ids("the")[0]) == 0  # not across sentences
    assert m.row_totals[START_ID] == len(TRAIN)


def test_invalid_alpha():
    with pytest.raises(ValueError):
        BayesianBigram(V, 0.0)


def test_log_evidence_brute_force():
    alpha = 0.3
    m = BayesianBigram(V, alpha).fit(ENC)
    expected = 0.0
    for u in range(V):
        n_u = m.row_totals[u]
        if n_u == 0:
            continue
        expected += math.lgamma(alpha * O) - math.lgamma(n_u + alpha * O)
        for w in range(V):
            if w not in (PAD_ID, START_ID):
                expected += math.lgamma(m.count(u, w) + alpha) - math.lgamma(alpha)
    assert m.log_evidence() == pytest.approx(expected)


def test_save_load_and_with_alpha(tmp_path):
    m = BayesianBigram(V, 0.2).fit(ENC)
    m.save(tmp_path / "bigram.npz")
    m2 = BayesianBigram.load(tmp_path / "bigram.npz")
    assert m2.alpha == 0.2
    assert np.allclose(m.probs(np.arange(V)), m2.probs(np.arange(V)))
    m3 = m.with_alpha(1.0)
    assert m3.alpha == 1.0 and np.allclose(m3.probs([START_ID]).sum(), 1)


def test_predict_prefix_uses_last_token():
    m = BayesianBigram(V, 0.1).fit(ENC)
    assert np.allclose(m.predict([]), m.probs([START_ID])[0])
    assert np.allclose(m.predict(ids("a", "cat")), m.probs(ids("cat"))[0])
    assert np.allclose(m.predict([UNK_ID]), m.probs([UNK_ID])[0])


def test_no_future_leakage():
    """Row t must not change when tokens after position t change."""
    m = BayesianBigram(V, 0.1).fit(ENC)
    s = VOCAB.encode_sentence(["the", "cat", "sat"])
    s2 = s.copy()
    s2[3] = ids("ran")[0]  # change the target of row 2 and later context
    a, b = m.batch_log_probs([s])[0], m.batch_log_probs([s2])[0]
    assert np.array_equal(a[:3], b[:3])


class FixedModel:
    """Returns the same hand-made distribution at every position."""

    def __init__(self, p):
        with np.errstate(divide="ignore"):
            self.lp = np.log(p)

    def batch_log_probs(self, sentences):
        return [np.tile(self.lp, (len(s) - 1, 1)) for s in sentences]


def test_evaluator_metrics_by_hand():
    # V = 7: ids 0 PAD, 1 UNK, 2 START, 3 END, 4 NUM, 5 x, 6 y
    p = np.array([0, 0.4, 0, 0.1, 0.2, 0.2, 0.1])
    sents = [np.array([START_ID, 5, UNK_ID, 6, END_ID])]
    m = evaluate(FixedModel(p), sents)
    # targets 5, UNK, 6, END -> UNK excluded
    assert m["unk_targets_excluded"] == 1 and m["all"]["targets"] == 3
    assert m["all"]["nll"] == pytest.approx(-np.mean(np.log([0.2, 0.1, 0.1])))
    # "all" candidates exclude <UNK>: NUM(0.2) x(0.2, tie -> lower id
    # first) END(0.1) y(0.1): ranks x=1, END=2, y=3
    assert m["all"]["top1"] == 0
    assert m["all"]["top5"] == 1
    assert m["words"]["targets"] == 2
    assert m["words"]["perplexity"] == pytest.approx(1 / math.sqrt(0.2 * 0.1))


def test_words_ranking_excludes_end():
    # <END> is the most probable token; x is the best displayable word.
    p = np.array([0, 0.1, 0, 0.4, 0.1, 0.3, 0.1])
    m = evaluate(FixedModel(p), [np.array([START_ID, 5, END_ID])])
    # all: targets x (rank 1, behind <END>) and <END> (rank 0)
    assert m["all"]["top1"] == pytest.approx(0.5)
    # words: target x among candidates {NUM, x, y} -> rank 0
    assert m["words"]["targets"] == 1 and m["words"]["top1"] == 1
    # NLL of x still uses the full, unrenormalized distribution
    assert m["words"]["nll"] == pytest.approx(-math.log(0.3))


def naive_metrics(rows, sentences, k):
    """Per-target reference: sort the candidate list explicitly."""
    res = {"all": [], "words": []}
    for lp_s, s in zip(rows, sentences):
        for lp, target in zip(lp_s, s[1:]):
            if target == UNK_ID:
                continue
            for g in ("all", "words"):
                if g == "words" and target == END_ID:
                    continue
                excluded = {PAD_ID, START_ID, UNK_ID}
                if g == "words":
                    excluded.add(END_ID)
                cand = sorted((w for w in range(len(lp)) if w not in excluded),
                              key=lambda w: (-lp[w], w))
                res[g].append((-lp[target], cand.index(target) < k))
    return {g: (np.mean([a for a, _ in r]), np.mean([b for _, b in r]))
            for g, r in res.items()}


class ReplayModel:
    """Serves precomputed log-probability rows in sentence order."""

    def __init__(self, rows):
        self.rows, self.i = rows, 0

    def batch_log_probs(self, batch):
        out = self.rows[self.i:self.i + len(batch)]
        self.i += len(batch)
        return out


def random_rows(rng, sentences, Vr):
    rows = []
    for s in sentences:
        logits = rng.normal(size=(len(s) - 1, Vr))
        logits[:, 8] = logits[:, 7]                  # exact ties
        logits[:, [PAD_ID, START_ID]] = -np.inf
        rows.append(logits - np.logaddexp.reduce(logits, axis=1, keepdims=True))
    return rows


def test_evaluator_matches_naive_reference():
    rng = np.random.default_rng(0)
    Vr = 12
    inner = [UNK_ID, END_ID] + list(range(NUM_ID, Vr))  # never PAD/START
    sents = [np.array([START_ID, *rng.choice(inner, n), END_ID])
             for n in rng.integers(1, 9, 40)]
    rows = random_rows(rng, sents, Vr)
    m = evaluate(ReplayModel(rows), sents, batch_size=7)
    for k in (1, 5):
        ref = naive_metrics(rows, sents, k)
        for g in ("all", "words"):
            assert m[g]["nll"] == pytest.approx(ref[g][0])
            assert m[g][f"top{k}"] == pytest.approx(ref[g][1])


def test_nll_only_mode_identical():
    m = BayesianBigram(V, 0.1).fit(ENC)
    full, fast = evaluate(m, ENC), evaluate(m, ENC, ranks=False)
    for g in ("all", "words"):
        assert fast[g]["nll"] == full[g]["nll"]
        assert "top1" not in fast[g]


def test_bigram_log_rows_match_probs():
    m = BayesianBigram(V, 0.05).fit(ENC)
    with np.errstate(divide="ignore"):
        expected = np.log(m.probs(ENC[0][:-1]))
    assert np.allclose(m.batch_log_probs([ENC[0]])[0], expected)


def test_evaluator_rejects_unnormalized():
    p = np.array([0, 0.4, 0, 0.1, 0.2, 0.2, 0.2])
    with pytest.raises(AssertionError):
        evaluate(FixedModel(p), [np.array([START_ID, 5, END_ID])])


def test_bigram_through_evaluator():
    m = BayesianBigram(V, 0.1).fit(ENC)
    r = evaluate(m, ENC)
    assert r["all"]["targets"] == sum(len(s) - 1 for s in ENC)
    assert 0 <= r["all"]["top1"] <= r["all"]["top5"] <= 1
