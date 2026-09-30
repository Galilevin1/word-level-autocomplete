"""HMM / Baum-Welch correctness tests on small synthetic problems."""

import itertools

import numpy as np
import pytest

from models.hmm import HMM, length_batches
from src.evaluate import evaluate
from src.vocabulary import END_ID, PAD_ID, START_ID, UNK_ID

V = 9  # ids 0-4 reserved, 5-8 words


def random_hmm(K=3, seed=1):
    rng = np.random.default_rng(seed)
    h = HMM(K, V, seed=seed)
    h.pi = rng.dirichlet(np.ones(K))
    h.A = rng.dirichlet(np.ones(K), size=K)
    B = rng.dirichlet(np.ones(V), size=K)
    B[:, [PAD_ID, START_ID]] = 0
    h.B = B / B.sum(axis=1, keepdims=True)
    return h


def paths(h, obs):
    """Yield (state path, joint probability p(path, obs))."""
    for path in itertools.product(range(h.K), repeat=len(obs)):
        p = h.pi[path[0]] * h.B[path[0], obs[0]]
        for t in range(1, len(obs)):
            p *= h.A[path[t - 1], path[t]] * h.B[path[t], obs[t]]
        yield path, p


def brute_force_loglik(h, obs):
    return np.log(sum(p for _, p in paths(h, obs)))


def sent(*obs):
    return np.array([START_ID, *obs])


def sample(h, n, rng, max_len=12):
    """Sentences from h: stop when <END> is emitted."""
    out = []
    for _ in range(n):
        z = rng.choice(h.K, p=h.pi)
        o = []
        while len(o) < max_len:
            w = rng.choice(V, p=h.B[z])
            o.append(w)
            if w == END_ID:
                break
            z = rng.choice(h.K, p=h.A[z])
        if o[-1] != END_ID:
            o.append(END_ID)
        out.append(sent(*o))
    return out


QUIET = {"log": lambda *_: None}


# ------------------------------------------------------------ forward pass
def test_forward_matches_brute_force():
    h = random_hmm()
    for obs in ([5, 6, 7, 3], [8], [5, 5, 1, 6, 3]):
        ll, n = h.log_likelihood([sent(*obs)])
        assert n == len(obs)
        assert ll == pytest.approx(brute_force_loglik(h, obs))


def test_padded_batch_equals_individual():
    h = random_hmm()
    sents = [sent(5, 6, 3), sent(7, 3), sent(5, 6, 7, 8, 5, 3), sent(3)]
    joint, _ = h.log_likelihood(sents, max_tokens=10_000)   # one padded batch
    separate = sum(h.log_likelihood([s])[0] for s in sents)
    assert joint == pytest.approx(separate)


def test_length_batches_cover_every_sentence_once():
    rng = np.random.default_rng(0)
    sents = [sent(*rng.integers(5, 9, n), 3) for n in rng.integers(0, 30, 200)]
    seen = []
    for idx, obs, mask in length_batches(sents, max_tokens=64):
        seen.extend(idx)
        for r, i in enumerate(idx):
            assert np.array_equal(obs[r, mask[r]], sents[i][1:])
    assert sorted(seen) == list(range(200))


def test_long_sequence_is_stable():
    h = random_hmm()
    rng = np.random.default_rng(0)
    ll, n = h.log_likelihood([sent(*rng.integers(5, 9, 5000), 3)])
    assert np.isfinite(ll) and n == 5001


# ------------------------------------------------- forward-backward / E-step
def test_posteriors_normalized_and_match_brute_force():
    h = random_hmm()
    obs = [5, 7, 7, 6, 3]
    gamma = h.state_posteriors(obs)
    assert np.allclose(gamma.sum(axis=1), 1.0)
    ref = np.zeros((len(obs), h.K))
    for path, p in paths(h, obs):
        ref[np.arange(len(obs)), path] += p
    assert np.allclose(gamma, ref / ref.sum(axis=1, keepdims=True))


def test_expected_counts_match_brute_force():
    """One EM step (no smoothing) equals re-estimation from enumerated paths.

    This checks the expected initial-state, transition (xi) and emission
    (gamma) counts, including a padded multi-sequence batch.
    """
    h = random_hmm(K=2, seed=3)
    h.eps_pi = h.eps_A = h.eps_B = 0.0
    seqs = [[5, 7, 6, 3], [8, 3], [6, 6, UNK_ID, 5, 3]]
    K = h.K
    pi_c, A_c, B_c = np.zeros(K), np.zeros((K, K)), np.zeros((K, V))
    for obs in seqs:
        Z = sum(p for _, p in paths(h, obs))
        for path, p in paths(h, obs):
            w = p / Z
            pi_c[path[0]] += w
            for t in range(len(obs)):
                B_c[path[t], obs[t]] += w
                if t:
                    A_c[path[t - 1], path[t]] += w
    h.em_iteration([sent(*o) for o in seqs])
    assert np.allclose(h.pi, pi_c / pi_c.sum())
    assert np.allclose(h.A, A_c / A_c.sum(1, keepdims=True))
    assert np.allclose(h.B, B_c / B_c.sum(1, keepdims=True))


# ---------------------------------------------------------- Baum-Welch runs
def test_baum_welch_monotone_and_finite():
    rng = np.random.default_rng(0)
    data = sample(random_hmm(K=3, seed=7), 300, rng)
    h = HMM(3, V, seed=0).init_params(data)
    h.fit(data, max_iter=25, tol=0, **QUIET)
    obj = [r["log_posterior_before"] for r in h.history]
    ll = [r["train_loglik_before"] for r in h.history]
    assert np.isfinite(obj).all() and np.isfinite(ll).all()
    assert all(b >= a - 1e-9 * abs(a) for a, b in zip(obj, obj[1:]))
    assert ll[-1] > ll[0]
    assert not any("warning" in r for r in h.history)
    assert max(r["gamma_sum_max_error"] for r in h.history) < 1e-10
    for p in (h.pi, *h.A, *h.B):
        assert p.sum() == pytest.approx(1.0) and (p >= 0).all()
    assert (h.B[:, [PAD_ID, START_ID]] == 0).all()


def test_selection_uses_common_evaluator(tmp_path):
    rng = np.random.default_rng(1)
    true = random_hmm(K=2, seed=5)
    train, val = sample(true, 200, rng), sample(true, 50, rng)
    select = lambda m: evaluate(m, val)["all"]["nll"]
    h = HMM(2, V, seed=0).init_params(train)
    h.fit(train, select_fn=select, eval_every=3, max_iter=10, tol=0, **QUIET)
    evaluated = {r["em_step"]: r["val_nll_after"]
                 for r in h.history if "val_nll_after" in r}
    assert sorted(evaluated) == [3, 6, 9, 10]
    assert h.best_iteration == min(evaluated, key=evaluated.get)
    assert select(h) == pytest.approx(min(evaluated.values()))

    h.save(tmp_path / "hmm.npz")
    h2 = HMM.load(tmp_path / "hmm.npz")
    for a, b in ((h.pi, h2.pi), (h.A, h2.A), (h.B, h2.B)):
        assert np.array_equal(a, b)
    assert np.array_equal(h2.predict([5, 6]), h.predict([5, 6]))
    assert select(h2) == select(h)
    assert h2.history == h.history and h2.best_iteration == h.best_iteration


# --------------------------------------------------------------- prediction
def test_predictive_is_ratio_of_likelihoods():
    h = random_hmm()
    prefix = [5, 8, 6]
    p = h.predict(prefix)
    assert p.sum() == pytest.approx(1.0)
    base = brute_force_loglik(h, prefix)
    for v in (5, 6, 3, 1):
        assert p[v] == pytest.approx(np.exp(brute_force_loglik(h, prefix + [v]) - base))
    assert np.allclose(h.predict([]), h.pi @ h.B)
    assert p[PAD_ID] == 0 and p[START_ID] == 0


def test_batch_log_probs_consistent_and_no_leakage():
    h = random_hmm()
    s = sent(5, 8, 6, 7, 3)
    rows = h.batch_log_probs([s, sent(6, 3)])[0]
    assert rows.shape == (len(s) - 1, V)
    for t in range(len(s) - 1):
        assert np.allclose(np.exp(rows[t]), h.predict(s[1:t + 1]))
    s2 = s.copy()
    s2[3:] = [7, 7, 3]                       # change the future only
    rows2 = h.batch_log_probs([s2])[0]
    # (a different batch shape can change BLAS rounding in the last ulp)
    assert np.allclose(rows[:3], rows2[:3], rtol=0, atol=1e-12)
    assert not np.allclose(rows[3], rows2[3])
    assert np.isfinite(evaluate(h, [s, s2])["all"]["nll"])
