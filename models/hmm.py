"""
Model B: categorical-emission HMM trained with Baum-Welch (EM).

Parameters: initial distribution pi (K), transitions A (K x K) and
emissions B (K x V). A sentence <START> w_1 ... w_n <END> is modelled as
the observation sequence o = (w_1, ..., w_n, <END>): <START> is not
emitted, its role is played by pi. <PAD> and <START> have zero emission
probability, so every predictive distribution lives on the same output
space as the other models.

Forward-backward uses Rabiner scaling (alpha_t normalized by c_t), so
log p(o) = sum_t log c_t and no underflow occurs. Sentences are
independent sequences; they are processed in length-sorted, padded
batches. At padded steps the emission likelihood is set to 1, which makes
c_t = 1 exactly, and padded steps are masked out of all expected counts.

The M-step is a MAP update under symmetric Dirichlet(1 + eps) priors,
i.e. expected counts + eps pseudo-counts; EM then monotonically
increases the log posterior log p(D | theta) + log p(theta).

Next-word prediction filters the prefix, propagates one step and
marginalizes the emission:
    P(o_{t+1} = v | o_{1:t}) = sum_{i,j} P(z_t = i | o_{1:t}) A_ij B_jv,
and P(o_1 = v) = sum_j pi_j B_jv for an empty prefix.
"""

import json
import time

import numpy as np

from src.vocabulary import NEVER_PREDICTED


def length_batches(sentences, max_tokens=65536):
    """Yield (indices, obs (S, T), mask (S, T)) for length-sorted batches.

    obs holds the observations s[1:] of each sentence, right-padded
    with 0; mask marks the real positions.
    """
    lengths = np.array([len(s) - 1 for s in sentences])
    order = np.argsort(lengths, kind="stable")
    start = 0
    while start < len(order):
        stop = start + 1
        # Grow while the padded batch (S x T_max) fits the token budget.
        while stop < len(order) and (stop - start + 1) * lengths[order[stop]] <= max_tokens:
            stop += 1
        idx = order[start:stop]
        T = lengths[idx].max()
        obs = np.zeros((len(idx), T), dtype=np.int64)
        mask = np.zeros((len(idx), T), dtype=bool)
        for r, i in enumerate(idx):
            obs[r, :lengths[i]] = sentences[i][1:]
            mask[r, :lengths[i]] = True
        yield idx, obs, mask
        start = stop


class HMM:

    def __init__(self, n_states, vocab_size, eps_pi=1e-2, eps_A=1e-2,
                 eps_B=1e-3, seed=0):
        self.K, self.V = n_states, vocab_size
        self.eps_pi, self.eps_A, self.eps_B = eps_pi, eps_A, eps_B
        self.seed = seed
        self.support = np.ones(vocab_size, dtype=bool)
        self.support[list(NEVER_PREDICTED)] = False
        self.history = []

    # ------------------------------------------------------- initialization
    def init_params(self, sentences):
        """Random pi, A; B = training unigram with random perturbation."""
        rng = np.random.default_rng(self.seed)
        self.pi = rng.dirichlet(np.ones(self.K))
        self.A = rng.dirichlet(np.ones(self.K), size=self.K)
        unigram = np.bincount(np.concatenate([s[1:] for s in sentences]),
                              minlength=self.V) + 1.0
        B = unigram[None, :] * rng.uniform(0.5, 1.5, size=(self.K, self.V))
        B[:, ~self.support] = 0.0
        self.B = B / B.sum(axis=1, keepdims=True)
        return self

    # ----------------------------------------------------- forward-backward
    def _emissions(self, obs, mask):
        """(T, S, K) emission likelihoods, 1 at padded positions."""
        em = np.ascontiguousarray(self.B.T)[obs.T]
        em[~mask.T] = 1.0
        return em

    def _forward(self, em):
        """Scaled forward pass. Returns alpha (T, S, K) and c (T, S)."""
        T, S, _ = em.shape
        alpha = np.empty_like(em)
        c = np.empty((T, S))
        a = self.pi[None, :] * em[0]
        for t in range(T):
            if t > 0:
                a = (a @ self.A) * em[t]
            c[t] = a.sum(axis=1)
            a = a / c[t][:, None]
            alpha[t] = a
        return alpha, c

    def _backward(self, em, c):
        """Scaled backward pass, consistent with _forward's c_t."""
        T = em.shape[0]
        beta = np.empty_like(em)
        beta[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (em[t + 1] * beta[t + 1] / c[t + 1][:, None]) @ self.A.T
        return beta

    def _e_step_batch(self, obs, mask, stats):
        """Accumulate expected counts of one batch into ``stats``."""
        em = self._emissions(obs, mask)
        alpha, c = self._forward(em)
        beta = self._backward(em, c)
        m = mask.T

        loglik = np.log(c[m]).sum()
        gamma = alpha * beta                       # P(z_t | o_{1:T}), (T,S,K)
        stats["pi"] += gamma[0].sum(axis=0)

        # xi_t(i,j) summed over t and sequences, only where t+1 is real:
        # alpha_t(i) A_ij em_{t+1}(j) beta_{t+1}(j) / c_{t+1}
        G = em[1:] * beta[1:] / c[1:, :, None] * m[1:, :, None]
        K = self.K
        stats["A"] += self.A * (alpha[:-1].reshape(-1, K).T @ G.reshape(-1, K))

        np.add.at(stats["B"], obs.T[m], gamma[m])  # (V, K) expected emissions
        stats["gamma_sum_err"] = max(stats.get("gamma_sum_err", 0.0),
                                     float(np.abs(gamma[m].sum(axis=1) - 1).max()))
        return loglik

    def _log_prior(self):
        """log p(theta) up to a constant: eps * sum log theta."""
        with np.errstate(divide="ignore"):
            return float(self.eps_pi * np.log(self.pi).sum()
                         + self.eps_A * np.log(self.A).sum()
                         + self.eps_B * np.log(self.B[:, self.support]).sum())

    def em_iteration(self, sentences, max_tokens=65536):
        """One Baum-Welch iteration. Returns log p(D | theta_old)."""
        stats = {"pi": np.zeros(self.K), "A": np.zeros((self.K, self.K)),
                 "B": np.zeros((self.V, self.K))}
        loglik = 0.0
        for _, obs, mask in length_batches(sentences, max_tokens):
            loglik += self._e_step_batch(obs, mask, stats)
        if not np.isfinite(loglik):
            raise FloatingPointError("non-finite log-likelihood in E-step")

        # M-step (MAP with Dirichlet pseudo-counts)
        pi = stats["pi"] + self.eps_pi
        A = stats["A"] + self.eps_A
        B = stats["B"].T + self.eps_B
        B[:, ~self.support] = 0.0
        self.pi = pi / pi.sum()
        self.A = A / A.sum(axis=1, keepdims=True)
        self.B = B / B.sum(axis=1, keepdims=True)
        self._last_gamma_err = stats["gamma_sum_err"]
        return loglik

    def log_likelihood(self, sentences, max_tokens=65536):
        """(total log-likelihood, number of observations)."""
        total, n = 0.0, 0
        for _, obs, mask in length_batches(sentences, max_tokens):
            _, c = self._forward(self._emissions(obs, mask))
            total += np.log(c[mask.T]).sum()
            n += int(mask.sum())
        return total, n

    # -------------------------------------------------------------- training
    def fit(self, train, select_fn=None, eval_every=1, max_iter=50, tol=1e-4,
            max_tokens=65536, log=print):
        """Baum-Welch until the relative objective gain < tol or max_iter.

        Monitoring: each history row t records the full-sequence training
        log-likelihood and log-posterior of theta_{t-1} (computed in E-step
        t), which EM must not decrease.

        Selection: select_fn(model) -> validation NLL under the common
        evaluation policy (src/evaluate.py). It is applied to theta_t after
        every ``eval_every`` M-steps and at the final step; the
        parameters with the lowest value are kept.
        """
        n_train = sum(len(s) - 1 for s in train)
        best = None
        prev_obj = None
        for it in range(1, max_iter + 1):
            t0 = time.time()
            log_prior = self._log_prior()
            ll = self.em_iteration(train, max_tokens)   # E at theta_{it-1}
            obj = ll + log_prior
            row = {"em_step": it,
                   "train_loglik_before": ll,
                   "train_nll_per_token_before": -ll / n_train,
                   "log_posterior_before": obj,
                   "gamma_sum_max_error": self._last_gamma_err}
            if prev_obj is not None and obj < prev_obj - 1e-9 * abs(prev_obj):
                row["warning"] = "log posterior decreased"
                log(f"WARNING: EM objective decreased {prev_obj} -> {obj}")
            converged = prev_obj is not None and (obj - prev_obj) < tol * abs(prev_obj)
            prev_obj = obj
            row["em_seconds"] = time.time() - t0

            msg = (f"  EM step {it:3d}: train NLL/token (theta_{it - 1}) "
                   f"{row['train_nll_per_token_before']:.4f}  ({row['em_seconds']:.1f}s)")
            if select_fn is not None and (it % eval_every == 0 or converged
                                          or it == max_iter):
                t1 = time.time()
                row["val_nll_after"] = float(select_fn(self))  # theta_it
                msg += (f"  val NLL (theta_{it}) {row['val_nll_after']:.4f}"
                        f" ({time.time() - t1:.1f}s)")
                if best is None or row["val_nll_after"] < best[0]:
                    best = (row["val_nll_after"], it,
                            (self.pi.copy(), self.A.copy(), self.B.copy()))
            self.history.append(row)
            log(msg)
            if converged:
                break

        if best is not None:
            self.pi, self.A, self.B = best[2]
            self.best_iteration = best[1]
        else:
            self.best_iteration = self.history[-1]["em_step"]
        return self

    # ------------------------------------------------------------ prediction
    def filter(self, obs):
        """Filtered state distributions P(z_t | o_{1:t}) for t = 1..len."""
        if len(obs) == 0:
            return np.empty((0, self.K))
        em = self._emissions(np.asarray(obs)[None, :], np.ones((1, len(obs)), bool))
        alpha, _ = self._forward(em)
        return alpha[:, 0, :]

    def state_posteriors(self, obs):
        """Smoothed P(z_t | o_{1:T}) for one sequence, shape (T, K)."""
        em = self._emissions(np.asarray(obs)[None, :], np.ones((1, len(obs)), bool))
        alpha, c = self._forward(em)
        return (alpha * self._backward(em, c))[:, 0, :]

    def predict(self, prefix_ids):
        """P(next token | prefix); prefix_ids excludes <START>."""
        if len(prefix_ids) == 0:
            state = self.pi
        else:
            state = self.filter(prefix_ids)[-1] @ self.A
        return state @ self.B

    def batch_log_probs(self, sentences, max_tokens=4096):
        """Row t predicts s[t+1] from s[1..t] (row 0 from pi alone).

        alpha_t only depends on o_1..o_t, so no target enters its own
        conditioning context. max_tokens bounds the (T, S, V) output.
        """
        out = []
        for idx, obs, mask in length_batches(sentences, max_tokens):
            S = len(idx)
            alpha, _ = self._forward(self._emissions(obs, mask))
            # state predictions: pi for the first token, alpha_t A after
            pred = np.concatenate(
                [np.broadcast_to(self.pi, (1, S, self.K)), alpha[:-1] @ self.A])
            # real positions only, ordered by sentence then time: (n, K)
            pred = pred.transpose(1, 0, 2)[mask]
            with np.errstate(divide="ignore"):
                logp = np.log(pred @ self.B)
            ends = np.cumsum(mask.sum(axis=1))
            for i, rows in zip(idx, np.split(logp, ends[:-1])):
                out.append((i, rows))
        out.sort(key=lambda x: x[0])
        return [rows for _, rows in out]

    # ----------------------------------------------------------- persistence
    def save(self, path, vocab_sha256=None):
        np.savez_compressed(
            path, pi=self.pi, A=self.A, B=self.B.astype(np.float64),
            config=json.dumps({"K": self.K, "V": self.V, "eps_pi": self.eps_pi,
                               "eps_A": self.eps_A, "eps_B": self.eps_B,
                               "seed": self.seed,
                               "best_iteration": getattr(self, "best_iteration", None),
                               "vocab_sha256": vocab_sha256 or ""}),
            history=json.dumps(self.history))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        cfg = json.loads(str(d["config"]))
        model = cls(cfg["K"], cfg["V"], cfg["eps_pi"], cfg["eps_A"],
                    cfg["eps_B"], cfg["seed"])
        model.pi, model.A, model.B = d["pi"], d["A"], d["B"]
        model.best_iteration = cfg["best_iteration"]
        model.vocab_sha256 = cfg.get("vocab_sha256", "")
        model.history = json.loads(str(d["history"]))
        return model
