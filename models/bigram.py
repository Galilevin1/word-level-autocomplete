"""
Model A: Bayesian bigram with a symmetric Dirichlet prior.

For each context word u the successor distribution theta_u has prior
Dir(alpha, ..., alpha) over the output space O (every token except <PAD>
and <START>, which can never follow). Given transition counts C(u, .),
the posterior is Dir(alpha + C(u, .)) and the posterior predictive is

    P(w | u, D) = (C(u, w) + alpha) / (n_u + alpha |O|),   n_u = sum_v C(u, v).

Counts are stored sparsely (CSR over observed pairs); a dense V x V
matrix is never built.
"""

import math
from pathlib import Path

import numpy as np

from src.vocabulary import NEVER_PREDICTED, START_ID


class BayesianBigram:

    def __init__(self, vocab_size, alpha=0.1):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.V = vocab_size
        self.alpha = float(alpha)
        # Output space O: all ids that can be a next token.
        self.support = np.ones(vocab_size, dtype=bool)
        self.support[list(NEVER_PREDICTED)] = False
        self.n_support = int(self.support.sum())

    # ------------------------------------------------------------------ fit
    def fit(self, sentences):
        """Count transitions inside each <START> ... <END> sentence.

        Pairs never cross sentence boundaries because each encoded
        sentence is processed separately.
        """
        prev = np.concatenate([s[:-1] for s in sentences])
        nxt = np.concatenate([s[1:] for s in sentences])
        keys, counts = np.unique(prev * self.V + nxt, return_counts=True)
        self._set_counts(keys // self.V, keys % self.V, counts)
        return self

    def _set_counts(self, rows, cols, counts):
        # CSR layout: successors of u are cols[indptr[u]:indptr[u+1]].
        self.cols = cols.astype(np.int64)
        self.counts = counts.astype(np.int64)
        self.indptr = np.zeros(self.V + 1, dtype=np.int64)
        np.cumsum(np.bincount(rows, minlength=self.V), out=self.indptr[1:])
        self.row_totals = np.bincount(rows, weights=self.counts,
                                      minlength=self.V).astype(np.int64)

    # ------------------------------------------------------------ inference
    def count(self, u, w):
        """C(u, w) via binary search in u's (sorted) successor list."""
        lo, hi = self.indptr[u], self.indptr[u + 1]
        j = lo + np.searchsorted(self.cols[lo:hi], w)
        return int(self.counts[j]) if j < hi and self.cols[j] == w else 0

    def probs(self, contexts):
        """Posterior predictive rows P(. | u) for an array of contexts: (N, V)."""
        contexts = np.asarray(contexts, dtype=np.int64)
        out = np.zeros((len(contexts), self.V))
        for i, u in enumerate(contexts):
            lo, hi = self.indptr[u], self.indptr[u + 1]
            denom = self.row_totals[u] + self.alpha * self.n_support
            out[i, self.support] = self.alpha / denom
            out[i, self.cols[lo:hi]] += self.counts[lo:hi] / denom
        return out

    def batch_log_probs(self, sentences):
        """Log predictive rows for every position of every sentence.

        Row t of sentence s is log P(. | s[t]) and predicts s[t+1]; only
        the most recent token is used, so no target can leak into its
        own conditioning context.
        """
        out = []
        for s in sentences:
            contexts = s[:-1]
            rows = np.full((len(contexts), self.V), -np.inf)
            for i, u in enumerate(contexts):
                lo, hi = self.indptr[u], self.indptr[u + 1]
                log_denom = np.log(self.row_totals[u] + self.alpha * self.n_support)
                rows[i, self.support] = np.log(self.alpha) - log_denom
                rows[i, self.cols[lo:hi]] = (
                    np.log(self.counts[lo:hi] + self.alpha) - log_denom)
            out.append(rows)
        return out

    def predict(self, prefix_ids):
        """Next-token distribution given an encoded sentence prefix.

        prefix_ids excludes <START>; the context is its last token, or
        <START> for an empty prefix.
        """
        u = prefix_ids[-1] if len(prefix_ids) else START_ID
        return self.probs([u])[0]

    # ------------------------------------------------------ Bayesian evidence
    def log_evidence(self, alpha=None):
        """log p(D | alpha): Dirichlet-multinomial marginal likelihood.

        sum_u [ lgamma(a|O|) - lgamma(n_u + a|O|)
                + sum_w (lgamma(C(u,w) + a) - lgamma(a)) ]
        Zero counts contribute nothing, so only observed pairs are summed.
        """
        a = self.alpha if alpha is None else float(alpha)
        lg = np.vectorize(math.lgamma, otypes=[float])
        n = self.row_totals[self.row_totals > 0]
        a_total = a * self.n_support
        return float(len(n) * math.lgamma(a_total) - lg(n + a_total).sum()
                     + lg(self.counts + a).sum()
                     - len(self.counts) * math.lgamma(a))

    # ---------------------------------------------------------- persistence
    def save(self, path, vocab_sha256=None):
        rows = np.repeat(np.arange(self.V), np.diff(self.indptr))
        np.savez_compressed(path, V=self.V, alpha=self.alpha,
                            rows=rows, cols=self.cols, counts=self.counts,
                            vocab_sha256=str(vocab_sha256 or ""))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        model = cls(int(d["V"]), float(d["alpha"]))
        model._set_counts(d["rows"], d["cols"], d["counts"])
        model.vocab_sha256 = str(d["vocab_sha256"]) if "vocab_sha256" in d else ""
        return model

    def with_alpha(self, alpha):
        """Same counts, different prior strength (for the alpha sweep)."""
        model = BayesianBigram(self.V, alpha)
        model.cols, model.counts = self.cols, self.counts
        model.indptr, model.row_totals = self.indptr, self.row_totals
        return model
