"""
Common evaluation for all four models.

Every model exposes ``batch_log_probs(sentences)`` returning, for each
encoded sentence s = <START> w_1 ... w_n <END>, an array of shape
(len(s) - 1, V) whose row t is log P(. | s[0..t]) -- the distribution
over the token s[t+1] given only the tokens before it.

Targets (identical for all models):
  * every position s[1:], i.e. each word and the final <END>;
  * <UNK> ground-truth targets are excluded and counted separately;
  * <NUM> is an ordinary word target; <PAD>/<START> are never targets.

Probability space: the full output space O = V minus {<PAD>, <START>}.
NLL always uses the model's normalized distribution over O as is (no
renormalization; <UNK> and <END> keep their probability mass).

Two metric groups:
  "all"   -- targets: words + <END>.
             top-k candidates: O minus {<UNK>} (<END> is rankable, as
             an end-of-sentence suggestion).
  "words" -- targets: words only (<END> targets excluded).
             top-k candidates: O minus {<UNK>, <END>}, exactly the set
             the autocomplete interface displays as word suggestions.

A target is in the top k if fewer than k candidates rank above it;
ties are broken by lower id (= higher training frequency), matching the
order in which the interface lists suggestions.
"""

import math

import numpy as np
import torch

from src.vocabulary import END_ID, NEVER_PREDICTED, UNK_ID

TOP_K = (1, 5)


def _ranks(lp, targets):
    """Ranks (0 = best) of targets under the "all" and "words" candidate sets.

    lp: (N, V) float64 tensor of log probabilities; targets: (N,) int64.
    Returns (rank_all, rank_words); rank_words is only meaningful for
    word targets (targets != <END>).
    """
    V = lp.shape[1]
    candidates = torch.ones(V, dtype=torch.bool)
    candidates[list(NEVER_PREDICTED) + [UNK_ID]] = False

    tgt = lp.gather(1, targets[:, None])
    ids = torch.arange(V)[None, :]
    above = (lp > tgt) | ((lp == tgt) & (ids < targets[:, None]))
    above &= candidates
    rank_all = above.sum(dim=1)
    # The words candidate set additionally removes <END>.
    rank_words = rank_all - above[:, END_ID].long()
    return rank_all, rank_words


@torch.no_grad()
def evaluate(model, sentences, batch_size=32, max_sentences=None, ranks=True):
    """Return metrics over all non-<UNK> targets of ``sentences``.

    ranks=False skips top-k accuracy (NLL/perplexity are computed by the
    identical code path); used for cheap per-iteration model selection.
    """

    if max_sentences is not None:
        sentences = sentences[:max_sentences]

    nll, rank_all, rank_words, is_end = [], [], [], []
    n_unk = n_targets = 0
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        lp = torch.as_tensor(np.concatenate(model.batch_log_probs(batch)),
                             dtype=torch.float64)
        targets = torch.as_tensor(np.concatenate([s[1:] for s in batch]))
        assert lp.shape == (len(targets), lp.shape[1])

        # Every predictive distribution must be normalized over O.
        log_sums = torch.logsumexp(lp, dim=1)
        if not torch.allclose(log_sums, torch.zeros_like(log_sums), atol=1e-4):
            raise AssertionError(
                f"log normalizers in [{log_sums.min()}, {log_sums.max()}]")

        n_targets += len(targets)
        keep = targets != UNK_ID
        n_unk += int((~keep).sum())
        lp, targets = lp[keep], targets[keep]

        nll.append(-lp.gather(1, targets[:, None])[:, 0].numpy())
        if ranks:
            r_all, r_words = _ranks(lp, targets)
            rank_all.append(r_all.numpy())
            rank_words.append(r_words.numpy())
        is_end.append((targets == END_ID).numpy())

    nll, is_end = np.concatenate(nll), np.concatenate(is_end)
    if ranks:
        rank_all, rank_words = np.concatenate(rank_all), np.concatenate(rank_words)
    if not np.isfinite(nll).all():
        raise AssertionError("infinite NLL: a target received zero probability")

    def summary(mask, r):
        m = float(nll[mask].mean())
        out = {"targets": int(mask.sum()), "nll": m, "perplexity": math.exp(m)}
        if ranks:
            for k in TOP_K:
                out[f"top{k}"] = float((r[mask] < k).mean())
        return out

    return {
        "sentences": len(sentences),
        "all_targets_incl_unk": n_targets,
        "unk_targets_excluded": n_unk,
        "unk_target_rate": n_unk / n_targets,
        "all": summary(np.ones_like(is_end), rank_all),
        "words": summary(~is_end, rank_words),
    }


def format_metrics(m):
    a, w = m["all"], m["words"]
    if "top1" not in a:
        return (f"all: NLL {a['nll']:.4f} PPL {a['perplexity']:8.2f} | "
                f"words: PPL {w['perplexity']:8.2f} | {a['targets']} targets")
    return (f"all: NLL {a['nll']:.4f} PPL {a['perplexity']:8.2f} "
            f"top1 {a['top1']:.4f} top5 {a['top5']:.4f} | "
            f"words: PPL {w['perplexity']:8.2f} top1 {w['top1']:.4f} "
            f"top5 {w['top5']:.4f} | {a['targets']} targets "
            f"(+{m['unk_targets_excluded']} <UNK> excluded)")
