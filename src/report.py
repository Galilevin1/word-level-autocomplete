"""
Final test evaluation, figures, tables and qualitative examples.

Everything here runs AFTER validation-based model selection: it only
reads the selection files written by run_experiments.py and the four
selected checkpoints. The test split is used for nothing but reporting.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.autocomplete import load_model, predict_next  # noqa: E402
from src.dataset import load_corpus  # noqa: E402
from src.evaluate import _ranks, evaluate  # noqa: E402
from src.vocabulary import (END_ID, NEVER_PREDICTED, NUM_ID,  # noqa: E402
                            UNK_ID)

MODELS = ["bigram", "hmm", "rnn", "gru"]
LABELS = {"bigram": "Bayesian bigram", "hmm": "HMM (Baum-Welch)",
          "rnn": "Vanilla RNN", "gru": "GRU"}
# Fixed categorical slots (validated reference order): colour follows the model.
COLORS = {"bigram": "#2a78d6", "hmm": "#eb6834", "rnn": "#1baf7a", "gru": "#eda100"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SELECTION_FILES = ["bigram_selection.json", "hmm_selection.json",
                   "neural_selection.json"]

PROMPTS = ["", "the game was", "in the", "he was born in", "the album was released",
           "the united states", "it is one of the", "during the", "she", "the zyzzyva was"]

FREQ_BUCKETS = [("top 100 words", 5, 105), ("words 101-1,000", 105, 1005),
                ("words 1,001-5,000", 1005, 5005), ("words 5,001+", 5005, None)]
POS_BUCKETS = [("1st word", 1, 2), ("words 2-5", 2, 6), ("words 6-15", 6, 16),
               ("words 16+", 16, None)]
SHORT = {"top 100 words": "top 100", "words 101-1,000": "101-1k",
         "words 1,001-5,000": "1k-5k", "words 5,001+": "5k+", "1st word": "1st",
         "words 2-5": "2-5", "words 6-15": "6-15", "words 16+": "16+"}


def md(token):
    """Markdown-safe token: reserved tokens such as <NUM> in code spans."""
    return f"`{token}`" if token.startswith("<") else token


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------- test stage
@torch.no_grad()
def target_details(model, sentences, batch_size=32):
    """Per-target NLL / ranks with exactly the evaluator's policy.

    Returns arrays over non-<UNK> targets: target id, position in the
    sentence (1 = first word), nll, rank_all, rank_words, and the best
    displayable word with its probability.
    """
    V = None
    out = {k: [] for k in ("target", "position", "nll", "rank_all",
                           "rank_words", "top1", "top1_p")}
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        lp = torch.as_tensor(np.concatenate(model.batch_log_probs(batch)),
                             dtype=torch.float64)
        V = lp.shape[1]
        targets = torch.as_tensor(np.concatenate([s[1:] for s in batch]))
        pos = torch.as_tensor(np.concatenate([np.arange(1, len(s)) for s in batch]))
        keep = targets != UNK_ID
        lp, targets, pos = lp[keep], targets[keep], pos[keep]
        r_all, r_words = _ranks(lp, targets)
        shown = lp.clone()
        shown[:, list(NEVER_PREDICTED) + [UNK_ID, END_ID]] = -torch.inf
        best = shown.max(dim=1)
        for k, v in (("target", targets), ("position", pos),
                     ("nll", -lp.gather(1, targets[:, None])[:, 0]),
                     ("rank_all", r_all), ("rank_words", r_words),
                     ("top1", best.indices), ("top1_p", best.values.exp())):
            out[k].append(v.numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def _bucket_metrics(d, mask):
    n = int(mask.sum())
    if n == 0:
        return {"targets": 0}
    ranks = np.where(d["target"][mask] == END_ID, d["rank_all"][mask], d["rank_words"][mask])
    return {"targets": n, "share": None,
            "nll": float(d["nll"][mask].mean()),
            "top1": float((ranks < 1).mean()), "top5": float((ranks < 5).mean())}


def breakdowns(d):
    """Accuracy/NLL by target frequency rank and by sentence position.

    Word targets use the words candidate set; <END> the all set.
    """
    t, is_word = d["target"], (d["target"] != END_ID) & (d["target"] != NUM_ID)
    freq = {"<NUM>": _bucket_metrics(d, t == NUM_ID),
            "<END>": _bucket_metrics(d, t == END_ID)}
    for name, lo, hi in FREQ_BUCKETS:
        freq[name] = _bucket_metrics(d, is_word & (t >= lo) & (t < (hi or 10**9)))
    words = t != END_ID
    pos = {name: _bucket_metrics(d, words & (d["position"] >= lo)
                                 & (d["position"] < (hi or 10**9)))
           for name, lo, hi in POS_BUCKETS}
    total = len(t)
    for group in (freq, pos):
        for m in group.values():
            if m.get("targets"):
                m["share"] = m["targets"] / total
    return {"by_target_frequency": freq, "by_position": pos}


DRY_PREFIX = "DRYRUN_validation_"
DRY_TITLE = "DRY RUN (validation split, smoke checkpoints; not a result)"


def fname(base, dry_run):
    """Output file name; dry-run files can never carry a final name."""
    return DRY_PREFIX + base if dry_run else base


def eval_label(dry_run):
    return "Validation (DRY RUN)" if dry_run else "Test"


def verify_selected_checkpoints(ckpt_dir, results_dir):
    """Check that the four inference checkpoints are the selected ones.

    Loads each through the hash-checked loader and compares it with the
    configuration recorded by the validation-selection stage.
    """
    b = load_json(results_dir / "bigram_selection.json")
    h = load_json(results_dir / "hmm_selection.json")
    n = load_json(results_dir / "neural_selection.json")
    vocab, bigram = load_model("bigram", ckpt_dir)
    for sel in (b, h, n):
        assert sel["vocab_sha256"] == vocab.sha256(), "selection used another vocabulary"
    assert bigram.alpha == b["selected_alpha"]
    _, hmm = load_model("hmm", ckpt_dir)
    row = next(r for r in h["grid"] if r["K"] == h["selected_K"])
    assert hmm.K == h["selected_K"] and hmm.best_iteration == row["best_em_step"]
    for cell in ("rnn", "gru"):
        _, m = load_model(cell, ckpt_dir)
        row = next(r for r in n["grid"] if r["name"] == n["selected"][cell])
        assert m.metadata["name"] == row["name"]
        assert m.metadata["epoch"] == row["selected_epoch"]
    print("verified: 4 selected checkpoints match the validation selection "
          f"(vocab sha256 {vocab.sha256()[:12]})")


def run_test_stage(ckpt_dir, results_dir, dry_run=False):
    """Evaluate the four validation-selected checkpoints on the test split.

    dry_run=True exercises the identical code on the VALIDATION split (for
    testing the pipeline before selection is complete); its outputs are
    written under DRYRUN_validation_* names and never used in the report.
    """
    missing = [f for f in SELECTION_FILES if not (results_dir / f).exists()]
    if missing:
        raise SystemExit(f"model selection incomplete (missing {missing}); "
                         "the test split is only evaluated afterwards")
    verify_selected_checkpoints(ckpt_dir, results_dir)
    split = "validation" if dry_run else "test"
    vocab, _ = load_model("bigram", ckpt_dir)
    _, enc = load_corpus(vocab=vocab)
    data = enc[split]
    results = {"dry_run": dry_run, "split": split, "vocab_sha256": vocab.sha256(),
               "sentences": len(data), "checkpoint_dir": str(ckpt_dir),
               "selected": selected_configs(results_dir), "models": {}}
    for name in MODELS:
        _, model = load_model(name, ckpt_dir)
        m = evaluate(model, data)
        d = target_details(model, data)
        # consistency with the common evaluator
        assert abs(d["nll"].mean() - m["all"]["nll"]) < 1e-9
        assert abs((d["rank_all"] < 5).mean() - m["all"]["top5"]) < 1e-12
        results["models"][name] = {"test": m, **breakdowns(d)}
        np.savez_compressed(results_dir / fname(f"test_details_{name}.npz", dry_run), **d)
        print(f"{name:6s} {split}: all NLL {m['all']['nll']:.4f} PPL {m['all']['perplexity']:.2f} "
              f"top1 {m['all']['top1']:.4f} top5 {m['all']['top5']:.4f} | words top1 "
              f"{m['words']['top1']:.4f} top5 {m['words']['top5']:.4f}")
    save_json(results, results_dir / fname("test_results.json", dry_run))
    return results


def selected_configs(results_dir):
    """Selected configuration of each model and its validation metrics."""
    b = load_json(results_dir / "bigram_selection.json")
    h = load_json(results_dir / "hmm_selection.json")
    n = load_json(results_dir / "neural_selection.json")
    out = {}
    row = next(r for r in b["grid"] if r["alpha"] == b["selected_alpha"])
    out["bigram"] = {"config": f"alpha = {b['selected_alpha']}", "validation": row["validation"]}
    row = next(r for r in h["grid"] if r["K"] == h["selected_K"])
    out["hmm"] = {"config": f"K = {row['K']} (EM step {row['best_em_step']})",
                  "validation": row["validation"]}
    for cell in ("rnn", "gru"):
        row = next(r for r in n["grid"] if r["name"] == n["selected"][cell])
        out[cell] = {"config": f"E = {row['emb_dim']}, H = {row['hidden_dim']} "
                               f"(epoch {row['selected_epoch']})",
                     "validation": row["validation"]}
    return out


# ------------------------------------------------------ qualitative examples
def qualitative_examples(ckpt_dir, results_dir, n_sentences=3, seed=0, dry_run=False):
    """Markdown with prompt completions and test-sentence walk-throughs."""
    split = "validation" if dry_run else "test"
    title = "# Qualitative autocomplete examples"
    lines = [f"{title} - {DRY_TITLE}" if dry_run else title, "",
             "Generated by `uv run run_experiments.py --stage report` from the "
             "selected checkpoints. Probabilities are the models' own "
             "(not renormalized); `P(end)` is the end-of-sentence probability.", "",
             "## Fixed prompts (top-5 suggestions)", ""]
    for text in PROMPTS:
        lines += [f"**Prompt:** `{text or '(empty)'}`", "",
                  "| Model | Suggestions | P(end) |", "|---|---|---|"]
        for name in MODELS:
            r = predict_next(text, name, 5, ckpt_dir)
            sugg = ", ".join(f"{md(w)} ({p:.3f})" for w, p in r["suggestions"])
            note = f" (unknown: {', '.join(r['unknown_words'])})" if r["unknown_words"] else ""
            lines.append(f"| {LABELS[name]}{note} | {sugg} | {r['p_end_of_sentence']:.3f} |")
        lines.append("")

    vocab, _ = load_model("bigram", ckpt_dir)
    _, enc = load_corpus(vocab=vocab)
    rng = np.random.default_rng(seed)
    pool = [s for s in enc[split] if 9 <= len(s) <= 16 and UNK_ID not in s]
    chosen = [pool[i] for i in sorted(rng.choice(len(pool), n_sentences, replace=False))]
    lines += [f"## {split.capitalize()}-sentence walk-throughs", "",
              f"{n_sentences} {split} sentences without `<UNK>` (length 7-14 words), "
              f"drawn with seed {seed}. Each cell is the model's top-1 displayable "
              "word given the true prefix; **bold** = top-1 correct, "
              "`+` = target within the top 5.", ""]
    rows_by_model = {name: load_model(name, ckpt_dir)[1].batch_log_probs(chosen)
                     for name in MODELS}
    for j, s in enumerate(chosen):
        words = vocab.decode(s[1:])
        lines += [f"**Sentence {j + 1}:** {' '.join(map(md, words[:-1]))}", "",
                  "| Target | " + " | ".join(LABELS[m] for m in MODELS) + " |",
                  "|---" * (len(MODELS) + 1) + "|"]
        for t, target in enumerate(s[1:]):
            cells = []
            for name in MODELS:
                lp = rows_by_model[name][j][t].copy()
                lp[list(NEVER_PREDICTED) + [UNK_ID]] = -np.inf
                ranked_all = np.argsort(-lp, kind="stable")        # "all" set
                lp[END_ID] = -np.inf
                ranked_words = np.argsort(-lp, kind="stable")      # "words" set
                ranked = ranked_all if target == END_ID else ranked_words
                hit = target in ranked[:5]
                top = md(vocab.itos[ranked_words[0]])
                cell = f"**{top}**" if ranked_words[0] == target else top
                cells.append(cell + (" +" if hit else ""))
            lines.append(f"| {md(vocab.itos[target])} | " + " | ".join(cells) + " |")
        lines.append("")
    (results_dir / fname("qualitative_examples.md", dry_run)).write_text(
        "\n".join(lines), encoding="utf-8")


def confident_errors_and_hits(results_dir, ckpt_dir, n=8, dry_run=False):
    """Most confident correct / wrong top-1 word predictions per model (test)."""
    vocab, _ = load_model("bigram", ckpt_dir)
    out = {"dry_run": dry_run}
    for name in MODELS:
        d = dict(np.load(results_dir / fname(f"test_details_{name}.npz", dry_run)))
        words = d["target"] != END_ID
        correct = words & (d["top1"] == d["target"])
        wrong = words & (d["top1"] != d["target"])
        pick = lambda m: np.argsort(-np.where(m, d["top1_p"], -1))[:n]
        out[name] = {
            "confident_correct": [(vocab.itos[d["top1"][i]], float(d["top1_p"][i]))
                                  for i in pick(correct)],
            "confident_wrong": [(vocab.itos[d["top1"][i]], vocab.itos[d["target"][i]],
                                 float(d["top1_p"][i])) for i in pick(wrong)],
        }
    return out


# ----------------------------------------------------------------- figures
def _style(ax, xlabel=None, ylabel=None, title=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")


def _save(fig, fig_dir, name, dry_run):
    """Save a figure; dry-run figures get a banner and a DRYRUN_ file name."""
    if dry_run:
        fig.text(0.5, 1.02, DRY_TITLE, ha="center", va="bottom", fontsize=12,
                 color="#d03b3b", weight="bold")
    fig.savefig(fig_dir / fname(name, dry_run), dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig_bigram(results_dir, fig_dir, dry_run=False):
    b = load_json(results_dir / "bigram_selection.json")
    stats = load_json(results_dir / "data_stats.json")
    n_transitions = stats["train"]["word_tokens"] + stats["train"]["sentences"]
    a = [r["alpha"] for r in b["grid"]]
    nll = [r["validation"]["all"]["nll"] for r in b["grid"]]
    ev = [-r["log_evidence"] / n_transitions for r in b["grid"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, y, lab, title in (
            (axes[0], nll, "validation NLL (nats / target)", "Validation NLL (selection criterion)"),
            (axes[1], ev, "-log p(D | alpha) / training transition",
             "Negative log evidence (training data)")):
        ax.plot(a, y, color=COLORS["bigram"], lw=2, marker="o", ms=5)
        best = int(np.argmin(y))
        ax.plot(a[best], y[best], "o", ms=9, mfc="white", mec=COLORS["bigram"], mew=2)
        ax.set_xscale("log")
        _style(ax, "Dirichlet concentration alpha (log scale)", lab,
               f"{title}\nminimum (circle) at alpha = {a[best]}")
    fig.suptitle("Bayesian bigram: choice of the Dirichlet prior", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout()
    _save(fig, fig_dir, "bigram_alpha.png", dry_run)


def fig_hmm(results_dir, fig_dir, dry_run=False):
    h = load_json(results_dir / "hmm_selection.json")
    series = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for color, row in zip(series, h["grid"]):
        hist = row["history"]
        steps = [r["em_step"] - 1 for r in hist]       # parameters theta_{t-1}
        axes[0].plot(steps, [r["train_nll_per_token_before"] for r in hist],
                     color=color, lw=2, label=f"K = {row['K']}")
        ev = [(r["em_step"], r["val_nll_after"]) for r in hist if "val_nll_after" in r]
        axes[1].plot(*zip(*ev), color=color, lw=2, marker="o", ms=5, label=f"K = {row['K']}")
        axes[1].plot(row["best_em_step"], dict(ev)[row["best_em_step"]], "o", ms=9,
                     mfc="white", mec=color, mew=2)
        # direct labels on the left panel only (right-panel ends coincide;
        # its legend identifies the series)
        axes[0].annotate(f"K={row['K']}", (steps[-1], hist[-1]["train_nll_per_token_before"]),
                         textcoords="offset points", xytext=(9, 0), color=INK2,
                         fontsize=8, va="center")
    _style(axes[0], "EM iteration", "train NLL / observation (nats)",
           "Baum-Welch training objective")
    _style(axes[1], "EM iteration", "validation NLL (evaluation policy)",
           "Validation NLL (selection; circle = selected)")
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    _save(fig, fig_dir, "hmm_convergence.png", dry_run)


def fig_neural(results_dir, fig_dir, dry_run=False):
    n = load_json(results_dir / "neural_selection.json")
    grid = n["grid"]
    fig, axes = plt.subplots(1, len(grid), figsize=(3.0 * len(grid), 3.0), sharey=True)
    for ax, row in zip(np.atleast_1d(axes), grid):
        ep = [r["epoch"] for r in row["history"]]
        ax.plot(ep, [r["train_loss"] for r in row["history"]], color="#2a78d6", lw=2,
                marker="o", ms=4, label="train loss")
        ax.plot(ep, [r["val_loss"] for r in row["history"]], color="#eb6834", lw=2,
                marker="o", ms=4, label="validation loss")
        sel = row["selected_epoch"]
        ax.axvline(sel, color=AXIS, lw=1, ls="--")
        ax.annotate("selected", (sel, ax.get_ylim()[1]), textcoords="offset points",
                    xytext=(3, -12), color=MUTED, fontsize=8)
        _style(ax, "epoch", "cross-entropy (nats / token)" if ax is axes[0] else None,
               f"{row['cell'].upper()}  H = {row['hidden_dim']}")
        ax.set_xticks(ep)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("RNN / GRU learning curves (teacher-forced cross-entropy incl. <UNK>)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    _save(fig, fig_dir, "neural_curves.png", dry_run)


def _bars(ax, values, fmt):
    x = np.arange(len(MODELS))
    bars = ax.bar(x, values, width=0.62, color=[COLORS[m] for m in MODELS],
                  edgecolor="white", linewidth=2)
    for b, v in zip(bars, values):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    color=INK, fontsize=8)
    ax.set_xticks(x, ["Bigram", "HMM", "RNN", "GRU"], fontsize=9)


def fig_test(results_dir, fig_dir, dry_run=False):
    t = load_json(results_dir / fname("test_results.json", dry_run))["models"]
    lab = eval_label(dry_run)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    _bars(axes[0], [t[m]["test"]["all"]["perplexity"] for m in MODELS], "{:.0f}")
    _style(axes[0], None, "perplexity (lower is better)", f"{lab} perplexity")
    _bars(axes[1], [100 * t[m]["test"]["words"]["top1"] for m in MODELS], "{:.1f}%")
    _style(axes[1], None, "% of word targets", f"{lab} top-1 word accuracy")
    _bars(axes[2], [100 * t[m]["test"]["words"]["top5"] for m in MODELS], "{:.1f}%")
    _style(axes[2], None, "% of word targets", f"{lab} top-5 word accuracy")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, fig_dir, "test_comparison.png", dry_run)


def fig_breakdown(results_dir, fig_dir, dry_run=False):
    t = load_json(results_dir / fname("test_results.json", dry_run))["models"]
    lab = eval_label(dry_run)
    groups = [("by_target_frequency", ["<NUM>"] + [b[0] for b in FREQ_BUCKETS] + ["<END>"],
               f"{lab}: top-5 accuracy by target type",
               "target (word buckets by training-frequency rank; share of targets)"),
              ("by_position", [b[0] for b in POS_BUCKETS],
               f"{lab}: top-5 word accuracy by position",
               "position of the target word (share of targets)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [6, 4]})
    for ax, (key, buckets, title, xlabel) in zip(axes, groups):
        x = np.arange(len(buckets))
        w = 0.8 / len(MODELS)
        for i, m in enumerate(MODELS):
            vals = [100 * t[m][key][b]["top5"] for b in buckets]
            ax.bar(x + (i - 1.5) * w, vals, width=w, color=COLORS[m], label=LABELS[m],
                   edgecolor="white", linewidth=1)
        share = [t[MODELS[0]][key][b]["share"] for b in buckets]
        ax.set_xticks(x, [f"{SHORT.get(b, b)}\n{s:.1%}" for b, s in zip(buckets, share)],
                      fontsize=8)
        _style(ax, xlabel, "top-5 accuracy (%)", title)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=4,
                   loc="upper center", bbox_to_anchor=(0.5, -0.3))
    fig.tight_layout()
    _save(fig, fig_dir, "test_breakdown.png", dry_run)


# ------------------------------------------------------------------- tables
def _fmt_metrics(m):
    a, w = m["all"], m["words"]
    return (f"{a['nll']:.3f} | {a['perplexity']:.1f} | {100 * a['top1']:.2f} | "
            f"{100 * a['top5']:.2f} | {100 * w['top1']:.2f} | {100 * w['top5']:.2f}")


METRIC_HEAD = ("NLL | PPL | top-1 all (%) | top-5 all (%) | top-1 words (%) | "
               "top-5 words (%)")
METRIC_SEP = "---|---|---|---|---|---"


def tables_markdown(results_dir, dry_run=False):
    stats = load_json(results_dir / "data_stats.json")
    b = load_json(results_dir / "bigram_selection.json")
    h = load_json(results_dir / "hmm_selection.json")
    n = load_json(results_dir / "neural_selection.json")
    L = [f"# Result tables - {DRY_TITLE}" if dry_run else "# Result tables", "",
         "All numbers are measured by `run_experiments.py`. Metrics exclude `<UNK>` "
         "targets; NLL in nats per target over the full output vocabulary; "
         "\"all\" includes `<END>` targets, \"words\" excludes them (see README).", "",
         "## Data", "", "| Split | Sentences | Word tokens | `<UNK>` rate | `<NUM>` tokens |",
         "|---|---|---|---|---|"]
    for k in ("train", "validation", "test"):
        s = stats[k]
        L.append(f"| {k} | {s['sentences']:,} | {s['word_tokens']:,} | "
                 f"{100 * s['unk_rate']:.2f}% | {s['num_tokens']:,} |")
    L += ["", f"Vocabulary: min_count = {stats['min_count']}, |V| = {stats['vocab_size']:,} "
          f"(5 reserved), sha256 `{stats['vocab_sha256'][:16]}...`", "",
          "| min_count | \\|V\\| | train `<UNK>` | validation `<UNK>` | test `<UNK>` |",
          "|---|---|---|---|---|"]
    for r in stats["min_count_sweep"]:
        L.append(f"| {r['min_count']} | {r['vocab_size']:,} | {100 * r['train_unk_rate']:.2f}% | "
                 f"{100 * r['validation_unk_rate']:.2f}% | {100 * r['test_unk_rate']:.2f}% |")

    L += ["", "## Validation: Bayesian bigram (alpha sweep)", "",
          f"| alpha | log evidence | {METRIC_HEAD} |", f"|---|---|{METRIC_SEP}|"]
    for r in b["grid"]:
        mark = " **(selected)**" if r["alpha"] == b["selected_alpha"] else ""
        L.append(f"| {r['alpha']}{mark} | {r['log_evidence']:.4e} | {_fmt_metrics(r['validation'])} |")

    L += ["", "## Validation: HMM (Baum-Welch)", "",
          f"| K | EM steps run | selected step | train time (min) | {METRIC_HEAD} |",
          f"|---|---|---|---|{METRIC_SEP}|"]
    for r in h["grid"]:
        mark = " **(selected)**" if r["K"] == h["selected_K"] else ""
        L.append(f"| {r['K']}{mark} | {r['em_steps_run']} | {r['best_em_step']} | "
                 f"{r['train_seconds'] / 60:.1f} | {_fmt_metrics(r['validation'])} |")

    L += ["", "## Validation: RNN / GRU", "",
          f"| Model | Parameters | Epochs run | Selected epoch | Train time (min) | {METRIC_HEAD} |",
          f"|---|---|---|---|---|{METRIC_SEP}|"]
    for r in n["grid"]:
        mark = " **(selected)**" if n["selected"][r["cell"]] == r["name"] else ""
        L.append(f"| {r['name']}{mark} | {r['parameters']['total']:,} | {len(r['history'])} | "
                 f"{r['selected_epoch']} | {r['train_seconds'] / 60:.1f} | "
                 f"{_fmt_metrics(r['validation'])} |")

    test_path = results_dir / fname("test_results.json", dry_run)
    lab = eval_label(dry_run)
    if test_path.exists():
        t = load_json(test_path)
        assert t["dry_run"] == dry_run
        L += ["", f"## Selected models: validation vs. {lab.lower()}", "",
              f"| Model | Configuration | Split | {METRIC_HEAD} |", f"|---|---|---|{METRIC_SEP}|"]
        for m in MODELS:
            sel = t["selected"][m]
            L.append(f"| {LABELS[m]} | {sel['config']} | validation | {_fmt_metrics(sel['validation'])} |")
            L.append(f"| {LABELS[m]} | {sel['config']} | **{lab}** | {_fmt_metrics(t['models'][m]['test'])} |")
        for key, title in (("by_target_frequency", "target type"), ("by_position", "position")):
            buckets = list(t["models"][MODELS[0]][key])
            L += ["", f"## {lab} top-5 accuracy (%) by {title}", "",
                  "| Bucket | share of targets | " + " | ".join(LABELS[m] for m in MODELS) + " |",
                  "|---|---|" + "---|" * len(MODELS)]
            for bk in buckets:
                share = t["models"][MODELS[0]][key][bk]["share"]
                L.append(f"| {md(bk)} | {100 * share:.1f}% | " + " | ".join(
                    f"{100 * t['models'][m][key][bk]['top5']:.1f}" for m in MODELS) + " |")
    (results_dir / fname("tables.md", dry_run)).write_text("\n".join(L) + "\n", encoding="utf-8")


def run_report_stage(ckpt_dir, results_dir, dry_run=False):
    """Figures, tables and examples; the test-dependent ones need test results."""
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig_bigram(results_dir, fig_dir, dry_run)
    fig_hmm(results_dir, fig_dir, dry_run)
    fig_neural(results_dir, fig_dir, dry_run)
    if (results_dir / fname("test_results.json", dry_run)).exists():
        fig_test(results_dir, fig_dir, dry_run)
        fig_breakdown(results_dir, fig_dir, dry_run)
        qualitative_examples(ckpt_dir, results_dir, dry_run=dry_run)
        save_json(confident_errors_and_hits(results_dir, ckpt_dir, dry_run=dry_run),
                  results_dir / fname("confident_predictions.json", dry_run))
    tables_markdown(results_dir, dry_run)
    print(f"wrote {fig_dir} and {results_dir / fname('tables.md', dry_run)}")
