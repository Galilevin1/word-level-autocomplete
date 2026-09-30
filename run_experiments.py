"""
Experiment driver: data statistics, model training/selection, evaluation.

    uv run run_experiments.py --stage data
    uv run run_experiments.py --stage bigram [--smoke]
    uv run run_experiments.py --stage hmm [--smoke] [--hmm-states 16 64 128]
    uv run run_experiments.py --stage neural [--smoke] [--cells rnn gru] [--hidden 128 256]
    uv run run_experiments.py --stage test      # only after all selection stages
    uv run run_experiments.py --stage report    # figures, tables, examples
"""

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch

from models.bigram import BayesianBigram
from models.hmm import HMM
from models.recurrent import RecurrentLM
from src.train import TrainConfig, get_device, set_seed, train_model
from src.dataset import SPLITS, corpus_stats, load_corpus, load_tokenized_splits
from src.evaluate import evaluate, format_metrics
from src.report import run_report_stage, run_test_stage
from src.vocabulary import DEFAULT_MIN_COUNT, Vocabulary, coverage_stats

RESULTS = Path("results")
CHECKPOINTS = Path("checkpoints")
VOCAB_PATH = CHECKPOINTS / "vocab.json"


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def stage_data(min_count):
    """Build the training-only vocabulary and record corpus statistics."""

    vocab, encoded = load_corpus(min_count=min_count)
    stats = corpus_stats(vocab, encoded)

    # Vocabulary size / <UNK> coverage for alternative thresholds
    # (justification of the chosen min_count).
    splits = load_tokenized_splits()
    sweep = []
    for mc in (1, 2, 3, 5, 10):
        v = Vocabulary.build(splits["train"], min_count=mc)
        row = {"min_count": mc, "vocab_size": len(v)}
        for k in SPLITS:
            enc = [v.encode(s) for s in splits[k]]
            row[f"{k}_unk_rate"] = coverage_stats(v, enc)["unk_rate"]
        sweep.append(row)
    stats["min_count_sweep"] = sweep

    written = freeze_vocabulary(vocab, VOCAB_PATH)
    stats["vocab_sha256"] = vocab.sha256()
    save_json(stats, RESULTS / "data_stats.json")
    print(("wrote " if written else "unchanged, reused ") + str(VOCAB_PATH)
          + f" (sha256 {vocab.sha256()[:12]})")

    print(f"min_count={min_count}  |V|={len(vocab)} (incl. 5 reserved)")
    for k in SPLITS:
        s = stats[k]
        print(f"  {k:10s} {s['sentences']:6d} sentences {s['word_tokens']:8d}"
              f" word tokens  <UNK> {s['unk_rate']:.2%}"
              f"  <NUM> {s['num_tokens']}")
    print("min_count sweep:")
    for r in sweep:
        print(f"  {r['min_count']:3d}: |V|={r['vocab_size']:6d}  " + "  ".join(
            f"{k}={r[f'{k}_unk_rate']:.2%}" for k in SPLITS))
    print(f"saved {RESULTS / 'data_stats.json'}")


def load_data(args):
    """Saved vocabulary + encoded splits, truncated in --smoke mode."""

    vocab = Vocabulary.load(VOCAB_PATH)
    _, encoded = load_corpus(vocab=vocab)
    if args.smoke:
        encoded["train"] = encoded["train"][:args.smoke_train]
        encoded["validation"] = encoded["validation"][:args.smoke_eval]
        encoded["test"] = encoded["test"][:args.smoke_eval]
    return vocab, encoded


def freeze_vocabulary(vocab, path):
    """Write the vocabulary once; afterwards only an identical one is accepted.

    An unchanged vocabulary is reused without rewriting. A different
    mapping is always refused (--overwrite does not apply): checkpoints
    trained on the old ids would otherwise remain in place as if valid.
    """
    path = Path(path)
    if path.exists():
        existing = Vocabulary.load(path)
        if existing.sha256() != vocab.sha256():
            raise SystemExit(
                f"{path} is frozen (sha256 {existing.sha256()[:12]}) and the "
                f"rebuilt vocabulary differs ({vocab.sha256()[:12]}). Clear "
                f"{path.parent}/ and retrain all models to change it.")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(path)
    return True


def output_dirs(args, outputs):
    """(results_dir, checkpoint_dir) for a stage.

    Smoke runs write to */smoke/ (with their own vocab.json copy) and
    never touch final results. A final run refuses to overwrite any of
    ``outputs`` (paths relative to results/ or checkpoints/) unless
    --overwrite is given.
    """
    if args.smoke:
        results_dir, ckpt_dir = RESULTS / "smoke", CHECKPOINTS / "smoke"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(VOCAB_PATH, ckpt_dir / "vocab.json")
        return results_dir, ckpt_dir
    existing = [p for p in outputs if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {', '.join(map(str, existing))} "
                         "(pass --overwrite to replace them)")
    return RESULTS, CHECKPOINTS


BIGRAM_ALPHAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def stage_bigram(args):
    """Count transitions once, select alpha by validation NLL."""

    results_dir, ckpt_dir = output_dirs(
        args, [RESULTS / "bigram_selection.json", CHECKPOINTS / "bigram.npz"])
    vocab, encoded = load_data(args)

    t0 = time.time()
    base = BayesianBigram(len(vocab)).fit(encoded["train"])
    count_seconds = time.time() - t0
    print(f"bigram: {len(base.counts)} distinct observed transitions "
          f"from {len(encoded['train'])} training sentences ({count_seconds:.1f}s)")

    rows = []
    for alpha in BIGRAM_ALPHAS:
        model = base.with_alpha(alpha)
        t0 = time.time()
        m = evaluate(model, encoded["validation"])
        rows.append({"alpha": alpha, "log_evidence": model.log_evidence(),
                     "validation": m, "eval_seconds": time.time() - t0})
        print(f"  alpha={alpha:<6} log p(D|a)={rows[-1]['log_evidence']:.4e}  "
              f"{format_metrics(m)}  ({time.time() - t0:.1f}s)")

    best = min(rows, key=lambda r: r["validation"]["all"]["nll"])
    print(f"selected alpha={best['alpha']} (lowest validation NLL); "
          f"highest evidence at alpha="
          f"{max(rows, key=lambda r: r['log_evidence'])['alpha']}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base.with_alpha(best["alpha"]).save(ckpt_dir / "bigram.npz",
                                        vocab_sha256=vocab.sha256())
    save_json({"vocab_sha256": vocab.sha256(),
               "train_sentences": len(encoded["train"]),
               "validation_sentences": len(encoded["validation"]),
               "distinct_transitions": len(base.counts),
               "count_seconds": count_seconds,
               "selected_alpha": best["alpha"], "grid": rows,
               "checkpoint": str(ckpt_dir / "bigram.npz")},
              results_dir / "bigram_selection.json")
    print(f"saved {ckpt_dir / 'bigram.npz'}")


def stage_hmm(args):
    """Baum-Welch for each K; select iteration and K by validation NLL."""

    results_dir, ckpt_dir = output_dirs(
        args, [RESULTS / "hmm_selection.json", CHECKPOINTS / "hmm.npz",
               *(CHECKPOINTS / "hmm_grid" / f"hmm_K{K}.npz" for K in args.hmm_states)])
    vocab, encoded = load_data(args)
    (ckpt_dir / "hmm_grid").mkdir(parents=True, exist_ok=True)
    val = encoded["validation"]

    def select_fn(model):
        return evaluate(model, val, ranks=False)["all"]["nll"]

    rows = []
    for K in args.hmm_states:
        print(f"HMM K={K}: Baum-Welch on {len(encoded['train'])} training sentences")
        t0 = time.time()
        model = HMM(K, len(vocab), seed=args.seed).init_params(encoded["train"])
        model.fit(encoded["train"], select_fn=select_fn,
                  eval_every=args.hmm_eval_every, max_iter=args.hmm_max_iter,
                  tol=args.hmm_tol)
        train_seconds = time.time() - t0
        m = evaluate(model, val)
        print(f"  K={K} selected EM step {model.best_iteration}: {format_metrics(m)}")
        path = ckpt_dir / "hmm_grid" / f"hmm_K{K}.npz"
        model.save(path, vocab_sha256=vocab.sha256())
        rows.append({"K": K, "seed": args.seed,
                     "eps": [model.eps_pi, model.eps_A, model.eps_B],
                     "best_em_step": model.best_iteration,
                     "em_steps_run": len(model.history),
                     "train_seconds": train_seconds, "validation": m,
                     "history": model.history, "checkpoint": str(path)})

    best = min(rows, key=lambda r: r["validation"]["all"]["nll"])
    shutil.copyfile(best["checkpoint"], ckpt_dir / "hmm.npz")
    save_json({"vocab_sha256": vocab.sha256(),
               "train_sentences": len(encoded["train"]),
               "validation_sentences": len(val), "seed": args.seed,
               "max_iter": args.hmm_max_iter, "tol": args.hmm_tol,
               "eval_every": args.hmm_eval_every,
               "selected_K": best["K"], "grid": rows},
              results_dir / "hmm_selection.json")
    print(f"selected K={best['K']}; saved {ckpt_dir / 'hmm.npz'}")


def stage_neural(args):
    """Train RNN and GRU with identical data, budget and selection rule."""

    names = [f"{c}_E{args.emb}_H{h}" for c in args.cells for h in args.hidden]
    results_dir, ckpt_dir = output_dirs(
        args, [RESULTS / "neural_selection.json",
               *(CHECKPOINTS / f"{c}.pt" for c in args.cells),
               *(CHECKPOINTS / "neural_grid" / f"{n}.pt" for n in names)])
    vocab, encoded = load_data(args)
    (ckpt_dir / "neural_grid").mkdir(parents=True, exist_ok=True)
    train, val = encoded["train"], encoded["validation"]
    device = get_device()
    env = {"device": str(device), "cuda_available": torch.cuda.is_available(),
           "torch": torch.__version__, "cpu_threads": torch.get_num_threads()}
    print(f"device: {env}")

    def select_fn(model):
        return evaluate(model, val, ranks=False)["all"]["nll"]

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, clip=args.clip, seed=args.seed)
    rows = []
    for cell in args.cells:
        for hidden in args.hidden:
            name = f"{cell}_E{args.emb}_H{hidden}"
            print(f"{name}: training on {len(train)} sentences")
            set_seed(args.seed)
            model = RecurrentLM(len(vocab), cell, args.emb, hidden,
                                dropout=args.dropout).to(device)
            path = ckpt_dir / "neural_grid" / f"{name}.pt"
            t0 = time.time()
            history = train_model(model, train, val, cfg, path, select_fn,
                                  name=name, train_sentences=len(train),
                                  vocab_sha256=vocab.sha256())
            train_seconds = time.time() - t0
            best = RecurrentLM.load(path, device)
            m = evaluate(best, val)
            print(f"  {name} selected epoch {best.metadata['epoch']}: {format_metrics(m)}")
            rows.append({"name": name, "cell": cell, "emb_dim": args.emb,
                         "hidden_dim": hidden, "dropout": args.dropout,
                         "seed": args.seed, "train_seconds": train_seconds,
                         "parameters": model.parameter_counts(),
                         "selected_epoch": best.metadata["epoch"],
                         "validation": m, "history": history,
                         "checkpoint": str(path)})

    selected = {}
    for cell in args.cells:
        best = min((r for r in rows if r["cell"] == cell),
                   key=lambda r: r["validation"]["all"]["nll"])
        shutil.copyfile(best["checkpoint"], ckpt_dir / f"{cell}.pt")
        selected[cell] = best["name"]
        print(f"selected {cell}: {best['name']} -> {ckpt_dir / f'{cell}.pt'}")

    save_json({"vocab_sha256": vocab.sha256(),
               "environment": env, "train_config": asdict(cfg),
               "train_sentences": len(train), "validation_sentences": len(val),
               "selected": selected, "grid": rows},
              results_dir / "neural_selection.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=["data", "bigram", "hmm", "neural", "test", "report"])
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true",
                        help="rerun a model stage, replacing its results and "
                             "checkpoints (the vocabulary can never change)")
    parser.add_argument("--smoke", action="store_true",
                        help="small data subset; outputs go to */smoke/")
    parser.add_argument("--smoke-train", type=int, default=5000)
    parser.add_argument("--smoke-eval", type=int, default=500)
    parser.add_argument("--hmm-states", type=int, nargs="+", default=[16, 64, 128])
    parser.add_argument("--hmm-max-iter", type=int, default=50)
    parser.add_argument("--hmm-eval-every", type=int, default=5)
    parser.add_argument("--hmm-tol", type=float, default=1e-4)
    parser.add_argument("--cells", nargs="+", default=["rnn", "gru"],
                        choices=["rnn", "gru"])
    parser.add_argument("--emb", type=int, default=128)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--clip", type=float, default=1.0)
    args = parser.parse_args()

    if args.stage == "data":
        stage_data(args.min_count)
    elif args.stage == "bigram":
        stage_bigram(args)
    elif args.stage == "hmm":
        stage_hmm(args)
    elif args.stage == "neural":
        stage_neural(args)
    elif args.stage == "test":
        # Only after all three validation-selection stages; verifies that the
        # four checkpoints are the selected ones before touching test data.
        output_dirs(args, [RESULTS / "test_results.json"])
        run_test_stage(CHECKPOINTS, RESULTS)
    elif args.stage == "report":
        run_report_stage(CHECKPOINTS, RESULTS)


if __name__ == "__main__":
    main()
