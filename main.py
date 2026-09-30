"""
Next-word autocomplete from the command line.

    uv run main.py --model gru --text "the game was" --top-k 5
    uv run main.py --model all --text "in the"
    uv run main.py --smoke --model hmm --text "the"   # smoke-test checkpoints

By default the selected models in checkpoints/ (full training) are used.
"""

import argparse

from src.autocomplete import (DEFAULT_CHECKPOINT_DIR, MODEL_FILES,
                              SMOKE_CHECKPOINT_DIR, predict_next)


def show(result):
    print(f"[{result['model']}] context: {' '.join(result['context']) or '(start of sentence)'}")
    if result["unknown_words"]:
        print(f"  unknown words (-> <UNK>): {', '.join(result['unknown_words'])}")
    for rank, (word, prob) in enumerate(result["suggestions"], 1):
        print(f"  {rank}. {word:<20s} {prob:.4f}")
    print(f"  P(end of sentence) = {result['p_end_of_sentence']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Suggest the next word.")
    parser.add_argument("--model", default="gru", choices=[*MODEL_FILES, "all"])
    parser.add_argument("--text", default="", help="text typed so far")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--smoke", action="store_true",
                        help=f"use the smoke-test checkpoints in {SMOKE_CHECKPOINT_DIR}")
    parser.add_argument("--checkpoint-dir", default=None,
                        help=f"override the checkpoint directory "
                             f"(default {DEFAULT_CHECKPOINT_DIR})")
    args = parser.parse_args()

    ckpt = args.checkpoint_dir or (SMOKE_CHECKPOINT_DIR if args.smoke
                                   else DEFAULT_CHECKPOINT_DIR)
    models = list(MODEL_FILES) if args.model == "all" else [args.model]
    for name in models:
        show(predict_next(args.text, name, args.top_k, ckpt))


if __name__ == "__main__":
    main()
