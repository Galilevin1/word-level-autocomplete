"""
Shared training loop for the RNN and GRU language models.

Both models are trained with exactly the same procedure: teacher-forced
next-token prediction, token-level cross-entropy that ignores <PAD>,
Adam, gradient-norm clipping, learning-rate halving when validation
stops improving, and early stopping. The checkpoint kept is the epoch
with the lowest validation NLL under the common evaluation policy
(src/evaluate.py), the same criterion used for the bigram and the HMM.
"""

import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from models.recurrent import pad
from src.vocabulary import PAD_ID


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 64          # sentences per batch
    lr: float = 2e-3
    clip: float = 1.0             # max global gradient norm
    lr_decay: float = 0.5         # factor when validation NLL does not improve
    patience: int = 2             # epochs without improvement before stopping
    seed: int = 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_tensors(sentences):
    """Teacher forcing: inputs s[:-1] and targets s[1:], right-padded."""
    inputs = pad([s[:-1] for s in sentences])
    targets = pad([s[1:] for s in sentences])
    return inputs, targets


def make_batches(sentences, batch_size, rng):
    """Length-bucketed batches (little padding), shuffled every epoch."""
    order = np.argsort([len(s) for s in sentences], kind="stable")
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches


LOSS = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")


def token_loss(model, inputs, targets):
    """(summed cross-entropy, number of non-<PAD> targets)."""
    logits, _ = model(inputs)
    loss = LOSS(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return loss, int((targets != PAD_ID).sum())


def train_epoch(model, sentences, optimizer, cfg, rng):
    model.train()
    device = model.device
    total, n_tokens, t0 = 0.0, 0, time.time()
    grad_norms = []
    for idx in make_batches(sentences, cfg.batch_size, rng):
        inputs, targets = batch_tensors([sentences[i] for i in idx])
        loss_sum, n = token_loss(model, inputs.to(device), targets.to(device))
        optimizer.zero_grad()
        (loss_sum / n).backward()
        grad_norms.append(float(nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)))
        optimizer.step()
        total += loss_sum.item()
        n_tokens += n
    seconds = time.time() - t0
    return {"train_loss": total / n_tokens, "tokens_per_second": n_tokens / seconds,
            "epoch_seconds": seconds, "grad_norm_median": float(np.median(grad_norms))}


@torch.no_grad()
def mean_loss(model, sentences, batch_size=128):
    """Mean token cross-entropy (the training objective, incl. <UNK>)."""
    model.eval()
    total, n_tokens = 0.0, 0
    for i in range(0, len(sentences), batch_size):
        inputs, targets = batch_tensors(sentences[i:i + batch_size])
        loss_sum, n = token_loss(model, inputs.to(model.device), targets.to(model.device))
        total += float(loss_sum)
        n_tokens += n
    return total / n_tokens


def train_model(model, train, validation, cfg, checkpoint_path, select_fn,
                log=print, **metadata):
    """Train; save the epoch with the lowest select_fn(model) value.

    select_fn(model) -> validation NLL under the common evaluation policy.
    Returns the per-epoch history.
    """
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history, best, stale = [], float("inf"), 0
    for epoch in range(1, cfg.epochs + 1):
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row.update(train_epoch(model, train, optimizer, cfg, rng))
        row["val_loss"] = mean_loss(model, validation)
        row["val_nll"] = float(select_fn(model))
        if not np.isfinite(row["train_loss"]):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
        improved = row["val_nll"] < best
        if improved:
            best, stale = row["val_nll"], 0
            model.save(checkpoint_path, epoch=epoch, val_nll=best,
                       train_config=asdict(cfg), **metadata)
        else:
            stale += 1
            for g in optimizer.param_groups:
                g["lr"] *= cfg.lr_decay
        row["selected"] = improved
        history.append(row)
        log(f"  epoch {epoch:2d}  train loss {row['train_loss']:.4f}  "
            f"val loss {row['val_loss']:.4f}  val NLL (eval policy) {row['val_nll']:.4f}"
            f"  {row['tokens_per_second']:,.0f} tok/s  {row['epoch_seconds']:.0f}s"
            + ("  *saved" if improved else ""))
        if stale >= cfg.patience:
            break
    return history
