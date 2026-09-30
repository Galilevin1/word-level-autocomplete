"""
Models C and D: word-level recurrent language models (vanilla RNN / GRU).

    ids (B, T) -> nn.Embedding -> (B, T, E) -> nn.RNN | nn.GRU -> (B, T, H)
               -> nn.Linear -> logits (B, T, V)

The two models share every component except the recurrent cell:
  RNN:  h_t = tanh(W_x x_t + W_h h_{t-1} + b)
  GRU:  r_t = sigma(W_r x_t + U_r h_{t-1})            (reset gate)
        z_t = sigma(W_z x_t + U_z h_{t-1})            (update gate)
        n_t = tanh(W_n x_t + r_t * (U_n h_{t-1}))     (candidate state)
        h_t = (1 - z_t) * n_t + z_t * h_{t-1}

The input is <START> w_1 ... w_n and the target at step t is the next
token, so logits[:, t] predict s[t+1] from s[0..t] only (the recurrence
is causal). Sequences are right-padded; padded steps come after all real
steps and therefore cannot influence them. <PAD> and <START> logits are
set to -inf, so the softmax is a distribution over the same output
space as the other models.
"""

import numpy as np
import torch
from torch import nn

from src.vocabulary import NEVER_PREDICTED, PAD_ID, START_ID

CELLS = {"rnn": nn.RNN, "gru": nn.GRU}


class RecurrentLM(nn.Module):

    def __init__(self, vocab_size, cell="gru", emb_dim=128, hidden_dim=256,
                 num_layers=1, dropout=0.2):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, cell=cell, emb_dim=emb_dim,
                           hidden_dim=hidden_dim, num_layers=num_layers,
                           dropout=dropout)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.rnn = CELLS[cell](emb_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, vocab_size)
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        mask[list(NEVER_PREDICTED)] = True
        self.register_buffer("never_predicted", mask, persistent=False)

    def forward(self, ids, hidden=None):
        """ids (B, T) -> logits (B, T, V), final hidden state."""
        x = self.dropout(self.embedding(ids))
        h, hidden = self.rnn(x, hidden)
        logits = self.out(self.dropout(h))
        return logits.masked_fill(self.never_predicted, float("-inf")), hidden

    @property
    def device(self):
        return self.out.weight.device

    def parameter_counts(self):
        """Trainable parameters per component and in total."""
        parts = {name: sum(p.numel() for p in module.parameters())
                 for name, module in (("embedding", self.embedding),
                                      ("recurrent", self.rnn),
                                      ("output", self.out))}
        parts["total"] = sum(parts.values())
        return parts

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def batch_log_probs(self, sentences):
        """Row t is log P(. | s[0..t]) for each encoded sentence s."""
        self.eval()
        inputs = pad([s[:-1] for s in sentences]).to(self.device)
        logp = torch.log_softmax(self(inputs)[0].float(), dim=-1).cpu().numpy()
        return [logp[i, :len(s) - 1] for i, s in enumerate(sentences)]

    @torch.no_grad()
    def predict(self, prefix_ids):
        """P(next token | <START> + prefix) as a numpy array."""
        self.eval()
        ids = torch.tensor([[START_ID, *prefix_ids]], device=self.device)
        return torch.softmax(self(ids)[0][0, -1].double(), dim=-1).cpu().numpy()

    # ---------------------------------------------------------- persistence
    def save(self, path, **metadata):
        """Inference checkpoint: architecture config + weights + metadata."""
        torch.save({"config": self.config, "state_dict": self.state_dict(),
                    "metadata": metadata}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = cls(**ckpt["config"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.metadata = ckpt["metadata"]
        model.vocab_sha256 = model.metadata.get("vocab_sha256", "")
        return model.eval()


def pad(seqs, value=PAD_ID):
    """List of 1-D int arrays -> right-padded LongTensor (B, T_max)."""
    T = max(len(s) for s in seqs)
    out = np.full((len(seqs), T), value, dtype=np.int64)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return torch.from_numpy(out)
