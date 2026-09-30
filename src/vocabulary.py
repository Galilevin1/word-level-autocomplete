"""
Word <-> id mapping shared by all four models.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.preprocess import NUM_TOKEN

PAD, UNK, START, END, NUM = "<PAD>", "<UNK>", "<START>", "<END>", NUM_TOKEN
SPECIALS = [PAD, UNK, START, END, NUM]
PAD_ID, UNK_ID, START_ID, END_ID, NUM_ID = range(len(SPECIALS))

# <NUM> is reserved but is an ordinary prediction target. <PAD> and
# <START> can never be the next token; <UNK> and <END> are valid outputs
# of every model but are not shown as word suggestions.
NEVER_PREDICTED = (PAD_ID, START_ID)
NOT_SUGGESTED = (PAD_ID, UNK_ID, START_ID, END_ID)

DEFAULT_MIN_COUNT = 5


class Vocabulary:
    """Fixed vocabulary; ids 0-4 are the reserved tokens."""

    def __init__(self, words, min_count=None):
        self.itos = list(SPECIALS) + [w for w in words if w not in SPECIALS]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.min_count = min_count

    @classmethod
    def build(cls, train_sentences, min_count=DEFAULT_MIN_COUNT):
        """Build from TRAINING sentences only; rarer words map to <UNK>.

        Words are ordered by descending frequency (ties alphabetical) so
        the mapping is deterministic.
        """
        counts = Counter(w for s in train_sentences for w in s)
        words = sorted((w for w, c in counts.items() if c >= min_count),
                       key=lambda w: (-counts[w], w))
        return cls(words, min_count)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        """Token list -> int64 id array (unknown words -> <UNK>)."""
        get = self.stoi.get
        return np.fromiter((get(t, UNK_ID) for t in tokens),
                           dtype=np.int64, count=len(tokens))

    def encode_sentence(self, tokens):
        """Encode one sentence as <START> w_1 ... w_n <END>.

        This is the only place where boundary markers are inserted.
        """
        return np.concatenate(([START_ID], self.encode(tokens), [END_ID]))

    def decode(self, ids):
        return [self.itos[int(i)] for i in ids]

    def sha256(self):
        """Deterministic fingerprint of the ordered id -> token mapping.

        Covers the full ``itos`` including reserved tokens. Stored in every
        model checkpoint and verified at load time, since two vocabularies
        of equal size can map ids to different words.
        """
        data = json.dumps(self.itos, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def save(self, path):
        data = {"min_count": self.min_count, "itos": self.itos}
        Path(path).write_text(json.dumps(data, ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        itos = data["itos"]
        if itos[:len(SPECIALS)] != SPECIALS:
            raise ValueError(f"{path}: unexpected special tokens")
        return cls(itos[len(SPECIALS):], data["min_count"])


def coverage_stats(vocab, encoded_sentences):
    """Token counts and <UNK> coverage for one encoded split.

    Word tokens exclude the <START>/<END> markers.
    """
    ids = np.concatenate(encoded_sentences)
    words = ids[(ids != START_ID) & (ids != END_ID)]
    n_unk = int((words == UNK_ID).sum())
    return {
        "sentences": len(encoded_sentences),
        "word_tokens": int(words.size),
        "unk_tokens": n_unk,
        "unk_rate": n_unk / words.size,
        "num_tokens": int((words == NUM_ID).sum()),
    }
