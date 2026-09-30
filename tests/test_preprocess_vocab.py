"""Preprocessing and vocabulary tests (no dataset download needed)."""

import numpy as np
import pytest

from src.preprocess import (TOKEN_RE, clean_wikitext_record, split_prefix,
                            tokenize_text, tokenize_wikitext_sentences)
from src.vocabulary import (END_ID, NEVER_PREDICTED, NOT_SUGGESTED, NUM_ID,
                            PAD_ID, SPECIALS, START_ID, UNK_ID, Vocabulary)


def pipeline(raw):
    text = clean_wikitext_record(raw)
    return tokenize_wikitext_sentences(text) if text else []


# Inputs are in WikiText-2 raw format (space tokenized, @x@ markup).
@pytest.mark.parametrize("raw, expected", [
    (" Dr. Smith moved to the U.S. Army in 1990 . He stayed . ",
     [["dr", "smith", "moved", "to", "the", "u.s.", "army", "in", "<NUM>"],
      ["he", "stayed"]]),
    (" The value of pi is 3 @.@ 14 approximately . ",
     [["the", "value", "of", "pi", "is", "<NUM>", "approximately"]]),
    (" It weighs 3.14 kg . ", [["it", "weighs", "<NUM>", "kg"]]),
    (" About 1 @,@ 000 fans and 250 players attended in 2011 . ",
     [["about", "<NUM>", "fans", "and", "<NUM>", "players", "attended",
       "in", "<NUM>"]]),
    (" In the 1990s the 21st division lost 12 @.@ 5 % . ",
     [["in", "the", "<NUM>", "the", "<NUM>", "division", "lost", "<NUM>"]]),
    (" The café in Senjō served crème brûlée . ",
     [["the", "café", "in", "senjō", "served", "crème", "brûlée"]]),
    (" I don 't think it 's over , they 're sure . ",
     [["i", "don't", "think", "it's", "over", "they're", "sure"]]),
    (" It rained ! Did it stop ? Yes . ",
     [["it", "rained"], ["did", "it", "stop"], ["yes"]]),
    (" a tactical role @-@ playing game ( Japanese ) . ",
     [["a", "tactical", "role", "playing", "game", "japanese"]]),
    (" = = Gameplay = = \n", []),
    ("   ", []),
])
def test_wikitext_pipeline(raw, expected):
    assert pipeline(raw) == expected


@pytest.mark.parametrize("text, expected", [
    ("3D", ["<NUM>"]),
    ("MP3", ["mp", "<NUM>"]),
    ("F-16", ["f", "<NUM>"]),
    ("COVID-19", ["covid", "<NUM>"]),
    ("B2B", ["b", "<NUM>"]),
    ("4x4", ["<NUM>"]),
    ("A1 motorway", ["a", "<NUM>", "motorway"]),
    ("747-400", ["<NUM>", "<NUM>"]),
    ("50km/h", ["<NUM>", "h"]),
    ("£1.5bn", ["<NUM>"]),
    ("v2.0", ["v", "<NUM>"]),
])
def test_mixed_alphanumeric(text, expected):
    assert tokenize_text(text) == expected


MIXED = ["3D", "MP3 player", "F-16s and B-52s", "COVID-19", "x86_64",
         "4x4 50km/h £1.5bn v2.0", "Boeing 747-400ER", "2nd 3rd 21st 1990s",
         "Senjō 戦場のヴァルキュリア3", "O'Brien's U.S.-based e.g. p.m.",
         "H2O CO₂ km² ½", "R2-D2 and C-3PO"]


@pytest.mark.parametrize("text", MIXED)
def test_no_letter_or_digit_is_dropped(text):
    """Every alphanumeric character must fall inside some token match."""
    text = clean_wikitext_record(text).lower()
    covered = set()
    for m in TOKEN_RE.finditer(text):
        covered.update(range(m.start(), m.end()))
    dropped = [c for i, c in enumerate(text) if c.isalnum() and i not in covered]
    assert dropped == []


def test_split_prefix_uses_current_sentence():
    assert split_prefix("the game was") == ["the", "game", "was"]
    assert split_prefix("It rained. Then the") == ["then", "the"]
    assert split_prefix("It rained.") == []
    assert split_prefix("") == []
    assert split_prefix("In 2011 it") == ["in", "<NUM>", "it"]


TRAIN = [["the", "cat", "sat"], ["the", "dog", "sat"], ["a", "cat"]]


def test_special_token_ids():
    v = Vocabulary.build(TRAIN, min_count=1)
    assert v.itos[:5] == SPECIALS
    assert (PAD_ID, UNK_ID, START_ID, END_ID, NUM_ID) == (0, 1, 2, 3, 4)
    # <NUM> is a legitimate prediction target and suggestion
    assert NUM_ID not in NEVER_PREDICTED and NUM_ID not in NOT_SUGGESTED


def test_vocab_built_from_training_only_with_min_count():
    v = Vocabulary.build(TRAIN, min_count=2)
    assert set(v.itos[5:]) == {"the", "cat", "sat"}  # dog, a occur once
    v1 = Vocabulary.build(TRAIN, min_count=1)
    assert "dog" in v1.stoi
    # A word seen only in validation/test can never enter the vocabulary
    assert "zebra" not in v1.stoi
    assert v1.encode(["zebra"]).tolist() == [UNK_ID]


def test_vocab_ordering_is_deterministic():
    v = Vocabulary.build(TRAIN, min_count=1)
    # counts: the 2, cat 2, sat 2, a 1, dog 1 -> frequency then alphabetical
    assert v.itos[5:] == ["cat", "sat", "the", "a", "dog"]


def test_encode_decode_round_trip():
    v = Vocabulary.build(TRAIN, min_count=1)
    ids = v.encode_sentence(["the", "dog", "<NUM>", "zebra"])
    assert ids.dtype == np.int64
    assert ids[0] == START_ID and ids[-1] == END_ID
    assert ids[3] == NUM_ID and ids[4] == UNK_ID
    assert v.decode(ids) == ["<START>", "the", "dog", "<NUM>", "<UNK>", "<END>"]


def test_save_load(tmp_path):
    v = Vocabulary.build(TRAIN, min_count=2)
    v.save(tmp_path / "vocab.json")
    w = Vocabulary.load(tmp_path / "vocab.json")
    assert w.itos == v.itos and w.min_count == 2
