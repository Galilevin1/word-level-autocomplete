"""
Text preprocessing utilities for the Bayes autocomplete project.
"""

import re


def remove_gutenberg_header(text):
    """Remove Project Gutenberg header and footer metadata."""

    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

    if start_marker in text:
        text = text.split(start_marker, 1)[1]

        # Remove the remainder of the START marker line
        text = text.split("\n", 1)[1]

    if end_marker in text:
        text = text.split(end_marker, 1)[0]

    return text.strip()


def extract_story(text):
    """Keep the Alice story, starting from Chapter I."""

    start_marker = "I--DOWN THE RABBIT-HOLE"

    if start_marker in text:
        text = text.split(start_marker, 1)[1]
        text = start_marker + text

    return text.strip()


def tokenize_sentences(text):
    """Convert text into lowercase word-token sentences with boundary tokens."""

    # Remove illustration markers
    text = re.sub(
        r"\[illustration[^\]]*\]",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Lowercase
    text = text.lower()

    # Split into sentences
    raw_sentences = re.split(r"[.!?]+", text)

    sentences = []

    for sentence in raw_sentences:

        # Keep words and contractions
        words = re.findall(
            r"\b[a-z]+(?:'[a-z]+)?\b",
            sentence
        )

        if words:
            sentences.append(
                ["<START>"] + words + ["<END>"]
            )

    return sentences

# ---------------------------------------------------------------------------
# WikiText-2
# ---------------------------------------------------------------------------

NUM_TOKEN = "<NUM>"

# A number such as 2011, 3.14 or 1,000; any letters/digits attached to it
# (21st, 1990s, 3D, 100m) are absorbed so nothing is silently dropped.
_NUMBER = r"\d+(?:[.,]\d+)*[^\W_]*"
# Dotted acronyms stay one token: "U.S." -> "u.s.", "e.g." -> "e.g.".
_ACRONYM = r"(?:[^\W\d_]\.){2,}"
# A word is a run of Unicode letters with an optional apostrophe suffix
# ("game's", "don't"). Digits and underscores are excluded.
_WORD = r"[^\W\d_]+(?:'[^\W\d_]+)?"
TOKEN_RE = re.compile(
    rf"(?P<num>{_NUMBER})|(?P<word>{_ACRONYM}|{_WORD})")

# WikiText is pre-tokenized: sentence-final punctuation is a standalone
# token surrounded by spaces. "U.S." / "Dr." / "3.14" are attached to
# their word, so they are not treated as boundaries.
_SENTENCE_END_RE = re.compile(r"(?:^|\s)[.!?]+(?=\s|$)")


def clean_wikitext_record(text):
    """Clean a WikiText-2 record before sentence tokenization."""

    text = text.strip()

    # Skip empty records
    if not text:
        return ""

    # Skip article titles and section headings
    if re.fullmatch(r"=+.*=+", text):
        return ""

    # Rejoin WikiText split markup: "1 @.@ 5" -> "1.5", "1 @,@ 000" ->
    # "1,000", "role @-@ playing" -> "role-playing".
    text = re.sub(r"\s*@([-.,])@\s*", r"\1", text)

    # Restore common contraction spacing: "don 't" -> "don't"
    text = re.sub(r"\s+'(s|t|re|ve|ll|d|m)\b", r"'\1", text)

    return text


def tokenize_text(text):
    """Lowercase word tokens of one text span; numbers become <NUM>.

    Letters and digits are never dropped. Punctuation and symbols are
    discarded (word-level model), and hyphenated compounds are split
    into their parts ("role-playing" -> "role", "playing").
    """

    return [
        NUM_TOKEN if m.group("num") else m.group("word")
        for m in TOKEN_RE.finditer(text.lower())
    ]


def tokenize_wikitext_sentences(text):
    """Split a cleaned record into sentences (token lists, no markers)."""

    sentences = []

    for span in _SENTENCE_END_RE.split(text):
        words = tokenize_text(span)
        if words:
            sentences.append(words)

    return sentences


def split_prefix(text):
    """Tokenize free user text; return the tokens of its last sentence.

    Used for autocomplete: a prediction conditions on the sentence being
    typed. If the text ends with sentence-final punctuation, the new
    sentence is empty. Unlike WikiText, user punctuation may be attached
    to words ("was.").
    """

    text = re.sub(r"([.!?]+)(?=\s|$)", r" \1", text.strip())
    spans = _SENTENCE_END_RE.split(text)
    return tokenize_text(spans[-1]) if spans else []


def preprocess_wikitext(dataset_split):
    """Convert a WikiText split into a list of tokenized sentences.

    Boundary markers <START>/<END> are added at encoding time by
    Vocabulary.encode_sentence, so every model receives identical
    sequences.
    """

    sentences = []

    for record in dataset_split:
        text = clean_wikitext_record(record["text"])
        if text:
            sentences.extend(tokenize_wikitext_sentences(text))

    return sentences