# Word-Level Autocomplete System

Final project for a Bayesian Inference course.

This project implements a word-level autocomplete system that predicts the next
word from a given text prefix. Four models are trained and evaluated on the same
WikiText-2 data under a common experimental protocol:

| Model | Context representation | Training |
|---|---|---|
| Bayesian bigram | Previous word | Dirichlet posterior from transition counts |
| HMM | Distribution over latent states | Baum-Welch (EM) |
| Vanilla RNN | Continuous recurrent hidden state | Adam, teacher forcing |
| GRU | Gated recurrent hidden state | Adam, teacher forcing |

The project compares how these different representations of context affect
next-word prediction.

## Dataset

The project uses the raw WikiText-2 corpus
(`Salesforce/wikitext`, `wikitext-2-raw-v1`) with the official training,
validation, and test splits.

The preprocessing pipeline:

- removes article titles and section headings,
- lowercases the text,
- splits the text into sentences,
- removes punctuation,
- replaces numeric expressions with `<NUM>`,
- constructs the vocabulary from the training split only,
- maps words occurring fewer than five times to `<UNK>`,
- adds `<START>`, `<END>`, and `<PAD>` special tokens.

The resulting vocabulary contains 20,316 tokens and is shared by all four
models.

## Repository structure

```text
word-level-autocomplete/
├── main.py                  # autocomplete command-line interface
├── run_experiments.py       # experimental pipeline
├── models/
│   ├── bigram.py
│   ├── hmm.py
│   └── recurrent.py
├── src/
│   ├── autocomplete.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train.py
│   └── vocabulary.py
├── tests/                   # pytest tests
├── data/
│   └── processed/           # processed WikiText-2 data
├── checkpoints/             # vocabulary and validation-selected models
├── pyproject.toml
└── uv.lock

```


## Installation

Requirements:

- Python 3.13
- [uv](https://docs.astral.sh/uv/)

Clone the repository or download the project files, then open a terminal in the project root directory.

Install the required dependencies with:

```bash
uv sync
```

The project dependencies are defined in `pyproject.toml`, and exact package versions are recorded in `uv.lock`.

## Running autocomplete

All project files are available in the GitHub repository:

https://github.com/Galilevin1/word-level-autocomplete

There are two ways to run the pretrained autocomplete models.

### Option 1: Use the submitted ZIP

Because of the submission ZIP file-size limit, the trained checkpoints may be omitted from the submitted ZIP.


To run autocomplete from the submitted ZIP without retraining:

1. Download `bigram.npz`, `hmm.npz`, `rnn.pt`, `gru.pt`, and `vocab.json` from the GitHub repository.
2. Place the files inside the local `checkpoints/` directory.
3. Install the dependencies using `uv sync`.
4. Run one of the autocomplete commands as following:

For example, to query the GRU:

```bash
uv run main.py --model gru --text "the game was" --top-k 5
```

To query all four models:

```bash
uv run main.py --model all --text "in the" --top-k 5
```

Available model choices are:

```text
bigram
hmm
rnn
gru
all
```

The pretrained files required for autocomplete are:

```text
checkpoints/
├── bigram.npz
├── hmm.npz
├── rnn.pt
├── gru.pt
└── vocab.json
```

During inference, only the current sentence is used as context, matching the training setup. Out-of-vocabulary input words are mapped to `<UNK>`. Suggested words exclude `<PAD>`, `<START>`, `<UNK>`, and `<END>`. The end-of-sentence probability is handled separately.

### Option 2: Use the complete GitHub repository

Clone or download the complete GitHub repository:

https://github.com/Galilevin1/word-level-autocomplete

Open a terminal in the repository root and install the dependencies:

```bash
uv sync
```

Run one of the autocomplete commands as following:

For example, to query the GRU:

```bash
uv run main.py --model gru --text "the game was" --top-k 5
```

To query all four models:

```bash
uv run main.py --model all --text "in the" --top-k 5
```

Available model choices are:

```text
bigram
hmm
rnn
gru
all
```

The pretrained files required for autocomplete are:

```text
checkpoints/
├── bigram.npz
├── hmm.npz
├── rnn.pt
├── gru.pt
└── vocab.json
```

During inference, only the current sentence is used as context, matching the training setup. Out-of-vocabulary input words are mapped to `<UNK>`. Suggested words exclude `<PAD>`, `<START>`, `<UNK>`, and `<END>`. The end-of-sentence probability is handled separately.


## Reproducing the experiments

The full experimental pipeline can be reproduced from the repository root.

First install the dependencies:

```bash
uv sync
```

Then run the following stages in order:

```bash
uv run run_experiments.py --stage data
uv run run_experiments.py --stage bigram
uv run run_experiments.py --stage hmm
uv run run_experiments.py --stage neural
uv run run_experiments.py --stage test
```

The stages perform the following tasks:

1. `data` - loads and preprocesses WikiText-2 and prepares the shared vocabulary.
2. `bigram` - trains and selects the Bayesian bigram model.
3. `hmm` - trains and selects the HMM.
4. `neural` - trains and selects the RNN and GRU models.
5. `test` - evaluates the validation-selected models on the held-out test split.

Model selection is based on validation NLL.

The main hyperparameter choices are:

- Bayesian bigram: Dirichlet concentration `alpha`
- HMM: number of hidden states `K`
- RNN: hidden size
- GRU: hidden size

The selected checkpoints are the same models used for the final reported test results.

## Running the tests

The project includes a pytest test suite.

Run all tests from the repository root with:

```bash
uv run pytest
```

## Results

The academic report contains the full mathematical formulation, model descriptions, experimental methodology, results, and discussion.