# CoCoNat

Python implementation of **CoCoNat: Training-Free Cross-Query Context
Conditioning for Named Entity Recognition**, with experiment and evaluation tools.

CoCoNat is an inference-time overlay for an already trained NER backbone. It detects
uncertain or inconsistent predicted mentions, forms small evidence groups among the
hard queries, applies the same backbone to each concatenated group, and combines the
span-wise second-pass scores. It does not update the backbone during CoCoNat inference.

The package provides a command-line interface, explicit token/span contracts,
validation-only model selection, measured result files, and tests. The included demo
is synthetic and its outputs are labelled as such.

## What is covered

The package supports the following experiments:

| Experiment | Command | Outputs |
| --- | --- | --- |
| Backbone vs. CoCoNat | `main` | exact entity P/R/F1, predictions, latency |
| Recent NER paradigms | `baselines` | accuracy, timing scope, provenance, coverage gaps |
| Dataset characteristics | `main` | hard-span rate, repeated-hard rate, F1 headroom |
| Error propagation | `main` | W-to-C, C-to-C, W-to-W, C-to-W and new-only spans |
| Detector/grouping/order/fusion ablations | `ablation` | CSV plus settings per variant |
| Kappa/delta sensitivity | `sensitivity` | sweep CSV and publication-ready PDF/PNG plot |
| Validation tuning and fixed transfer | `tune,main` | full validation grid and fixed `(9, 0.8)` comparison |
| Temperature calibration | `calibration` | validation-fitted temperature, ECE, F1, hard-set Jaccard |
| Length and label controls | `main` / `merge` | natural and label-balanced strata |
| Manual C-to-W analysis | `audit-export`, `audit-summarize` | unannotated sample and human-coded summary |

## Installation

Python 3.10 or newer is required. Create an isolated environment and install the package
from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[hf,analysis]"
```

Install PyTorch using the command recommended for the local CUDA version before the
last command when GPU acceleration is required. Optional native adapters are separate
extras because their dependency constraints can conflict:

```bash
python -m pip install -e ".[gliner]"   # or .[flair], .[spacy]
```

Inspect the environment without downloading a model:

```bash
coconat doctor
```

## Verify the installation

The fastest end-to-end check runs every analysis on generated data with a deterministic
toy backend:

```bash
coconat demo --output outputs/demo
```

Run the complete unit and local tiny-Transformer tests with:

```bash
python -m unittest discover -s tests -v
ruff check src tests
```

The Transformer tests construct random tiny BERT and Longformer checkpoints locally.

## Data contract

Local CoNLL input is whitespace-separated, with one token per line, a BIO/BIOES/BILOU
tag column, and a blank line between queries. `-DOCSTART-` creates document boundaries.
Set `--token-column` and `--tag-column` when the source has multiple columns.

Normalize a dataset and generate a runnable configuration:

```bash
coconat prepare \
  --name conll2003 \
  --format conll \
  --train raw/conll2003/train.txt \
  --validation raw/conll2003/valid.txt \
  --test raw/conll2003/test.txt \
  --labels PER,ORG,LOC,MISC \
  --output data/conll2003
```

Omit `--labels` only when every official type occurs in the training split. Declaring the
official schema is safer. The normalized JSONL form uses half-open **word** offsets:

```json
{"id":"q1","tokens":["Acme","opened","in","Paris","."],"doc_id":"d1","spans":[{"start":0,"end":1,"label":"ORG"},{"start":3,"end":4,"label":"LOC"}]}
```

Gold labels remain in evaluation files, but inference methods receive immutable `Query`
objects containing only ID, tokens, and optional document ID. Exports for external
systems omit gold by default.

## Train a backbone

Edit the generated `data/conll2003/experiment.yaml`, then run:

```bash
coconat train --config data/conll2003/experiment.yaml
```

Training uses only the configured train and validation splits. The saved checkpoint has
an explicit BIO head and a `training_manifest.json` containing split fingerprints,
versions, and selection details. CoCoNat itself remains training-free; this command is
only for producing the supervised backbone used by the paper.

Longformer document-context training is enabled with:

```yaml
training:
  base_checkpoint: allenai/longformer-base-4096
  context_mode: document
model:
  kind: hf
  checkpoint: ../../checkpoints/conll2003/longformer-base-4096
  max_length: 4096
```

Every query must have a `doc_id` in document mode. Whole queries are packed by document
without crossing documents; over-length inputs are split only at complete word
boundaries.

## Run all CoCoNat analyses

```bash
coconat run --config data/conll2003/experiment.yaml \
  --tasks tune,main,ablation,sensitivity,calibration
```

`tune` searches only the validation split. Its selected `(kappa, delta)` is then used on
test. Each invocation creates a new timestamped run directory, so prior measurements are
never overwritten. To execute all eight dataset configurations sequentially:

```bash
coconat matrix \
  --configs data/*/experiment.yaml \
  --tasks tune,main,ablation,sensitivity,calibration
```

Merge completed dataset runs after checking that the paths are the intended runs:

```bash
coconat merge --runs outputs/conll/RUN outputs/ontonotes/RUN outputs/wnut/RUN \
  outputs/fin/RUN outputs/bionlp/RUN outputs/bc5cdr/RUN \
  outputs/mit_movie/RUN outputs/mit_restaurant/RUN \
  --output outputs/merged
```

The merge creates dataset-characteristic, fixed-setting, calibration, baseline, pooled
length, natural-label, and label-balanced summaries.

## Recent baselines

There are three integration paths:

1. Native resident adapters for Hugging Face token classifiers, GLiNER, Flair, and a
   trained spaCy SpanCategorizer.
2. A generic local or hosted instruction-LLM adapter using the same indexed-word JSON
   prompt. Hosted calls require both an environment variable and `--allow-api`.
3. A strict external-command or prediction-import bridge for official ScdNER,
   PromptNER, GPT-NER, IRRA, and ReasoningNER implementations.

The third path is deliberate: these systems have distinct training/retrieval/reasoning
procedures, so relabelling a generic prompt run as an official-method result would be
invalid. Copy the relevant entries from
[configs/recent_baselines.example.yaml](configs/recent_baselines.example.yaml), provide a
concrete source manifest, and enable only integrations that have been installed and
verified. See [docs/BASELINES.md](docs/BASELINES.md) for the prediction protocol.

Run baseline entries with:

```bash
coconat run --config data/conll2003/experiment.yaml --tasks baselines
```

Use `--require-all` in the final reproduction run. It fails if a required method is
disabled or missing instead of silently producing an incomplete table.

Hosted inference is intentionally opt-in:

```bash
export OPENAI_API_KEY="..."
coconat run --config data/conll2003/experiment.yaml \
  --tasks baselines --allow-api --require-all
```

## External baselines

Export token-stable, gold-blind inputs:

```bash
coconat export --config data/conll2003/experiment.yaml \
  --split test --output external/conll2003-input.jsonl
```

Convert an official system's BIO, word-offset, character-offset, or generative output:

```bash
coconat convert-predictions --config data/conll2003/experiment.yaml \
  --input external/raw-predictions.jsonl --format bio \
  --output external/predictions.jsonl
```

## Manual error audit

After all eight main runs, pool the actual C-to-W cases and draw the manuscript sample:

```bash
coconat audit-export \
  --cases outputs/*/RUN/main/transition_cases.jsonl \
  --sample-size 100 --seed 42 --output outputs/manual-c-to-w.csv
```
