# Reproducibility status

## What the automated checks establish

The test suite verifies deterministic schema conversion, hard-query detection, top-two
inconsistency behavior, all four error transitions, temperature fitting, ECE binning,
group fallbacks, MAX/MEAN aggregation, data leakage guards, external-output validation,
API opt-in, provenance validation, and manual-audit behavior.

When the `hf` extra is installed, local CPU integration tests additionally construct
tiny random BERT and Longformer models and verify:

- first-subtoken word/logit alignment;
- native separator removal and original-query offsets;
- complete-word chunking of an 80-word input;
- group budget rejection rather than truncation;
- real hidden-state query and mention embeddings;
- special/subword masking in training targets;
- one-epoch training, checkpoint saving, reloading, and prediction;
- document-context packing; and
- Longformer global-attention inference.

GitHub Actions runs offline tests on Python 3.10 and 3.12 and a separate CPU
tiny-Transformer job. The synthetic demo runs every experiment/reporting branch.

## What the checks do not establish

Tiny random models test software paths, not scientific quality. The release does not
contain the eight licensed datasets, trained benchmark checkpoints, official third-party
repositories, retrieval indices, hosted API credentials, an A100/RTX measurement, or
human annotations. Consequently, no placeholder result should be cited as a reproduced
paper result.

A complete scientific reproduction requires the exact released splits and schemas,
frozen checkpoint/source revisions, the stated hardware, enabled required baselines, and
a completed manual audit. Each real run records enough information to diagnose gaps but
cannot manufacture unavailable external assets.

## Determinism boundaries

Python/NumPy/Transformers seeds are recorded. K-means and GMM receive the configured
seed; random-order ablations record each seed. CUDA kernels and hosted services may still
have platform-level nondeterminism. Report all three timing samples rather than only the
last one. External APIs may change serving infrastructure even for a named model
snapshot, so preserve response metadata and the run date.

## Result integrity

- Draft table numbers never enter computation.
- Missing measurements remain absent or null.
- Synthetic output is visibly tagged and rejected by normal cross-run merge.
- A source manifest with an unedited placeholder is rejected.
- Configurations and results redact credential-like keys.
- Split fingerprints detect accidental reuse of identical validation/test inputs.
- Final baseline coverage is enforceable with `--require-all`.
