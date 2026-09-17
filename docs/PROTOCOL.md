# Experimental protocol

This document fixes the definitions used by the code. Change it only before running the
final experiment matrix; otherwise results from different configurations should not be
pooled.

## Units and coordinates

- A query is one released test item in its released order.
- A span is represented by zero-based, half-open word offsets `[start, end)` and one
  entity type. Byte, character, and subword offsets are never used for scoring.
- The main task is flat NER. A prediction is correct only if start, end, and type all
  match one gold span exactly.
- Micro precision, recall, and F1 are computed over entity spans. Token accuracy and
  partial-overlap credit are not reported.
- Each dataset is a separate query collection. Evidence groups never cross datasets.

The complete test split is available to the grouping algorithm as unlabeled input.
Detection and grouping see test tokens and first-pass predictions, but not test gold.
The immutable `Query` inference object has no gold field. Gold is introduced only by
metric and diagnostic functions after prediction.

## First-pass span confidence

For BIO token classifiers, every word is represented by its first subtoken. A decoded
multiword entity's confidence is the arithmetic mean of the predicted BIO tag
probabilities of its words. This is the `p(s)` value used by the detector. The complete
first-subtoken logit matrix is retained for temperature scaling and second-pass
per-label span scores.

Native span models use their emitted span confidence. Imported predictions without a
score remain valid for exact P/R/F1, but ECE is reported as unavailable rather than
assuming confidence 1.0 is calibrated.

## Low-confidence screening

Let `P` be the multiset of confidences of every first-pass predicted entity in the query
collection. The population mean and standard deviation are used:

```text
threshold = mean(P) - kappa * std(P, ddof=0)
```

A predicted span is low-confidence when its score is at or below this threshold. If the
batch contains no predicted entities, the threshold is undefined. If the standard
deviation is numerically zero, no span is flagged because the batch supplies no relative
confidence information.

## Inconsistency screening

Surface strings are constructed by joining the original dataset tokens of a predicted
span with one ASCII space. Matching is case-sensitive by default. For one surface form,
let `c(1) >= c(2) >= ...` be its predicted-label frequencies. With at least two distinct
labels,

```text
R(s) = c(1) / (c(1) + c(2)).
```

The span is inconsistent when `R(s) <= delta`, where `delta` is in `[0.5, 1]`. This is
the normalized top-two statistic from the paper, not the entropy of the full label
distribution. The entropy/odds equivalence in Equation (6) requires `delta < 1`; the
`delta = 1` endpoint remains in the empirical sweep and simply admits every surface form
with at least two predicted labels.

A query is hard if at least one of its first-pass predicted spans activates an enabled
trigger. `detector: both` is their logical union. Queries with no predicted spans cannot
become hard under this detector.

## Evidence groups and token budgets

The main configuration uses exact hard-surface matching. Only hard queries participate.
A group is created for each hard surface form, and one query may belong to several
overlapping groups. Singleton groups are discarded and fall back to the first pass.

The K-means and diagonal-GMM ablations cluster the hard-query embeddings produced by the
same backbone. The requested cluster count defaults to the number of hard forms and is
capped by the number of hard queries and distinct embedding vectors. GMM membership is
multi-label when a posterior meets `gmm_threshold`; if none does, argmax membership is
used. For ordering, an embedding group's representative is the hard mention whose
backbone vector has greatest cosine similarity to the group centroid.

Groups are first divided deterministically to satisfy `max_group_size`. The model's
native separator token is inserted between queries. If a resulting group still exceeds
the token budget, the highest-confidence anchor is retained and complete member queries
are added greedily in the selected order. Tokens and partial queries are never silently
truncated. An omitted query keeps its first-pass output unless another overlapping group
successfully processes it. Every inclusion, omission, and reason is recorded in
`main/groups.json` or the relevant ablation settings.

## Gradual order and random comparison

One highest-confidence representative is selected per first-pass label. The two strongest
conflicting anchors occupy opposite ends; additional label anchors are spread through the
remaining positions. Lower-confidence representatives are placed next to their own
anchor with deterministic tie-breaking by query ID.

The random-order ablation uses five consecutive, recorded seeds by default. Its report
contains each run and the mean/sample standard deviation. It is not a single favorable
random draw.

## Second pass and aggregation

The unchanged backbone tags each concatenated group. Separator predictions are removed,
and word-level logits/spans are projected back to the original query. A query belonging
to one successful group inherits that group output. For multiple group contexts, each
candidate boundary is evaluated separately for that query occurrence:

- `MAX` takes the largest per-label path score across contexts.
- `MEAN` averages each label's path score over **all** contexts, including contexts that
  favor `O`, and selects the largest mean.

For a BIO classifier, a candidate type's score is the mean probability of its B tag at
the first word and its I tag at subsequent words. The O score is the mean O probability.
For native span systems, an absent candidate contributes O mass rather than disappearing
from the MEAN denominator. The final flat output resolves overlapping candidates by
confidence with deterministic ties. Scores are never pooled between distinct occurrences
of the same surface form; group membership does not force a uniform label.

Non-hard queries are copied byte-for-byte at the span-contract level from the first pass.

## Hyperparameter selection and sensitivity

The default revision grid is:

```text
kappa: 1, 2, ..., 12
delta: 0.5, 0.6, ..., 1.0
```

The Cartesian grid is evaluated on validation only. Highest validation F1 wins. The
predeclared tie break is fewer processed queries, then larger kappa, then smaller delta.
The selected pair is applied once to test. The fixed-transfer analysis separately uses
`(kappa, delta) = (9, 0.8)` on every dataset and records whether the comparison's selected
setting actually came from validation.

The sensitivity task holds one selected parameter fixed while sweeping the other. It
records both exact entity F1 and the number of hard queries, then renders a four-panel
PDF/PNG figure.

## Temperature and ECE

A single positive temperature is fitted on validation by minimizing first-subtoken
word-level negative log likelihood. Optimization occurs in log-temperature space with a
fixed bounded scalar search; temperature 1 is explicitly included as a candidate. Test
gold is never passed to the fit function.

The calibrated logits preserve each word's argmax tag but may change cross-span confidence
rankings. The report therefore contains:

- 15-bin equal-frequency ECE over **predicted entities**;
- raw and calibrated final test F1;
- Jaccard similarity of the raw and calibrated hard-span sets; and
- a check that the first-pass predicted span set was unchanged.

ECE bins are formed after sorting predicted entities by confidence. Empty predicted sets
or backends without real confidence receive a null result and a reason.

## Dataset characteristics and strata

`hard_fraction` is hard first-pass predicted spans divided by all first-pass predicted
spans. `repeated_hard_fraction` is hard predicted spans whose exact surface occurs in at
least two test queries, divided by all hard predicted spans. `error_headroom` is
`1 - first_pass_F1`; it is descriptive headroom, not an error probability.

Length bins are `<=15`, `16-25`, `26-40`, and `>40` words, represented as `01-15`,
`16-25`, `26-40`, and `41+` in CSV for stable sorting. The trigger diagnostic reports
mean first-pass entity confidence, the fraction selected by the low-confidence trigger,
and exact error among selected predicted spans.

Natural label strata retain their observed frequency. The frequency-control analysis
also downsamples each `(dataset, label)` group to the smallest group size with a recorded
seed. The two analyses must be reported separately.

## First-to-second-pass transitions

Here **W means wrong** and **C means correct**. The primary table's denominator is every
first-pass predicted span in a query that was actually reprocessed. Each first-pass span
is paired with its deterministic maximum-overlap second-pass span in the same query;
pairing uses only boundaries and never gold correctness. This follows the paper literally
and does not require second-pass partners to be unique.

- `W_C`: an incorrect first-pass span is paired with a correct second-pass span.
- `C_C`: a correct first-pass span remains paired with a correct span.
- `W_W`: an incorrect first-pass span remains incorrect or has no second-pass partner.
- `C_W`: a correct first-pass span becomes incorrect or has no second-pass partner.

Second-pass spans that are not the selected partner of any first-pass span are reported as
`new_only_second_spans`; hiding them in W-to-C/W-to-W would change the denominator.
For transparency, the output also includes a stricter one-to-one greedy sensitivity
diagnostic and gold-anchored counts. Neither transition table replaces exact entity P/R/F1.

## Human audit

The manual audit pools only real C-to-W cases, samples at most 100 with a fixed seed, and
leaves `reason` blank. Allowed categories are `homonymous_surface`, `weak_evidence`,
`token_budget`, `boundary_change`, and `other`. The summarizer reports annotated and
unannotated counts and marks the audit complete only when every nonempty sample has a
valid human label. No automated heuristic is presented as manual analysis.

## Timing

The default is one warm-up followed by three measured passes. CUDA synchronization occurs
at measurement boundaries. CoCoNat's component timing includes first pass, detection and
grouping, contextual second pass, and aggregation. Model loading and result serialization
are reported outside the resident inference measurement.

Hosted timing includes request serialization, network time, response parsing, and API
service time. An external command includes process startup and model loading. Imported
predictions have no latency. Relative latency is populated only when both compared values
were measured, and the output always retains timing scope and hardware provenance.
