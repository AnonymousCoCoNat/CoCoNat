# Revision experiment guide

## One-dataset end-to-end sequence

```bash
# 1. Normalize released splits and create the initial configuration.
coconat prepare --name DATASET --format conll \
  --train raw/DATASET/train.txt --validation raw/DATASET/valid.txt \
  --test raw/DATASET/test.txt --labels TYPE1,TYPE2 \
  --output data/DATASET

# 2. Review paths, official label schema, and training settings.
coconat doctor --config data/DATASET/experiment.yaml

# 3. Fine-tune the supervised BIO backbone.
coconat train --config data/DATASET/experiment.yaml

# 4. Run every CoCoNat revision analysis that uses this backbone.
coconat run --config data/DATASET/experiment.yaml \
  --tasks tune,main,ablation,sensitivity,calibration

# 5. Enable verified recent baselines and run them independently.
coconat run --config data/DATASET/experiment.yaml \
  --tasks baselines --require-all
```

The eight paper datasets are CoNLL-2003, OntoNotes 5.0, WNUT-2017, FIN,
BioNLP-2004, BC5CDR, MIT Movie, and MIT Restaurant. Do not infer that similarly named
online mirrors have the paper's released split or label mapping. Normalize the exact
copies used for the study and retain their checksums outside this repository.

## Output map

| Location | Contents | Revision use |
| --- | --- | --- |
| `manifest.json` | completion, environment, model/source identity, split hashes | provenance and completeness |
| `resolved_config.json` | credential-redacted settings | methods appendix |
| `selected_pipeline.json` | selected detector settings and source | Table 11 |
| `tuning/validation_grid.csv` | complete validation Cartesian grid | selection audit |
| `main/metrics.json` | backbone and CoCoNat exact entity scores | Table 3 |
| `main/dataset_characteristics.json` | hard/repeated/headroom statistics | Table 5 |
| `main/transitions.json` | four transitions plus alternative diagnostics | Table 6 |
| `main/transition_cases.jsonl` | row-level aligned cases | manual audit |
| `main/length_strata.csv` | four length bins | Table 13 |
| `main/label_strata.csv` | natural label distribution | label analysis |
| `main/label_balanced_strata.csv` | within-dataset equal-size label control | frequency control |
| `main/groups.json` | requested/included/omitted members and reasons | error analysis |
| `main/clustering.json` | embedding and effective cluster details | Table 8 |
| `main/latency.json` | warm-ups, repeats, component timing | Tables 3/4 |
| `ablation/ablations.csv` | detector, grouping, random ordering, MAX/MEAN | Tables 7-10 |
| `sensitivity/sensitivity.csv` | kappa/delta F1 and hard counts | Figure 4 |
| `calibration/calibration.json` | T, ECE, F1, overlap | Table 12 |
| `baselines/comparison.csv` | enabled recent systems | Table 4 input |
| `report/*.tex` | generated fragments from that run only | manuscript insertion |
| `report/sensitivity.{pdf,png}` | generated sweep figure | visual inspection/insertion |

All generated LaTeX tables contain decimal values computed from recorded outputs. Synthetic
tables carry an explicit comment and visible warning. The reporting layer never fills a
missing method with the draft's assumed value.

## Eight-dataset matrix

Once all configurations point to trained checkpoints:

```bash
coconat matrix --configs \
  data/conll2003/experiment.yaml \
  data/ontonotes5/experiment.yaml \
  data/wnut2017/experiment.yaml \
  data/fin/experiment.yaml \
  data/bionlp2004/experiment.yaml \
  data/bc5cdr/experiment.yaml \
  data/mit_movie/experiment.yaml \
  data/mit_restaurant/experiment.yaml \
  --tasks tune,main,ablation,sensitivity,calibration
```

The command is intentionally sequential. This prevents several large checkpoints from
competing for one GPU and makes failure attribution clear. Each configuration may use a
different output parent.

After the runs finish, use their exact timestamped paths with `coconat merge`. The merge
refuses a failed manifest and, by default, a synthetic run. The resulting CSV files are
the cross-dataset source of truth; review them before copying generated LaTeX.

## Backbone matrix for the original comparison

Create one configuration per `(dataset, backbone checkpoint)` and give each an independent
output directory. The supported resident adapters are:

- `hf` for DistilBERT, BERT, DeBERTa, XLM-R, Longformer, or another token classifier with
  an exact dataset BIO head;
- `flair` for a Flair `SequenceTagger`;
- `spacy` for a trained SpanCategorizer that exposes span scores; and
- `gliner` for GLiNER zero-shot transfer with an explicit semantic label map.

Do not point `hf` at a generic language-model checkpoint at inference time. The adapter
verifies the full BIO head and fails rather than treating `LABEL_0` as O.

## Recommended final-run checks

1. Confirm `validation_sha256` and `test_sha256` differ.
2. Confirm `synthetic` is false in every manifest.
3. Confirm every required baseline has `status: completed`.
4. Confirm resolved commits/checkpoints and hardware match the experimental setting.
5. Compare timing only within compatible `timing_scope` values.
6. Inspect all `invalid_offsets`, parse failures, omitted group members, and null ECEs.
7. Complete and archive the 100-case human audit.
8. Re-run from a clean checkout using the recorded commands and commit.
