"""All numerical revision analyses operate on model outputs, never draft table values."""

from __future__ import annotations

from dataclasses import asdict, replace
import itertools
from pathlib import Path
import time

import numpy as np

from .calibration import fit_temperature
from .io import save_predictions, write_csv, write_json, write_jsonl
from .metrics import (dataset_characteristics, entity_ece, entity_metrics, jaccard,
                      label_balanced_strata, screening_observations, strata_summary, transitions)
from .pipeline import run_pipeline


def compact_scores(examples, result):
    first, final = entity_metrics(examples, result.first), entity_metrics(examples, result.final)
    return dict(first_f1=first["f1"], precision=final["precision"], recall=final["recall"],
                f1=final["f1"], gain=final["f1"] - first["f1"],
                hard_queries=len(result.detection.hard_queries), hard_spans=len(result.detection.hard_spans),
                processed_queries=len(result.processed), groups=sum(bool(g["included"]) for g in result.groups))


def tune(examples, backend, pipeline, output, kappas, deltas, split="validation"):
    if split != "validation":
        raise ValueError("Hyperparameter selection is restricted to validation, never test")
    first = backend.predict([e.query for e in examples])
    rows = []
    for kappa, delta in itertools.product(kappas, deltas):
        candidate = replace(pipeline, kappa=float(kappa), delta=float(delta))
        result = run_pipeline([e.query for e in examples], backend, candidate, first=first)
        row = dict(kappa=kappa, delta=delta, **compact_scores(examples, result))
        rows.append(row)
        write_csv(output / "validation_grid.csv", rows)
        print(f"Validation kappa={kappa:g} delta={delta:g} F1={100 * row['f1']:.3f}", flush=True)
    if not rows:
        raise ValueError("The validation grid must not be empty")
    # Predeclared conservative tie break: less intervention, larger kappa, smaller delta.
    best = min(rows, key=lambda r: (-r["f1"], r["processed_queries"], -r["kappa"], r["delta"]))
    selected = replace(pipeline, kappa=float(best["kappa"]), delta=float(best["delta"]))
    write_json(output / "selection.json", dict(fit_split=split, selected=asdict(selected),
                                               validation_f1=best["f1"],
                                               tie_break="fewer_processed_then_larger_kappa_then_smaller_delta"))
    return selected, first


def benchmark_pipeline(examples, backend, pipeline, repeats=3, warmups=1):
    if repeats < 1 or warmups < 0:
        raise ValueError("repeats >= 1 and warmups >= 0 are required")
    queries = [e.query for e in examples]
    for _ in range(warmups):
        run_pipeline(queries, backend, pipeline)
    results, timings = [], []
    for _ in range(repeats):
        result = run_pipeline(queries, backend, pipeline)
        timings.append(result.timing)
        results.append(compact_scores(examples, result))
    rows = []
    for key in ("first_seconds", "grouping_seconds", "second_seconds", "aggregation_seconds", "total_seconds"):
        values = [t[key] for t in timings]
        rows.append(dict(component=key, mean_seconds=float(np.mean(values)),
                         std_seconds=float(np.std(values, ddof=1)) if repeats > 1 else 0.0,
                         seconds_per_1000=float(np.mean(values)) * 1000 / len(examples)))
    return result, dict(repeats=repeats, warmups=warmups, measured_timings=timings,
                        components=rows, run_scores=results, includes_model_loading=False,
                        includes_result_serialization=False, query_count=len(examples))


def analyze_main(examples, result, output, dataset_name, pipeline):
    output = Path(output)
    first = entity_metrics(examples, result.first)
    final = entity_metrics(examples, result.final)
    table, cases = transitions(examples, result)
    characteristics = dataset_characteristics(examples, result, pipeline.case_sensitive)
    observations = screening_observations(examples, result, dataset_name)
    write_json(output / "metrics.json", dict(backbone=first, coconat=final))
    write_json(output / "dataset_characteristics.json", characteristics)
    write_json(output / "transitions.json", table)
    write_jsonl(output / "transition_cases.jsonl", cases)
    write_jsonl(output / "screening_observations.jsonl", observations)
    write_csv(output / "length_strata.csv", strata_summary(observations, "length_bin"))
    write_csv(output / "label_strata.csv", strata_summary(observations, "label"))
    write_csv(output / "label_balanced_strata.csv", label_balanced_strata(observations, pipeline.seed))
    write_json(output / "groups.json", result.groups)
    write_json(output / "detection.json", result.detection)
    write_json(output / "clustering.json", result.cluster_info)
    save_predictions(output / "first_predictions.jsonl", examples, result.first)
    save_predictions(output / "final_predictions.jsonl", examples, result.final)
    write_json(output / "first_entity_ece.json", entity_ece(examples, result.first))
    return dict(metrics=dict(backbone=first, coconat=final), characteristics=characteristics, transitions=table)


def ablations(examples, backend, pipeline, first, output, random_repeats=5, include_embedding=True):
    variants = [
        ("detector_none", replace(pipeline, detector="none")),
        ("detector_confidence", replace(pipeline, detector="confidence")),
        ("detector_inconsistency", replace(pipeline, detector="inconsistency")),
        ("full", pipeline),
        ("aggregation_max", replace(pipeline, aggregation="max")),
    ]
    if include_embedding:
        variants.extend((name, replace(pipeline, grouping=name)) for name in ("kmeans", "gmm"))
    variants.extend((f"random_{seed}", replace(pipeline, ordering="random", seed=seed))
                    for seed in range(pipeline.seed, pipeline.seed + random_repeats))
    rows = []
    for name, candidate in variants:
        result = run_pipeline([e.query for e in examples], backend, candidate, first=first)
        transition, _ = transitions(examples, result)
        row = dict(variant=name, **compact_scores(examples, result),
                   C_W=transition["counts"]["C_W"], W_C=transition["counts"]["W_C"],
                   transition_denominator=transition["n"],
                   C_W_fraction=transition["fractions"]["C_W"])
        rows.append(row)
        write_csv(output / "ablations.csv", rows)
        write_json(output / f"{name}_settings.json", dict(pipeline=asdict(candidate),
                                                         clustering=result.cluster_info))
        print(f"Ablation {name}: F1={row['f1'] * 100:.3f}", flush=True)
    random_rows = [r for r in rows if r["variant"].startswith("random_")]
    write_json(output / "random_order_summary.json", dict(
        runs=len(random_rows), f1_mean=float(np.mean([r["f1"] for r in random_rows])) if random_rows else None,
        f1_std=float(np.std([r["f1"] for r in random_rows], ddof=1)) if len(random_rows) > 1 else None,
        corruption_fraction_mean=float(np.mean([r["C_W_fraction"] for r in random_rows
                                                if r["C_W_fraction"] is not None]))
        if any(r["C_W_fraction"] is not None for r in random_rows) else None))
    return rows


def sensitivity(examples, backend, pipeline, first, output, kappas, deltas):
    rows = []
    for parameter, values in (("kappa", kappas), ("delta", deltas)):
        for value in values:
            candidate = replace(pipeline, **{parameter: float(value)})
            result = run_pipeline([e.query for e in examples], backend, candidate, first=first)
            rows.append(dict(parameter=parameter, value=value, kappa=candidate.kappa,
                             delta=candidate.delta, **compact_scores(examples, result)))
            write_csv(output / "sensitivity.csv", rows)
    return rows


def calibration_experiment(validation, examples, backend, pipeline, first,
                           output, validation_first=None):
    validation_first = validation_first or backend.predict([e.query for e in validation])
    fit = fit_temperature(validation, validation_first)
    raw = run_pipeline([e.query for e in examples], backend, pipeline, first=first)
    calibrated = run_pipeline([e.query for e in examples], backend, pipeline, first=first,
                              temperature=fit["temperature"])
    report = dict(**fit, raw_ece=entity_ece(examples, raw.first),
                  calibrated_ece=entity_ece(examples, calibrated.first),
                  raw_final_f1=entity_metrics(examples, raw.final)["f1"],
                  calibrated_final_f1=entity_metrics(examples, calibrated.final)["f1"],
                  hard_span_jaccard=jaccard(raw.detection.hard_spans, calibrated.detection.hard_spans),
                  first_span_set_unchanged=all(
                      {s.key() for s in raw.first[qid].spans} == {s.key() for s in calibrated.first[qid].spans}
                      for qid in first))
    write_json(output / "calibration.json", report)
    return report


def benchmark_backend(examples, backend, repeats=3, warmups=1, document_mode=False):
    if repeats < 1 or warmups < 0:
        raise ValueError("repeats >= 1 and warmups >= 0 are required")
    queries = [e.query for e in examples]

    def predict():
        if not document_mode:
            return backend.predict(queries)
        from collections import defaultdict
        documents = defaultdict(list)
        for q in queries:
            if q.doc_id is None:
                raise ValueError("Long-context document evaluation requires explicit doc_id")
            documents[q.doc_id].append(q)
        output = {}
        for doc in documents.values():
            current = []

            def flush():
                if current:
                    output.update(backend.predict_groups([current])[0] if len(current) > 1
                                  else backend.predict(current))
                    current.clear()

            for q in doc:
                if not backend.group_fits([q]):
                    flush()
                    output.update(backend.predict([q]))
                else:
                    if current and not backend.group_fits(current + [q]):
                        flush()
                    current.append(q)
            flush()
        return output

    imported = backend.timing_scope.startswith("imported_predictions")
    if imported:
        return predict(), dict(seconds_per_1000=None, timing_scope=backend.timing_scope,
                               reason="original_inference_was_not_timed_here")
    for _ in range(warmups):
        predict()
    elapsed, scores = [], []
    for _ in range(repeats):
        backend.synchronize()
        started = time.perf_counter()
        predictions = predict()
        backend.synchronize()
        elapsed.append(time.perf_counter() - started)
        scores.append(entity_metrics(examples, predictions)["f1"])
    return predictions, dict(seconds_per_1000=float(np.mean(elapsed)) * 1000 / len(queries),
                              measured_seconds=elapsed, repeats=repeats, warmups=warmups,
                              f1_by_repeat=scores, timing_scope=backend.timing_scope,
                              document_context=document_mode)
