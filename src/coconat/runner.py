"""Reproducible suite orchestration, including explicit baseline coverage status."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid

from .audit import export_audit
from .backends import build_backend
from .config import PipelineConfig
from .data import load_split
from .experiments import (ablations, analyze_main, benchmark_backend, benchmark_pipeline,
                          calibration_experiment, compact_scores, sensitivity, tune)
from .io import (environment_info, labeled_fingerprint, query_fingerprint, redact_config,
                 save_predictions, write_csv, write_json)
from .metrics import entity_metrics
from .pipeline import run_pipeline


ALL_TASKS = ("tune", "main", "ablation", "sensitivity", "calibration", "baselines")


def _new_run_dir(base):
    path = Path(base) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_suite(config, tasks=ALL_TASKS, output=None, allow_api=False, require_all=False):
    tasks = list(dict.fromkeys(tasks))
    unknown = set(tasks) - set(ALL_TASKS)
    if unknown:
        raise ValueError(f"Unknown tasks: {sorted(unknown)}")
    if require_all and "baselines" in tasks:
        incomplete = [b.get("name", b.get("checkpoint", b["kind"])) for b in config.get("baselines", [])
                      if not b.get("enabled", True)]
        configured = {b.get("name", b.get("checkpoint", b["kind"])) for b in config.get("baselines", [])}
        incomplete += sorted(set(config.get("required_baselines", [])) - configured)
        if incomplete or not config.get("baselines"):
            raise ValueError(f"--require-all rejects disabled/missing baselines: {incomplete}")
    root = _new_run_dir(output or config["output_dir"])
    write_json(root / "resolved_config.json", redact_config(config))
    manifest = dict(status="running", requested_tasks=tasks, environment=environment_info(),
                    completed_tasks=[], baseline_coverage=[])
    write_json(root / "manifest.json", manifest)
    print(f"Run directory: {root}", flush=True)
    summary = {}
    try:
        examples = load_split(config["dataset"], "test")
        validation = load_split(config["dataset"], "validation") if set(tasks) & {"tune", "calibration"} else []
        if validation and query_fingerprint(validation) == query_fingerprint(examples):
            raise ValueError("Test and validation inputs are identical; refusing a leakage-prone run")
        pipeline = PipelineConfig.from_dict(config.get("pipeline", {}))
        experiment = config.get("experiments", {})
        model_started = time.perf_counter()
        backend = build_backend(config["model"], config["dataset"]["labels"], allow_api=allow_api) if (
            set(tasks) - {"baselines"}) else None
        manifest.update(dataset_name=config["dataset"].get("name", "dataset"),
                        query_sha256=query_fingerprint(examples), test_sha256=labeled_fingerprint(examples),
                        validation_sha256=labeled_fingerprint(validation) if validation else None,
                        synthetic=backend.synthetic if backend else False,
                        backend=backend.describe() if backend else {"name": "baseline-only"},
                        model_load_seconds=time.perf_counter() - model_started)
        if backend is not None and backend.synthetic:
            print("SYNTHETIC SMOKE TEST: these outputs are not research measurements.", flush=True)
        write_json(root / "manifest.json", manifest)
        val_first = None
        if "tune" in tasks:
            pipeline, val_first = tune(validation, backend, pipeline, root / "tuning",
                                       experiment.get("grid_kappas", list(range(1, 13))),
                                       experiment.get("grid_deltas", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
            manifest["completed_tasks"].append("tune")
        write_json(root / "selected_pipeline.json", dict(pipeline=asdict(pipeline),
                    selection="validation_grid" if "tune" in tasks else "configured_values_not_searched_here"))
        repeats, warmups = int(experiment.get("repeats", 3)), int(experiment.get("warmups", 1))
        main_result = None
        if "main" in tasks:
            main_result, latency = benchmark_pipeline(examples, backend, pipeline, repeats, warmups)
            summary["main"] = analyze_main(examples, main_result, root / "main", manifest["dataset_name"], pipeline)
            write_json(root / "main" / "latency.json", latency)
            summary["latency"] = latency
            fixed = PipelineConfig.from_dict({**asdict(pipeline), **experiment.get("fixed", {"kappa": 9.0, "delta": 0.8})})
            fixed_result = run_pipeline([e.query for e in examples], backend, fixed, first=main_result.first)
            fixed_scores = compact_scores(examples, fixed_result)
            selected_f1 = summary["main"]["metrics"]["coconat"]["f1"]
            summary["fixed_transfer"] = dict(selected_f1=selected_f1, fixed_f1=fixed_scores["f1"],
                                             delta=fixed_scores["f1"] - selected_f1, fixed=asdict(fixed),
                                             selected_is_validation_tuned="tune" in tasks)
            write_json(root / "main" / "fixed_transfer.json", summary["fixed_transfer"])
            export_audit([root / "main" / "transition_cases.jsonl"], root / "main" / "audit.csv", seed=pipeline.seed)
            manifest["completed_tasks"].append("main")
        first = main_result.first if main_result else None
        if first is None and set(tasks) & {"ablation", "sensitivity", "calibration"}:
            first = backend.predict([e.query for e in examples])
        if "ablation" in tasks:
            summary["ablations"] = ablations(examples, backend, pipeline, first, root / "ablation",
                                              experiment.get("random_repeats", 5),
                                              experiment.get("embedding_ablation", True))
            manifest["completed_tasks"].append("ablation")
        if "sensitivity" in tasks:
            summary["sensitivity"] = sensitivity(examples, backend, pipeline, first, root / "sensitivity",
                experiment.get("sensitivity_kappas", list(range(3, 13))),
                experiment.get("sensitivity_deltas", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
            manifest["completed_tasks"].append("sensitivity")
        if "calibration" in tasks:
            summary["calibration"] = calibration_experiment(validation, examples, backend, pipeline,
                                        first, root / "calibration", validation_first=val_first)
            manifest["completed_tasks"].append("calibration")
        # Release the resident backbone before allocating another large model.
        del backend
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        if "baselines" in tasks:
            rows = []
            baselines = config.get("baselines", [])
            train = load_split(config["dataset"], "train") if any(
                b.get("enabled", True) and b["kind"] in {"local_llm", "hosted_llm"}
                and b.get("shots_per_type", 5) > 0 for b in baselines) else []
            for index, baseline in enumerate(baselines):
                name = baseline.get("name", baseline.get("checkpoint", baseline["kind"]))
                if not baseline.get("enabled", True):
                    manifest["baseline_coverage"].append(dict(name=name, status="not_run", reason="disabled_in_config"))
                    continue
                candidate = build_backend(baseline, config["dataset"]["labels"], train=train, allow_api=allow_api)
                predictions, timing = benchmark_backend(examples, candidate, repeats, warmups,
                                                        baseline.get("context_mode") == "document")
                scores = entity_metrics(examples, predictions)
                row = dict(method=name, family=baseline.get("family", baseline["kind"]),
                           regime=baseline.get("regime", "unspecified"), f1=scores["f1"],
                           precision=scores["precision"], recall=scores["recall"],
                           seconds_per_1000=timing["seconds_per_1000"], timing_scope=candidate.timing_scope,
                           synthetic=candidate.synthetic, context_mode=baseline.get("context_mode", "sentence"))
                main_latency = next((c["seconds_per_1000"] for c in summary.get("latency", {}).get("components", [])
                                     if c["component"] == "total_seconds"), None)
                row["relative_latency"] = timing["seconds_per_1000"] / main_latency if (
                    timing["seconds_per_1000"] is not None and main_latency) else None
                row["ratio_note"] = "operational_only; compare recorded hardware_and_timing_scopes"
                manifest["synthetic"] = manifest["synthetic"] or candidate.synthetic
                rows.append(row)
                dest = root / "baselines" / f"baseline_{index:02d}"
                save_predictions(dest / "predictions.jsonl", examples, predictions)
                write_json(dest / "result.json", dict(model=candidate.describe(), timing=timing, scores=scores,
                                                       config=redact_config(baseline)))
                write_csv(root / "baselines" / "comparison.csv", rows)
                manifest["baseline_coverage"].append(dict(name=name, status="completed", path=str(dest)))
                del candidate
                gc.collect()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except (NameError, UnboundLocalError):
                    pass
            if not baselines:
                manifest["baseline_coverage"].append(dict(name="all", status="not_run", reason="no_baselines_configured"))
            configured = {b.get("name", b.get("checkpoint", b["kind"])) for b in baselines}
            for name in sorted(set(config.get("required_baselines", [])) - configured):
                manifest["baseline_coverage"].append(dict(name=name, status="not_run", reason="missing_from_config"))
            summary["baselines"] = rows
            manifest["completed_tasks"].append("baselines")
        gaps = any(row["status"] != "completed" for row in manifest["baseline_coverage"])
        manifest["status"] = "completed_with_baseline_gaps" if gaps else "completed"
        summary["synthetic"] = manifest["synthetic"]
        write_json(root / "summary.json", summary)
        write_json(root / "manifest.json", manifest)
        from .report import report_run
        report_run(root)
        print(f"Finished: {manifest['status']} -> {root}", flush=True)
        return root
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure_type"] = type(exc).__name__
        write_json(root / "manifest.json", manifest)
        write_json(root / "partial_summary.json", summary)
        raise
