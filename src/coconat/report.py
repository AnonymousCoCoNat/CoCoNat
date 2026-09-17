"""CSV, LaTeX, plots, and multi-dataset summaries generated only from measured runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .data import read_jsonl
from .io import write_csv, write_json
from .metrics import label_balanced_strata, strata_summary


def _tex(value):
    text = str(value)
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(char, char) for char in text)


def latex_table(path, headers, rows, caption, synthetic=False):
    prefix = "% SYNTHETIC TEST OUTPUT - NOT FOR PUBLICATION\n" if synthetic else "% Computed from recorded outputs.\n"
    lines = [prefix + r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{" + _tex(caption) + "}",
             r"\begin{tabular}{" + "l" + "r" * (len(headers) - 1) + "}", r"\toprule",
             " & ".join(_tex(h) for h in headers) + r" \\", r"\midrule"]
    lines.extend(" & ".join(_tex(v) for v in row) + r" \\" for row in rows)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def pct(value):
    return "--" if value is None else f"{value * 100:.2f}"


def report_run(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    synthetic = manifest.get("synthetic", False)
    report = root / "report"
    report.mkdir(exist_ok=True)
    if "main" in summary:
        metrics = summary["main"]["metrics"]
        rows = [[method, pct(values["precision"]), pct(values["recall"]), pct(values["f1"])]
                for method, values in metrics.items()]
        latex_table(report / "performance.tex", ["Method", "P", "R", "F1"], rows,
                    "Exact-match entity precision, recall, and F1 (percent).", synthetic)
        trans = summary["main"]["transitions"]
        rows = [[name.replace("_", " to "), trans["counts"][name], pct(trans["fractions"][name])]
                for name in ("W_C", "C_C", "W_W", "C_W")]
        rows.append(["Net correction", trans["net_corrections"], pct(trans["net_correction_fraction"])])
        latex_table(report / "transitions.tex", ["Transition", "Count", "Percent"], rows,
                    "First-pass predicted-span transitions in reprocessed queries; W is wrong and C is correct.", synthetic)
    if summary.get("baselines"):
        rows = [[r["method"], r["regime"], pct(r["f1"]),
                 "--" if r["seconds_per_1000"] is None else f"{r['seconds_per_1000']:.3f}"]
                for r in summary["baselines"]]
        latex_table(report / "baselines.tex", ["Method", "Regime", "F1", "s/1k"], rows,
                    "Recent baselines. Timing scopes and hardware are recorded separately.", synthetic)
    if "calibration" in summary:
        c = summary["calibration"]
        latex_table(report / "calibration.tex", ["T", "Raw ECE", "Cal. ECE", "Raw F1", "Cal. F1", "Overlap"],
                    [[f"{c['temperature']:.3f}", pct(c["raw_ece"]["ece"]), pct(c["calibrated_ece"]["ece"]),
                      pct(c["raw_final_f1"]), pct(c["calibrated_final_f1"]), f"{c['hard_span_jaccard']:.3f}"]],
                    "Validation-fitted temperature, entity ECE, and hard-span Jaccard overlap.", synthetic)
    if "ablations" in summary:
        latex_table(report / "ablations.tex", ["Variant", "F1", "Hard queries", "C to W (%)"],
                    [[r["variant"], pct(r["f1"]), r["hard_queries"], pct(r["C_W_fraction"])]
                     for r in summary["ablations"]],
                    "Detector, grouping, ordering, and aggregation ablations.", synthetic)
    if "fixed_transfer" in summary:
        fixed = summary["fixed_transfer"]
        latex_table(report / "fixed_transfer.tex", ["Selected F1", "Fixed F1", "Delta (pp)"],
                    [[pct(fixed["selected_f1"]), pct(fixed["fixed_f1"]), pct(fixed["delta"])]],
                    "One fixed setting versus the selected configuration; selection provenance is in selected_pipeline.json.", synthetic)
    for source_name in ("length_strata", "label_strata", "label_balanced_strata"):
        source = root / "main" / (source_name + ".csv")
        if source.is_file():
            with source.open(newline="") as stream:
                records = list(csv.DictReader(stream))
            latex_table(report / (source_name + ".tex"), ["Stratum", "Mean confidence", "Flagged (%)", "Error among flagged (%)"],
                        [[r["stratum"], f"{float(r['mean_confidence']):.3f}", pct(float(r["flagged_fraction"])),
                          pct(float(r["error_among_flagged"])) if r["error_among_flagged"] else "--"] for r in records],
                        "Low-confidence screening by " + source_name.replace("_", " ") + ".", synthetic)
    text = ["# CoCoNat run report", "", "SYNTHETIC TEST OUTPUT - NOT FOR PUBLICATION" if synthetic else "Measured experiment outputs.",
            "", f"Status: {manifest['status']}", f"Dataset: {manifest.get('dataset_name', 'unknown')}",
            "", "Missing or disabled baselines are listed in manifest.json. Missing measurements are never imputed.",
            "", "Inspect transition_cases.jsonl and annotate audit.csv before making manual error-cause claims."]
    (report / "README.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    if (root / "sensitivity" / "sensitivity.csv").is_file():
        plot_sensitivity(root)


def plot_sensitivity(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    with (root / "sensitivity" / "sensitivity.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    manifest = json.loads((root / "manifest.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
    for i, name in enumerate(("kappa", "delta")):
        selected = sorted((r for r in rows if r["parameter"] == name), key=lambda r: float(r["value"]))
        for j, (field, label) in enumerate((("f1", "Entity F1 (%)"), ("hard_queries", "Hard queries"))):
            values = [float(r[field]) * (100 if field == "f1" else 1) for r in selected]
            axes[i, j].plot([float(r["value"]) for r in selected], values, marker="o")
            axes[i, j].set(xlabel=name, ylabel=label)
            axes[i, j].grid(alpha=0.2)
    fig.suptitle("SYNTHETIC TEST - NOT FOR PUBLICATION" if manifest.get("synthetic") else manifest["dataset_name"])
    (root / "report").mkdir(exist_ok=True)
    fig.savefig(root / "report" / "sensitivity.png", dpi=180)
    fig.savefig(root / "report" / "sensitivity.pdf")
    plt.close(fig)


def merge_runs(roots, output, allow_synthetic=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty merge output directory")
    output.mkdir(parents=True, exist_ok=True)
    characteristics, performance, fixed = [], [], []
    calibration, baselines, recent, observations, provenance = [], [], [], [], []
    for root in map(Path, roots):
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("synthetic") and not allow_synthetic:
            raise ValueError("Synthetic runs require --allow-synthetic and must not enter publication tables")
        if not manifest["status"].startswith("completed"):
            raise ValueError(f"Cannot merge incomplete run {root}")
        summary = json.loads((root / "summary.json").read_text())
        name = manifest["dataset_name"]
        meta = dict(dataset=name, model=manifest["backend"]["name"], run=str(root))
        if "main" in summary:
            characteristics.append({**meta, **summary["main"]["characteristics"]})
            observations.extend(read_jsonl(root / "main" / "screening_observations.jsonl"))
            for stage, scores in summary["main"]["metrics"].items():
                performance.append({**meta, "stage": stage, "precision": scores["precision"],
                                    "recall": scores["recall"], "f1": scores["f1"]})
            components = {row["component"]: row["seconds_per_1000"]
                          for row in summary.get("latency", {}).get("components", [])}
            first_latency, total_latency = components.get("first_seconds"), components.get("total_seconds")
            recent.append({**meta, "method": manifest["backend"]["name"],
                           "family": "supervised backbone", "regime": "full supervision",
                           "f1": summary["main"]["metrics"]["backbone"]["f1"],
                           "seconds_per_1000": first_latency,
                           "relative_latency": first_latency / total_latency if first_latency and total_latency else None,
                           "timing_scope": manifest["backend"].get("timing_scope")})
            recent.append({**meta, "method": manifest["backend"]["name"] + " + CoCoNat",
                           "family": "inference overlay", "regime": "full supervision",
                           "f1": summary["main"]["metrics"]["coconat"]["f1"],
                           "seconds_per_1000": total_latency, "relative_latency": 1.0 if total_latency else None,
                           "timing_scope": manifest["backend"].get("timing_scope")})
        if "fixed_transfer" in summary:
            fixed.append({**meta, **summary["fixed_transfer"]})
        if "calibration" in summary:
            c = summary["calibration"]
            calibration.append({**meta, "temperature": c["temperature"], "raw_ece": c["raw_ece"]["ece"],
                                "calibrated_ece": c["calibrated_ece"]["ece"], "raw_f1": c["raw_final_f1"],
                                "calibrated_f1": c["calibrated_final_f1"], "overlap": c["hard_span_jaccard"]})
        current_baselines = [{**meta, **row} for row in summary.get("baselines", [])]
        baselines.extend(current_baselines)
        recent.extend(current_baselines)
        provenance.append(manifest)
    write_csv(output / "dataset_characteristics.csv", characteristics)
    write_csv(output / "performance.csv", performance)
    write_csv(output / "fixed_transfer.csv", fixed)
    write_csv(output / "calibration.csv", calibration)
    write_csv(output / "baselines.csv", baselines)
    write_csv(output / "recent_comparison.csv", recent)
    write_csv(output / "pooled_length_strata.csv", strata_summary(observations, "length_bin"))
    write_csv(output / "pooled_label_strata.csv", strata_summary(observations, "label"))
    write_csv(output / "pooled_label_balanced.csv", label_balanced_strata(observations))
    write_json(output / "source_runs.json", provenance)
    latex_table(output / "dataset_characteristics.tex",
        ["Dataset", "Hard (%)", "Repeated (%)", "Backbone F1", "+CoCoNat F1", "Gain (pp)"],
        [[r["dataset"], pct(r["hard_fraction"]), pct(r["repeated_hard_fraction"]), pct(r["first_f1"]),
          pct(r["final_f1"]), pct(r["gain"])] for r in characteristics],
        "Dataset characteristics from measured model outputs.", any(m.get("synthetic") for m in provenance))
    latex_table(output / "performance.tex", ["Dataset", "Model", "Stage", "P", "R", "F1"],
        [[r["dataset"], r["model"], r["stage"], pct(r["precision"]), pct(r["recall"]), pct(r["f1"])]
         for r in performance], "Backbone and CoCoNat exact-match entity results.",
        any(m.get("synthetic") for m in provenance))
    latex_table(output / "recent_comparison.tex", ["Dataset", "Method", "Regime", "F1", "s/1k", "Relative"],
        [[r["dataset"], r["method"], r["regime"], pct(r["f1"]),
          "--" if r.get("seconds_per_1000") is None else f"{r['seconds_per_1000']:.3f}",
          "--" if r.get("relative_latency") is None else f"{r['relative_latency']:.2f}x"] for r in recent],
        "Recent-system comparison; supervision and timing scopes must be read with the source CSV.",
        any(m.get("synthetic") for m in provenance))
