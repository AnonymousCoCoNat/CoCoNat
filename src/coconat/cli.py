"""Command-line entry point; imports never start training or network requests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml

from .config import load_config
from .data import example_from_dict, export_split, load_split, read_conll, read_jsonl
from .io import environment_info, write_jsonl


def parser():
    p = argparse.ArgumentParser(prog="coconat", description="CoCoNat training and reproducible revision experiments")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("run", "matrix"):
        run = sub.add_parser(name, help="Run selected experiments or dataset configs sequentially")
        run.add_argument("--config" if name == "run" else "--configs", required=True,
                         nargs=None if name == "run" else "+")
        run.add_argument("--tasks", default="all", help="all or comma-separated tune,main,ablation,sensitivity,calibration,baselines")
        run.add_argument("--output", help="Parent output directory; prior runs are preserved")
        run.add_argument("--allow-api", action="store_true", help="Allow potentially billable hosted calls")
        run.add_argument("--require-all", action="store_true", help="Fail on disabled/missing baselines")
    demo = sub.add_parser("demo", help="Run every analysis on synthetic data")
    demo.add_argument("--output", default="outputs/demo")
    train = sub.add_parser("train", help="Fine-tune an HF BIO backbone on train/validation only")
    train.add_argument("--config", required=True)
    train.add_argument("--output")
    prepare = sub.add_parser("prepare", help="Normalize local splits and generate an experiment config")
    for split in ("train", "validation", "test"):
        prepare.add_argument("--" + split, required=True)
    prepare.add_argument("--format", choices=("conll", "jsonl"), default="conll")
    prepare.add_argument("--name", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--token-column", type=int, default=0)
    prepare.add_argument("--tag-column", type=int, default=-1)
    prepare.add_argument("--labels", help="Official comma-separated schema; otherwise inferred from train")
    export = sub.add_parser("export", help="Export gold-blind inputs for official implementations")
    export.add_argument("--config", required=True)
    export.add_argument("--split", default="test", choices=("train", "validation", "test"))
    export.add_argument("--include-gold", action="store_true", help="Train/validation only")
    export.add_argument("--output", required=True)
    audit = sub.add_parser("audit-export", help="Pool and sample real C-to-W cases for manual annotation")
    audit.add_argument("--cases", nargs="+", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--sample-size", type=int, default=100)
    audit.add_argument("--seed", type=int, default=42)
    audit_sum = sub.add_parser("audit-summarize", help="Summarize user-annotated error causes")
    audit_sum.add_argument("--input", required=True)
    audit_sum.add_argument("--output", required=True)
    merge = sub.add_parser("merge", help="Merge runs and pool length/label analyses")
    merge.add_argument("--runs", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--allow-synthetic", action="store_true")
    plot = sub.add_parser("plot", help="Plot a completed sensitivity sweep")
    plot.add_argument("--run", required=True)
    doctor = sub.add_parser("doctor", help="Inspect dependencies without model downloads")
    doctor.add_argument("--config")
    convert = sub.add_parser("convert-predictions", help="Normalize official per-query predictions")
    convert.add_argument("--config", required=True)
    convert.add_argument("--input", required=True)
    convert.add_argument("--format", choices=("bio", "word", "char", "generation"), required=True)
    convert.add_argument("--output", required=True)
    return p


def prepare_data(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty dataset output directory")
    splits = {}
    for split in ("train", "validation", "test"):
        source = getattr(args, split)
        examples = read_conll(source, args.token_column, args.tag_column) if args.format == "conll" else [
            example_from_dict(row, i) for i, row in enumerate(read_jsonl(source))]
        if not examples:
            raise ValueError(f"Empty {split} data")
        splits[split] = examples
    labels = ([label.strip() for label in args.labels.split(",") if label.strip()]
              if args.labels else sorted({s.label for e in splits["train"] for s in e.gold}))
    if not labels or len(labels) != len(set(labels)) or "O" in labels:
        raise ValueError("--labels must contain unique entity types and must exclude O")
    for split, examples in splits.items():
        unknown = {s.label for e in examples for s in e.gold} - set(labels)
        if unknown:
            raise ValueError(f"{split} has types outside the declared schema: {sorted(unknown)}; use --labels")
    for split, examples in splits.items():
        export_split(examples, output / f"{split}.jsonl", include_gold=True)
    checkpoint = os.path.relpath(Path.cwd() / "checkpoints" / args.name / "deberta-v3-base", output)
    result_dir = os.path.relpath(Path.cwd() / "outputs" / args.name, output)
    config = dict(dataset=dict(name=args.name, format="jsonl", labels=labels,
                               train="train.jsonl", validation="validation.jsonl", test="test.jsonl"),
                  model=dict(kind="hf", checkpoint="./" + checkpoint, device="auto", max_length=512, batch_size=8),
                  training=dict(base_checkpoint="microsoft/deberta-v3-base", epochs=3, batch_size=8,
                                learning_rate=2e-5, seed=42),
                  pipeline=dict(kappa=9.0, delta=0.8, detector="both", grouping="entity",
                                ordering="gradual", aggregation="mean", case_sensitive=True, seed=42),
                  experiments=dict(repeats=3, warmups=1, random_repeats=5, embedding_ablation=True),
                  baselines=[], output_dir=result_dir)
    path = output / "experiment.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Prepared data and experiment config: {path}")
    print("Add entries from configs/baselines.yaml for a complete modern comparison.")


def convert_predictions(args):
    from .backends.base import char_span_to_word
    from .backends.external import parse_prediction_rows
    from .backends.llm import parse_generation
    from .schema import tags_to_spans
    config = load_config(args.config)
    examples = load_split(config["dataset"], "test")
    qmap = {e.query.id: e.query for e in examples}
    rows = []
    for row in read_jsonl(args.input):
        qid = str(row["id"])
        if qid not in qmap:
            raise ValueError(f"Unknown query ID: {qid}")
        q = qmap[qid]
        if row.get("tokens") != list(q.tokens):
            raise ValueError(f"Input tokens must be echoed exactly for {qid}")
        if args.format == "bio":
            if len(row["predicted_tags"]) != len(q.tokens):
                raise ValueError(f"BIO length mismatch for {qid}")
            spans = [{"start": s.start, "end": s.end, "label": s.label} for s in tags_to_spans(row["predicted_tags"])]
        elif args.format == "generation":
            ev = parse_generation(row["response"], q, config["dataset"]["labels"])
            spans = [{"start": s.start, "end": s.end, "label": s.label} for s in ev.spans]
        elif args.format == "char":
            spans = []
            for s in row["spans"]:
                boundary = char_span_to_word(q, s["start"], s["end"])
                if boundary is None:
                    raise ValueError(f"Unmappable character span for {qid}; inspect the upstream tokenizer")
                spans.append({**s, "start": boundary[0], "end": boundary[1]})
        else:
            spans = row["spans"]
        rows.append(dict(id=qid, tokens=list(q.tokens), spans=spans))
    parse_prediction_rows([e.query for e in examples], rows, config["dataset"]["labels"])
    write_jsonl(args.output, rows)
    print(f"Validated {len(rows)} prediction rows: {args.output}")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in {"run", "matrix"}:
            from .runner import ALL_TASKS, run_suite
            tasks = ALL_TASKS if args.tasks == "all" else [task.strip() for task in args.tasks.split(",") if task.strip()]
            for path in [args.config] if args.command == "run" else args.configs:
                run_suite(load_config(path), tasks, args.output, args.allow_api, args.require_all)
        elif args.command == "demo":
            from .demo import make_demo
            from .runner import run_suite
            run_suite(load_config(make_demo(args.output)), require_all=True)
        elif args.command == "train":
            from .training import train
            print(f"Checkpoint saved: {train(load_config(args.config), args.output)}")
        elif args.command == "prepare":
            prepare_data(args)
        elif args.command == "export":
            if args.include_gold and args.split == "test":
                raise ValueError("Inference exports must not include gold test annotations")
            config = load_config(args.config)
            export_split(load_split(config["dataset"], args.split), args.output, args.include_gold)
        elif args.command == "audit-export":
            from .audit import export_audit
            print(f"Exported {export_audit(args.cases, args.output, args.sample_size, args.seed)} real cases")
        elif args.command == "audit-summarize":
            from .audit import summarize_audit
            summarize_audit(args.input, args.output)
        elif args.command == "merge":
            from .report import merge_runs
            merge_runs(args.runs, args.output, args.allow_synthetic)
        elif args.command == "plot":
            from .report import plot_sensitivity
            plot_sensitivity(args.run)
        elif args.command == "doctor":
            info = environment_info()
            if args.config:
                config = load_config(args.config)
                info["config_valid"] = True
                info["checkpoint_exists_locally"] = Path(config["model"].get("checkpoint", "__none__")).exists()
            print(json.dumps(info, indent=2))
        elif args.command == "convert-predictions":
            convert_predictions(args)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError, PermissionError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
