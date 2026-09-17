"""Export real failure cases for human annotation; never invent manual-audit percentages."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from .data import read_jsonl
from .io import write_csv, write_json


REASONS = ("homonymous_surface", "weak_evidence", "token_budget", "boundary_change", "other")


def export_audit(case_files, output, sample_size=100, seed=42):
    pool = []
    for file in case_files:
        for row in read_jsonl(file):
            if row["transition"] == "C_W":
                pool.append({"source_run": str(Path(file).resolve().parent), **row})
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(pool), min(sample_size, len(pool)), replace=False) if pool else []
    rows = []
    for index in selected:
        row = pool[int(index)]
        rows.append(dict(case_id=f"case-{int(index):06d}", source_run=row["source_run"],
                         query_id=row["query_id"], text=row["text"], before=row["before"],
                         after=row["after"], gold=row["gold"], reason="", notes=""))
    write_csv(output, rows, columns=["case_id", "source_run", "query_id", "text", "before", "after", "gold", "reason", "notes"])
    write_json(str(output) + ".manifest.json", dict(pool_size=len(pool), requested=sample_size,
                                                    sampled=len(rows), seed=seed, allowed_reasons=REASONS,
                                                    annotation_status="unannotated"))
    return len(rows)


def summarize_audit(path, output):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate case IDs in audit annotations")
    invalid = {r["reason"] for r in rows if r["reason"] and r["reason"] not in REASONS}
    if invalid:
        raise ValueError(f"Unknown audit reasons: {sorted(invalid)}")
    counts = Counter(r["reason"] for r in rows if r["reason"])
    n = sum(counts.values())
    write_json(output, dict(total=len(rows), annotated=n, unannotated=len(rows) - n,
                            complete=n == len(rows) and n > 0, counts=dict(counts),
                            fractions={reason: counts[reason] / n if n else None for reason in REASONS}))
