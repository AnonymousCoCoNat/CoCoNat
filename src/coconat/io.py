"""Portable result writing and minimal, credential-free provenance."""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot encode {type(value)} as JSON")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                                    default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n")


def write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    columns = columns or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, default=_json_default) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def query_fingerprint(examples):
    digest = hashlib.sha256()
    for ex in examples:
        digest.update(json.dumps(ex.query.to_dict(), sort_keys=True, ensure_ascii=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def labeled_fingerprint(examples):
    digest = hashlib.sha256(query_fingerprint(examples).encode())
    for ex in examples:
        digest.update(json.dumps([s.key() for s in ex.gold]).encode())
    return digest.hexdigest()


def redact_config(value):
    if isinstance(value, dict):
        forbidden = {"api_key", "password", "secret", "authorization", "access_token"}
        return {k: "[REDACTED]" if k.lower() in forbidden else redact_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_config(v) for v in value]
    return value


def environment_info():
    versions = {}
    for package in ("coconat", "numpy", "torch", "transformers", "datasets", "gliner", "scipy", "scikit-learn"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    info = dict(python=sys.version, platform=platform.platform(), processor=platform.processor(), packages=versions)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        info["git_commit"] = commit.stdout.strip() if commit.returncode == 0 else None
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5)
        info["git_dirty"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["git_commit"] = None
    try:
        import torch
        info["cuda_version"] = torch.version.cuda
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        info["gpus"] = []
    return info


def save_predictions(path, examples, predictions):
    write_jsonl(path, [dict(id=ex.query.id, tokens=list(ex.query.tokens),
                           spans=[s.to_dict() for s in predictions[ex.query.id].spans],
                           metadata=predictions[ex.query.id].metadata) for ex in examples])
