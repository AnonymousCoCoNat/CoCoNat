"""Validated experiment configuration with paths relative to the YAML file."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import math
import yaml


@dataclass(frozen=True)
class PipelineConfig:
    kappa: float = 9.0
    delta: float = 0.8
    detector: str = "both"
    grouping: str = "entity"
    ordering: str = "gradual"
    aggregation: str = "mean"
    case_sensitive: bool = True
    max_group_size: int = 10
    seed: int = 42
    n_clusters: int | None = None
    gmm_threshold: float = 0.2

    def __post_init__(self):
        if not math.isfinite(self.kappa) or self.kappa < 0:
            raise ValueError("kappa must be finite and >= 0")
        # delta=1 is the endpoint ablation in Figure 4, not a finite odds threshold.
        if not 0.5 <= self.delta <= 1:
            raise ValueError("delta must be in [0.5, 1]")
        choices = dict(detector={"none", "confidence", "inconsistency", "both"},
                       grouping={"entity", "kmeans", "gmm"},
                       ordering={"gradual", "random", "original"}, aggregation={"mean", "max"})
        for key, allowed in choices.items():
            if getattr(self, key) not in allowed:
                raise ValueError(f"Invalid {key}; expected one of {sorted(allowed)}")
        if self.max_group_size < 2 or not 0 < self.gmm_threshold <= 1:
            raise ValueError("max_group_size must be >= 2 and gmm_threshold in (0, 1]")
        if self.n_clusters is not None and self.n_clusters < 1:
            raise ValueError("n_clusters must be positive")

    @classmethod
    def from_dict(cls, value):
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown pipeline options: {sorted(unknown)}")
        return cls(**value)


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    config = dict(config)
    root = path.parent

    def resolve(value):
        return str((root / value).resolve())

    ds = config.get("dataset", {})
    if not ds.get("labels") or len(ds["labels"]) != len(set(ds["labels"])) or "O" in ds["labels"]:
        raise ValueError("dataset.labels must list unique entity types, excluding O")
    for split in ("train", "validation", "test"):
        if split in ds and ds.get("format") != "hf":
            ds[split] = resolve(ds[split])
    config["output_dir"] = resolve(config.get("output_dir", "../outputs/run"))
    for model in [config.get("model", {})] + config.get("baselines", []):
        for key in ("prediction_path", "source_manifest", "cwd", "prompt_file"):
            if model.get(key):
                model[key] = resolve(model[key])
        if model.get("checkpoint", "").startswith(("./", "../")):
            model["checkpoint"] = resolve(model["checkpoint"])
    config["_config_path"] = str(path)
    PipelineConfig.from_dict(config.get("pipeline", {}))
    return config
