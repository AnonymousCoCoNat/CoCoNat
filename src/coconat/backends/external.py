"""Gold-blind bridge to official implementations, or strict import of their outputs."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile

from .base import Backend
from ..data import read_jsonl
from ..schema import Evidence, Span, validate_evidence


def load_source_manifest(path):
    """Load concrete upstream provenance and reject unedited templates."""
    if not path:
        raise ValueError("Official integrations require a source_manifest for provenance")
    with Path(path).open(encoding="utf-8") as stream:
        provenance = json.load(stream)
    required = {"method", "source_url", "source_revision", "checkpoint", "supervision"}
    missing = required - set(provenance)
    if missing:
        raise ValueError(f"Source manifest lacks {sorted(missing)}")
    invalid = [key for key in required if not isinstance(provenance[key], str)
               or not provenance[key].strip() or "REPLACE" in provenance[key].upper()]
    if invalid:
        raise ValueError(f"Source manifest has unset provenance fields: {sorted(invalid)}")
    return provenance


def parse_prediction_rows(queries, rows, labels, label_map=None):
    qmap = {q.id: q for q in queries}
    output = {}
    mapping = label_map or {label: label for label in labels}
    for row in rows:
        qid = str(row["id"])
        if qid not in qmap or qid in output:
            raise ValueError(f"Unknown or duplicate prediction ID: {qid}")
        q = qmap[qid]
        if row.get("tokens") != list(q.tokens):
            raise ValueError(f"Predictions for {qid} must echo the exact dataset tokens")
        spans = []
        for item in row["spans"]:
            if item["label"] not in mapping:
                raise ValueError(f"Missing external label mapping for {item['label']}")
            if type(item["start"]) is not int or type(item["end"]) is not int:
                raise ValueError("External start/end must be integer word offsets")
            spans.append(Span(item["start"], item["end"], mapping[item["label"]],
                              float(item.get("score", 1.0))))
        confidence_available = all("score" in s for s in row["spans"])
        output[qid] = Evidence(tuple(spans), confidence_available=confidence_available,
                               metadata=row.get("metadata", {}))
        validate_evidence(q, output[qid], labels)
    if set(output) != set(qmap):
        raise ValueError(f"Missing prediction IDs: {sorted(set(qmap) - set(output))[:10]}")
    return output


class ExternalPredictions(Backend):
    timing_scope = "imported_predictions_no_inference_latency"

    def __init__(self, prediction_path, labels, name="official-output-import",
                 source_manifest=None, label_map=None, **kwargs):
        self.name, self.labels = name, tuple(labels)
        self.rows = list(read_jsonl(prediction_path))
        self.label_map = label_map
        self.provenance = load_source_manifest(source_manifest)

    def predict(self, queries):
        return parse_prediction_rows(queries, self.rows, self.labels, self.label_map)

    def predict_groups(self, groups):
        raise ValueError("Stored first-pass predictions cannot simulate a contextualized second pass")

    def describe(self):
        return {**super().describe(), "source": self.provenance}


class ExternalCommand(Backend):
    timing_scope = "external_process_including_startup_and_model_loading"

    def __init__(self, command, labels, name="official-external-command", cwd=None,
                 timeout=3600, label_map=None, source_manifest=None, **kwargs):
        if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
            raise ValueError("command must be an argument list, never a shell string")
        self.command, self.labels, self.name = command, tuple(labels), name
        self.cwd, self.timeout, self.label_map = cwd, timeout, label_map
        self.provenance = load_source_manifest(source_manifest)

    def predict(self, queries):
        with tempfile.TemporaryDirectory(prefix="coconat_bridge_") as temporary:
            source, target = Path(temporary) / "request.json", Path(temporary) / "predictions.jsonl"
            source.write_text(json.dumps(dict(protocol_version=1, labels=list(self.labels),
                                              queries=[q.to_dict() for q in queries])), encoding="utf-8")
            command = [arg.replace("{input}", str(source)).replace("{output}", str(target))
                       for arg in self.command]
            completed = subprocess.run(command, cwd=self.cwd, check=False, capture_output=True,
                                       text=True, timeout=self.timeout, shell=False)
            if completed.returncode:
                raise RuntimeError(f"External runner exited {completed.returncode}. "
                                   "Run the configured command directly to inspect its stderr.")
            if not target.is_file():
                raise RuntimeError("External runner must write one JSONL prediction row per query to {output}")
            rows = list(read_jsonl(target))
        return parse_prediction_rows(queries, rows, self.labels, self.label_map)

    def describe(self):
        return {**super().describe(), "source": self.provenance}
