"""Explicit split loading and lossless token-index conversion for NER datasets."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import Example, Query, Span, tags_to_spans


def example_from_dict(row, fallback_id):
    query = Query(str(row.get("id", fallback_id)), tuple(row["tokens"]), row.get("doc_id"))
    if "spans" in row:
        gold = tuple(Span(int(s["start"]), int(s["end"]), str(s["label"])) for s in row["spans"])
    elif "ner_tags" in row:
        if len(row["ner_tags"]) != len(query.tokens):
            raise ValueError(f"Token/tag length mismatch for {query.id}")
        gold = tags_to_spans(row["ner_tags"])
    else:
        raise ValueError(f"Missing gold spans or string ner_tags in {query.id}")
    return Example(query, gold)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for i, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{i}: {exc}") from exc


def read_conll(path, token_column=0, tag_column=-1):
    """Read whitespace-separated CoNLL BIO data, preserving DOCSTART boundaries.

    This is not a generic CoNLL-U parser: specify the NER column explicitly.
    """
    examples, tokens, tags, doc_index = [], [], [], 0

    def emit():
        if tokens:
            examples.append(Example(Query(str(len(examples)), tuple(tokens), str(doc_index)),
                                    tags_to_spans(tags)))
            tokens.clear()
            tags.clear()

    with Path(path).open(encoding="utf-8-sig") as stream:
        for i, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                emit()
            elif line.startswith("-DOCSTART-"):
                emit()
                doc_index += 1
            elif line.startswith("#"):
                continue
            else:
                columns = line.split()
                try:
                    tokens.append(columns[token_column])
                    tags.append(columns[tag_column])
                except IndexError as exc:
                    raise ValueError(f"Missing CoNLL column at {path}:{i}") from exc
    emit()
    return examples


def load_split(config, split):
    if split not in {"train", "validation", "test"}:
        raise ValueError("Split must be train, validation, or test")
    if config["format"] == "hf":
        from datasets import load_dataset

        ds = load_dataset(config["path"], config.get("subset"),
                          split=config.get("splits", {}).get(split, split),
                          revision=config.get("revision"), trust_remote_code=False)
        tag_field = config.get("tag_field", "ner_tags")
        token_field = config.get("token_field", "tokens")
        feature = ds.features[tag_field]
        names = config.get("tag_names") or getattr(getattr(feature, "feature", None), "names", None)
        examples = []
        for i, row in enumerate(ds):
            tags = row[tag_field]
            if tags and isinstance(tags[0], int):
                if names is None:
                    raise ValueError("Integer tags require tag_names or a ClassLabel feature")
                tags = [names[t] for t in tags]
            examples.append(example_from_dict(dict(
                id=str(row.get("id", i)), tokens=row[token_field], ner_tags=tags,
                doc_id=row.get(config.get("doc_field", "doc_id"))), i))
    else:
        path = Path(config[split])
        if not path.is_file():
            raise FileNotFoundError(f"Dataset {split} file not found: {path}")
        if config["format"] == "jsonl":
            examples = [example_from_dict(row, i) for i, row in enumerate(read_jsonl(path))]
        elif config["format"] == "conll":
            examples = read_conll(path, config.get("token_column", 0), config.get("tag_column", -1))
        else:
            raise ValueError(f"Unsupported dataset format {config['format']!r}")
    ids = [e.query.id for e in examples]
    if not examples or len(ids) != len(set(ids)):
        raise ValueError(f"Empty split or duplicate query IDs in {split}")
    labels = set(config["labels"])
    unknown = {s.label for e in examples for s in e.gold} - labels
    if unknown:
        raise ValueError(f"Gold types missing from the configured schema: {sorted(unknown)}")
    return examples


def export_split(examples, path, include_gold=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for ex in examples:
            row = ex.query.to_dict()
            if include_gold:
                row["spans"] = [s.to_dict() for s in ex.gold]
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
