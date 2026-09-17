"""Synthetic fixtures for exercising the suite without downloading models."""

from pathlib import Path
import uuid

import yaml

from .io import write_jsonl


def make_demo(base):
    folder = Path(base).resolve() / ("fixture_" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    sentences = [
        ("Acme reported revenue .", "ORG"), ("Acme arrived yesterday .", "ORG"),
        ("Amazon announced revenue .", "ORG"), ("Amazon arrived today .", "ORG"),
        ("The Amazon river flows .", "LOC"), ("Orion is a company .", "ORG"),
        ("Orion arrived in London .", "ORG"), ("Jordan is a country .", "LOC"),
        ("Jordan played sports .", "PER"), ("Samdory visited Paris .", "PER"),
        ("Takagi visited London .", "PER"), ("There are no names here .", None),
    ]
    for split in ("train", "validation", "test"):
        rows = []
        for repetition in range(2 if split == "train" else 1):
            for i, (text, label) in enumerate(sentences):
                tokens = text.split() + [f"{split}{repetition}"]
                tags = ["O"] * len(tokens)
                if label:
                    tags[1 if tokens[0] == "The" else 0] = "B-" + label
                for j, token in enumerate(tokens):
                    if token in {"Paris", "London"}:
                        tags[j] = "B-LOC"
                rows.append(dict(id=f"{split}-{repetition}-{i}", tokens=tokens, ner_tags=tags,
                                 doc_id=f"doc-{i // 3}"))
        write_jsonl(folder / f"{split}.jsonl", rows)
    config = dict(dataset=dict(name="synthetic-demo", format="jsonl", labels=["ORG", "PER", "LOC"],
                               train="train.jsonl", validation="validation.jsonl", test="test.jsonl"),
                  model=dict(kind="toy", max_length=128),
                  pipeline=dict(kappa=1.0, delta=0.8, seed=42), output_dir="../runs",
                  experiments=dict(grid_kappas=[0, 1, 2], grid_deltas=[0.5, 0.8, 1.0],
                                   sensitivity_kappas=[0, 1, 2, 3], sensitivity_deltas=[0.5, 0.8, 1.0],
                                   repeats=3, warmups=1, random_repeats=5, embedding_ablation=True),
                  baselines=[dict(kind="toy", name="synthetic-backbone", family="test-only",
                                  regime="synthetic", enabled=True)])
    config_path = folder / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path
