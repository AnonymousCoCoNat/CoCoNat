"""Shared contracts. All entity offsets are half-open WORD offsets, not bytes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int
    label: str
    score: float = field(default=1.0, compare=False)

    def __post_init__(self):
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Invalid half-open span: {self.start}:{self.end}")
        if not self.label or self.label == "O":
            raise ValueError("An entity must have a non-O type")
        if not np.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("Entity confidence must be finite and in [0, 1]")

    def key(self):
        return self.start, self.end, self.label

    def to_dict(self):
        return dict(start=self.start, end=self.end, label=self.label, score=float(self.score))


@dataclass(frozen=True)
class Query:
    """Inference input deliberately excludes gold annotations."""

    id: str
    tokens: tuple[str, ...]
    doc_id: str | None = None

    def __post_init__(self):
        if not self.id or not self.tokens or any(not t or any(c.isspace() for c in t) for t in self.tokens):
            raise ValueError("Queries need an ID and nonempty, whitespace-free word tokens")

    @property
    def text(self):
        # Canonicalization is explicit: one space between original dataset tokens.
        return " ".join(self.tokens)

    @property
    def char_offsets(self):
        result, offset = [], 0
        for token in self.tokens:
            result.append((offset, offset + len(token)))
            offset += len(token) + 1
        return result

    def surface(self, span: Span):
        return " ".join(self.tokens[span.start:span.end])

    def to_dict(self):
        return dict(id=self.id, tokens=list(self.tokens), text=self.text, doc_id=self.doc_id)


@dataclass(frozen=True)
class Example:
    query: Query
    gold: tuple[Span, ...]

    def __post_init__(self):
        if any(s.end > len(self.query.tokens) for s in self.gold):
            raise ValueError(f"Gold span outside query {self.query.id}")
        if len(set(s.key() for s in self.gold)) != len(self.gold):
            raise ValueError(f"Duplicate gold span in {self.query.id}")


def softmax(logits, temperature=1.0):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    values = np.asarray(logits, dtype=np.float64) / temperature
    values = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=-1, keepdims=True)


def tags_to_spans(tags: Sequence[str], scores=None):
    """Decode BIO/BIOES/BILOU. Orphan I/E/L begins a new span, as in seqeval."""
    spans, start, current, conf = [], None, None, []

    def close(end):
        nonlocal start, current, conf
        if start is not None:
            spans.append(Span(start, end, current, float(np.mean(conf))))
        start, current, conf = None, None, []

    for i, tag in enumerate(tags):
        if tag == "O":
            close(i)
            continue
        if "-" not in tag or tag.split("-", 1)[0] not in {"B", "I", "E", "S", "L", "U"}:
            raise ValueError(f"Expected BIO/BIOES/BILOU tag, got {tag!r}")
        prefix, label = tag.split("-", 1)
        if not label:
            raise ValueError(f"Empty entity type in {tag!r}")
        if prefix in {"B", "S", "U"} or current != label:
            close(i)
        if start is None:
            start, current = i, label
        conf.append(1.0 if scores is None else float(scores[i]))
        if prefix in {"S", "U", "E", "L"}:
            close(i + 1)
    close(len(tags))
    return tuple(spans)


def spans_to_tags(n_words: int, spans: Sequence[Span]):
    tags = ["O"] * n_words
    for span in sorted(spans):
        if span.end > n_words or any(t != "O" for t in tags[span.start:span.end]):
            raise ValueError("BIO training requires flat, nonoverlapping gold spans")
        tags[span.start] = "B-" + span.label
        for i in range(span.start + 1, span.end):
            tags[i] = "I-" + span.label
    return tags


@dataclass
class Evidence:
    spans: tuple[Span, ...]
    # HF backbones retain first-subtoken WORD logits for calibration and span scoring.
    logits: np.ndarray | None = None
    tag_names: tuple[str, ...] = ()
    temperature: float = 1.0
    confidence_available: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_logits(cls, logits, tag_names, temperature=1.0):
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim != 2 or logits.shape[1] != len(tag_names) or not np.isfinite(logits).all():
            raise ValueError("Expected finite [n_words, n_tags] logits")
        probs = softmax(logits, temperature)
        ids = np.argmax(logits, axis=-1)
        spans = tags_to_spans([tag_names[i] for i in ids], probs[np.arange(len(ids)), ids])
        return cls(spans, logits, tuple(tag_names), temperature)

    def with_temperature(self, temperature):
        if self.logits is None:
            raise ValueError("Temperature scaling requires unnormalized token logits")
        return Evidence.from_logits(self.logits, self.tag_names, temperature)

    def scores_for_boundary(self, start, end, labels):
        """Score each type at one fixed boundary; no pooling between occurrences.

        For BIO backbones, a type score is the arithmetic mean of the B/I
        probabilities along that candidate path. O is the mean O probability.
        These are path scores, not normalized probabilities of complete spans.
        Native span backends use observed confidence and assign absent mass to O.
        """
        if self.logits is not None:
            probs = softmax(self.logits[start:end], self.temperature)
            index = {tag: i for i, tag in enumerate(self.tag_names)}
            result = {"O": float(probs[:, index["O"]].mean())}
            for label in labels:
                first = probs[0, index["B-" + label]]
                rest = probs[1:, index["I-" + label]].sum()
                result[label] = float((first + rest) / (end - start))
            return result
        matches = [s for s in self.spans if s.start == start and s.end == end]
        result = {label: 0.0 for label in labels}
        for span in matches:
            result[span.label] = max(result.get(span.label, 0.0), span.score)
        result["O"] = max(0.0, 1.0 - max(result.values(), default=0.0))
        return result


def validate_evidence(query: Query, evidence: Evidence, labels):
    for span in evidence.spans:
        if span.end > len(query.tokens) or span.label not in labels:
            raise ValueError(f"Invalid prediction for {query.id}: {span}")
    if len({s.key() for s in evidence.spans}) != len(evidence.spans):
        raise ValueError(f"Duplicate predictions for {query.id}")
    if evidence.logits is not None and len(evidence.logits) != len(query.tokens):
        raise ValueError(f"Word/logit alignment mismatch for {query.id}")
