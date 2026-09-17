"""Deterministic synthetic backend for tests only; it never reads gold labels."""

from __future__ import annotations

import numpy as np

from .base import Backend
from ..schema import Evidence


class ToyBackend(Backend):
    name = "toy-synthetic-NOT-A-PAPER-RESULT"
    synthetic = True

    def __init__(self, labels=("ORG", "PER", "LOC"), max_length=256, **kwargs):
        self.labels = tuple(labels)
        self.tags = ("O",) + tuple(tag for label in labels for tag in ("B-" + label, "I-" + label))
        self.max_length = max_length

    def _evidence(self, query, context):
        logits = np.full((len(query.tokens), len(self.tags)), -3.0)
        logits[:, 0] = 3.0
        for i, word in enumerate(query.tokens):
            label, confidence_logit = None, 3.2
            if word in {"Amazon", "Acme", "Orion"}:
                label = "ORG" if "revenue" in query.tokens or "company" in query.tokens else "PER"
                if "river" in query.tokens:
                    label = "LOC"
                elif len(context) > 1 and any("revenue" in q.tokens for q in context):
                    label = "ORG"
                confidence_logit = 1.4 if "arrived" in query.tokens else 3.2
            elif word in {"Jordan", "Samdory", "Takagi"}:
                label = "LOC" if "country" in query.tokens else "PER"
            elif word in {"London", "Paris"}:
                label = "LOC"
            if label in self.labels:
                logits[i, :] = -2.0
                logits[i, 0] = 0.1
                logits[i, self.tags.index("B-" + label)] = confidence_logit
        return Evidence.from_logits(logits, self.tags)

    def predict(self, queries):
        return {q.id: self._evidence(q, [q]) for q in queries}

    def predict_groups(self, groups):
        return [{q.id: self._evidence(q, group) for q in group} for group in groups]
