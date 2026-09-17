"""Backend interface shared by native encoders, span models, and external runners."""

from __future__ import annotations

import hashlib

import numpy as np

from ..schema import Evidence, Query, Span


class Backend:
    labels: tuple[str, ...]
    name = "backend"
    synthetic = False
    max_length = 512
    embedding_description = "hashed unigram bag-of-words (diagnostic fallback)"
    timing_scope = "resident_model_inference_including_tokenization"

    def predict(self, queries):
        raise NotImplementedError

    def synchronize(self):
        pass

    def group_fits(self, queries):
        return sum(len(q.tokens) for q in queries) + len(queries) - 1 <= self.max_length

    def predict_groups(self, groups):
        """Generic textual context wrapper for native span models."""
        merged, mappings = [], []
        for i, group in enumerate(groups):
            tokens, offsets = [], []
            for q in group:
                if tokens:
                    tokens.append(".")
                offsets.append((q, len(tokens)))
                tokens.extend(q.tokens)
            merged.append(Query(f"context_{i}", tuple(tokens)))
            mappings.append(offsets)
        predictions = self.predict(merged)
        outputs = []
        for merged_q, offsets in zip(merged, mappings):
            evidence = predictions[merged_q.id]
            output = {}
            for q, offset in offsets:
                spans = tuple(Span(s.start - offset, s.end - offset, s.label, s.score)
                              for s in evidence.spans
                              if offset <= s.start < s.end <= offset + len(q.tokens))
                output[q.id] = Evidence(spans, confidence_available=evidence.confidence_available)
            outputs.append(output)
        return outputs

    def embed(self, queries):
        # This fallback is explicitly identified in every output manifest.
        embeddings = np.zeros((len(queries), 256), dtype=float)
        for i, q in enumerate(queries):
            for word in q.tokens:
                digest = hashlib.sha256(word.casefold().encode()).digest()
                embeddings[i, int.from_bytes(digest[:4], "big") % 256] += 1.0
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norm, 1e-12)

    def describe(self):
        return dict(name=self.name, synthetic=self.synthetic, max_length=self.max_length,
                    timing_scope=self.timing_scope)

    def embed_mentions(self, queries, mentions):
        targets, keys = [], []
        for q in queries:
            for span in mentions[q.id]:
                targets.append(Query(f"{q.id}:{span.start}:{span.end}", q.tokens[span.start:span.end]))
                keys.append((q.id, span.start, span.end))
        return dict(zip(keys, self.embed(targets)))


def char_span_to_word(query, start, end):
    """Require exact token boundaries; never snap a malformed prediction to gold."""
    offsets = query.char_offsets
    starts = {a: i for i, (a, _) in enumerate(offsets)}
    ends = {b: i + 1 for i, (_, b) in enumerate(offsets)}
    if start not in starts or end not in ends or starts[start] >= ends[end]:
        return None
    return starts[start], ends[end]
