"""Selective second-pass inference without access to gold labels."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import time

import numpy as np

from .config import PipelineConfig
from .schema import Evidence, Query, Span, validate_evidence


@dataclass
class Detection:
    hard_queries: set[str]
    hard_spans: set[tuple]
    low_spans: set[tuple]
    inconsistent_forms: set[str]
    hard_forms: set[str]
    mean: float | None
    std: float | None
    threshold: float | None
    ratios: dict[str, float]


def mention_key(query_id, span):
    return (query_id, span.start, span.end, span.label)


def normalize_surface(surface, case_sensitive=True):
    return surface if case_sensitive else surface.casefold()


def detect(queries, first, config: PipelineConfig):
    records, counts = [], defaultdict(Counter)
    for q in queries:
        for span in first[q.id].spans:
            form = normalize_surface(q.surface(span), config.case_sensitive)
            records.append((q.id, span, form))
            counts[form][span.label] += 1
    confidence = np.asarray([s.score for _, s, _ in records], dtype=float)
    mean = float(confidence.mean()) if len(confidence) else None
    std = float(confidence.std(ddof=0)) if len(confidence) else None
    threshold = mean - config.kappa * std if mean is not None else None
    ratios, inconsistent = {}, set()
    for form, labels in counts.items():
        frequencies = sorted(labels.values(), reverse=True)
        ratios[form] = frequencies[0] / sum(frequencies[:2])
        if len(frequencies) >= 2 and ratios[form] <= config.delta:
            inconsistent.add(form)
    low, hard, forms, hard_q = set(), set(), set(), set()
    for qid, span, form in records:
        key = mention_key(qid, span)
        # No low-confidence flag when all confidences are identical.
        is_low = std is not None and std > 1e-12 and span.score <= threshold
        if is_low:
            low.add(key)
        active = ((config.detector in {"confidence", "both"} and is_low)
                  or (config.detector in {"inconsistency", "both"} and form in inconsistent))
        if active:
            hard.add(key)
            forms.add(form)
            hard_q.add(qid)
    return Detection(hard_q, hard, low, inconsistent, forms, mean, std, threshold, ratios)


def gradual_ordering(records):
    """Place top two label anchors at the ends, then fill nearest free slots.

    records are (query_id, representative_label, confidence). Ties are resolved
    by query ID, so output is independent of hash iteration or platform.
    """
    records = list(records)
    if len({r[0] for r in records}) != len(records):
        raise ValueError("Each query can occur only once within a group")
    per_label = defaultdict(list)
    for rec in records:
        per_label[rec[1]].append(rec)
    for rows in per_label.values():
        rows.sort(key=lambda r: (-r[2], r[0]))
    labels = sorted(per_label, key=lambda label: (-per_label[label][0][2], label))
    n = len(records)
    if not n:
        return []
    if len(labels) == 1:
        return per_label[labels[0]]
    positions = [0, n - 1]
    while len(positions) < len(labels):
        free = [i for i in range(n) if i not in positions]
        positions.append(max(free, key=lambda i: (min(abs(i - p) for p in positions), -i)))
    result = [None] * n
    for label, pos in zip(labels, positions):
        result[pos] = per_label[label][0]
    for label, pos in zip(labels, positions):
        free = sorted((i for i, row in enumerate(result) if row is None),
                      key=lambda i: (abs(i - pos), i))
        for i, row in zip(free, per_label[label][1:]):
            result[i] = row
    assert all(row is not None for row in result)
    return result


def _representative(q, evidence, hard, form, case_sensitive):
    matching = [s for s in evidence.spans
                if form is not None and normalize_surface(q.surface(s), case_sensitive) == form]
    if not matching:
        matching = [s for s in evidence.spans if mention_key(q.id, s) in hard.hard_spans]
    candidates = matching or list(evidence.spans)
    if not candidates:
        return (q.id, "O", 0.0)
    span = max(candidates, key=lambda s: (s.score, -s.start, s.label))
    return (q.id, span.label, span.score)


def make_groups(queries, first, detection, config, backend):
    hard_qs = [q for q in queries if q.id in detection.hard_queries]
    qmap = {q.id: q for q in hard_qs}
    groups = []
    cluster_info = {}
    representatives = {}
    if config.grouping == "entity":
        lookup = defaultdict(list)
        for q in hard_qs:
            forms = {normalize_surface(q.surface(s), config.case_sensitive) for s in first[q.id].spans}
            for form in sorted(forms & detection.hard_forms):
                lookup[form].append(q.id)
        groups = [(form, ids) for form, ids in sorted(lookup.items())]
    elif len(hard_qs) >= 2:
        from sklearn.cluster import KMeans
        from sklearn.mixture import GaussianMixture

        embedding = np.asarray(backend.embed(hard_qs), dtype=float)
        if embedding.ndim != 2 or len(embedding) != len(hard_qs) or not np.isfinite(embedding).all():
            raise ValueError("Backend embeddings must be finite [n_queries, hidden_size]")
        requested = config.n_clusters or max(1, len(detection.hard_forms))
        k = min(requested, len(hard_qs), len(np.unique(embedding, axis=0)))
        cluster_info = dict(requested_clusters=requested, effective_clusters=k,
                            embedding_source=backend.embedding_description)
        if config.grouping == "kmeans":
            assigned = KMeans(n_clusters=k, n_init=10, random_state=config.seed).fit_predict(embedding)
            memberships = [[int(a)] for a in assigned]
        else:
            probs = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-5,
                                    random_state=config.seed).fit(embedding).predict_proba(embedding)
            memberships = [np.flatnonzero(p >= config.gmm_threshold).tolist() or [int(p.argmax())]
                           for p in probs]
        groups = [(None, [q.id for q, member in zip(hard_qs, memberships) if i in member])
                  for i in range(k)]
        mentions = {q.id: [s for s in first[q.id].spans
                           if mention_key(q.id, s) in detection.hard_spans] for q in hard_qs}
        mention_vectors = backend.embed_mentions(hard_qs, mentions)
        for group_index, (_, members) in enumerate(groups):
            indices = [i for i, q in enumerate(hard_qs) if q.id in members]
            if not indices:
                continue
            centroid = embedding[indices].mean(axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-12)
            for qid in members:
                chosen = max(mentions[qid], key=lambda s: (
                    float(np.dot(centroid, mention_vectors[(qid, s.start, s.end)])), s.score, -s.start))
                representatives[(group_index, qid)] = (qid, chosen.label, chosen.score)
        cluster_info["representative"] = "hard_mention_nearest_group_centroid_cosine"
    rng = np.random.default_rng(config.seed)
    planned, traces = [], []
    for index, (form, ids) in enumerate(groups):
        # Subdivide deterministically before ordering, retaining the original size bound.
        parts = np.array_split(ids, int(np.ceil(len(ids) / config.max_group_size))) if ids else []
        for part_id, part in enumerate(parts):
            rows = [representatives.get((index, qid)) or
                    _representative(qmap[qid], first[qid], detection, form, config.case_sensitive)
                    for qid in part.tolist()]
            if not rows:
                continue
            anchor = min(rows, key=lambda row: (-row[2], row[0]))[0]
            if config.ordering == "gradual":
                rows = gradual_ordering(rows)
            elif config.ordering == "random":
                rng.shuffle(rows)
            ordered_ids = [row[0] for row in rows]
            selected = [anchor] if backend.group_fits([qmap[anchor]]) else []
            if selected:
                for qid in ordered_ids:
                    if qid != anchor and backend.group_fits([qmap[i] for i in selected + [qid]]):
                        selected.append(qid)
            kept = [qid for qid in ordered_ids if qid in selected]
            processed = len(kept) >= 2
            traces.append(dict(group_id=f"g{index}.{part_id}", form=form, anchor=anchor,
                               requested=ordered_ids, included=kept if processed else [],
                               omitted=[i for i in ordered_ids if i not in kept],
                               reason="processed" if processed else "singleton_or_token_budget"))
            if processed:
                planned.append([qmap[qid] for qid in kept])
    return planned, traces, cluster_info


def aggregate(evidences, labels, policy):
    """Resolve a union of candidate span boundaries for ONE query occurrence."""
    if len(evidences) == 1:
        return evidences[0]
    boundaries = sorted({(s.start, s.end) for e in evidences for s in e.spans})
    candidates = []
    for start, end in boundaries:
        vectors = [e.scores_for_boundary(start, end, labels) for e in evidences]
        scores = {label: float(np.mean([v[label] for v in vectors])) if policy == "mean"
                  else max(v[label] for v in vectors) for label in ("O", *labels)}
        label = min(scores, key=lambda lab: (-scores[lab], lab != "O", lab))
        if label != "O":
            candidates.append(Span(start, end, label, scores[label]))
    # Flat NER: deterministic highest-confidence selection of overlapping candidates.
    kept = []
    for span in sorted(candidates, key=lambda s: (-s.score, s.start, s.end, s.label)):
        if not any(span.start < other.end and other.start < span.end for other in kept):
            kept.append(span)
    return Evidence(tuple(sorted(kept)), metadata={"aggregation": policy,
                                                  "contexts": len(evidences)})


@dataclass
class RunResult:
    first: dict[str, Evidence]
    final: dict[str, Evidence]
    detection: Detection
    processed: set[str]
    groups: list[dict]
    timing: dict
    cluster_info: dict


def run_pipeline(queries: list[Query], backend, config: PipelineConfig,
                 first=None, temperature=1.0):
    if len({q.id for q in queries}) != len(queries):
        raise ValueError("Query IDs must be unique within a collection")
    backend.synchronize()
    start = time.perf_counter()
    cached = first is not None
    if first is None:
        first = backend.predict(queries)
    if set(first) != {q.id for q in queries}:
        raise ValueError("First-pass outputs do not cover the query collection exactly")
    if temperature != 1.0:
        first = {qid: ev.with_temperature(temperature) for qid, ev in first.items()}
    for q in queries:
        validate_evidence(q, first[q.id], backend.labels)
    backend.synchronize()
    after_first = time.perf_counter()
    detection = detect(queries, first, config)
    groups, traces, cluster_info = make_groups(queries, first, detection, config, backend)
    backend.synchronize()
    after_group = time.perf_counter()
    grouped = backend.predict_groups(groups) if groups else []
    if len(grouped) != len(groups):
        raise ValueError("Backend did not return one mapping per group")
    contexts = defaultdict(list)
    for group, output in zip(groups, grouped):
        if set(output) != {q.id for q in group}:
            raise ValueError("Group output IDs differ from input IDs")
        for q in group:
            evidence = output[q.id]
            if temperature != 1.0:
                evidence = evidence.with_temperature(temperature)
            validate_evidence(q, evidence, backend.labels)
            contexts[q.id].append(evidence)
    backend.synchronize()
    after_second = time.perf_counter()
    final = dict(first)
    for qid, evidence in contexts.items():
        if qid not in detection.hard_queries:
            raise AssertionError("Non-hard queries must never be updated")
        final[qid] = aggregate(evidence, backend.labels, config.aggregation)
    backend.synchronize()
    end = time.perf_counter()
    return RunResult(first, final, detection, set(contexts), traces, dict(
        first_seconds=after_first - start, grouping_seconds=after_group - after_first,
        second_seconds=after_second - after_group, aggregation_seconds=end - after_second,
        total_seconds=end - start, cached_first_pass=cached), cluster_info)
