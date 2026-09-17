"""Exact entity evaluation, explicitly denominated transition and calibration analyses."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .pipeline import mention_key, normalize_surface


def entity_metrics(examples, predictions):
    tp = fp = fn = 0
    by_label = defaultdict(lambda: Counter(tp=0, fp=0, fn=0))
    if set(predictions) != {e.query.id for e in examples}:
        raise ValueError("Evaluation requires exactly one output for every query")
    for ex in examples:
        gold = {s.key() for s in ex.gold}
        pred = {s.key() for s in predictions[ex.query.id].spans}
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
        for name, spans in (("tp", pred & gold), ("fp", pred - gold), ("fn", gold - pred)):
            for _, _, label in spans:
                by_label[label][name] += 1

    def scores(a, b, c):
        precision = a / (a + b) if a + b else 0.0
        recall = a / (a + c) if a + c else 0.0
        f1 = 2 * a / (2 * a + b + c) if 2 * a + b + c else 0.0
        return dict(tp=a, fp=b, fn=c, precision=precision, recall=recall, f1=f1)

    result = scores(tp, fp, fn)
    result["per_label"] = {k: scores(v["tp"], v["fp"], v["fn"]) for k, v in sorted(by_label.items())}
    return result


def entity_ece(examples, predictions, n_bins=15):
    """Equal-frequency ECE over predicted entities, not gold entities or tokens."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    observations = []
    for ex in examples:
        ev = predictions[ex.query.id]
        if not ev.confidence_available:
            return dict(ece=None, bins=[], n=0, reason="backend_has_no_entity_confidence")
        gold = {s.key() for s in ex.gold}
        observations.extend((s.score, float(s.key() in gold)) for s in ev.spans)
    if not observations:
        return dict(ece=None, bins=[], n=0, reason="no_predicted_entities")
    arr = np.asarray(sorted(observations, key=lambda x: x[0]), dtype=float)
    bins, ece = [], 0.0
    for group in np.array_split(arr, min(n_bins, len(arr))):
        conf, accuracy = group.mean(axis=0)
        gap = abs(conf - accuracy)
        ece += len(group) * gap / len(arr)
        bins.append(dict(n=len(group), confidence=float(conf), accuracy=float(accuracy), gap=float(gap)))
    return dict(ece=float(ece), bins=bins, n=len(arr))


def transitions(examples, result):
    """Paper-defined per-span maximum-overlap alignment, with gold-blind pairs.

    The table denominator is FIRST-PASS predicted spans in reprocessed queries.
    Newly predicted spans with no first-pass counterpart are reported separately.
    A one-to-one greedy sensitivity diagnostic is also emitted. Neither diagnostic
    substitutes for exact entity P/R/F1.
    """
    counts = Counter({key: 0 for key in ("W_C", "C_C", "W_W", "C_W")})
    one_to_one = counts.copy()
    gold_counts = counts.copy()
    cases, new_only = [], Counter(total=0, correct=0, wrong=0)
    for ex in examples:
        qid = ex.query.id
        if qid not in result.processed:
            continue
        before, after = result.first[qid].spans, result.final[qid].spans
        gold = {s.key() for s in ex.gold}
        edges = []
        for i, a in enumerate(before):
            for j, b in enumerate(after):
                overlap = max(0, min(a.end, b.end) - max(a.start, b.start))
                if overlap:
                    exact_boundary = a.start == b.start and a.end == b.end
                    edges.append((-overlap, -int(exact_boundary), i, j))
        pairs, used = {}, set()
        for _, _, i, j in sorted(edges):
            if i not in pairs and j not in used:
                pairs[i] = j
                used.add(j)
        paper_used = set()
        for i, a in enumerate(before):
            candidates = sorted(edge for edge in edges if edge[2] == i)
            b = after[candidates[0][3]] if candidates else None
            if candidates:
                paper_used.add(candidates[0][3])
            status = ("C" if a.key() in gold else "W") + "_" + (
                "C" if b is not None and b.key() in gold else "W")
            counts[status] += 1
            one = after[pairs[i]] if i in pairs else None
            one_to_one[("C" if a.key() in gold else "W") + "_" + (
                "C" if one is not None and one.key() in gold else "W")] += 1
            cases.append(dict(query_id=qid, text=ex.query.text, tokens=list(ex.query.tokens),
                              transition=status, before=a.to_dict(), after=b.to_dict() if b else None,
                              gold=[s.to_dict() for s in ex.gold],
                              hard=mention_key(qid, a) in result.detection.hard_spans))
        for j, b in enumerate(after):
            if j not in paper_used:
                new_only["total"] += 1
                new_only["correct" if b.key() in gold else "wrong"] += 1
        before_keys, after_keys = {s.key() for s in before}, {s.key() for s in after}
        for key in gold:
            gold_counts[("C" if key in before_keys else "W") + "_" +
                        ("C" if key in after_keys else "W")] += 1
    total = sum(counts.values())
    return dict(counts=dict(counts), n=total,
                fractions={key: value / total if total else None for key, value in counts.items()},
                net_corrections=counts["W_C"] - counts["C_W"],
                net_correction_fraction=(counts["W_C"] - counts["C_W"]) / total if total else None,
                correction_corruption_ratio=counts["W_C"] / counts["C_W"] if counts["C_W"] else None,
                denominator="first_pass_spans_in_reprocessed_queries",
                matching="independent_maximum_overlap_per_first_span_gold_blind",
                one_to_one_greedy_counts=dict(one_to_one),
                gold_anchored_counts=dict(gold_counts), new_only_second_spans=dict(new_only)), cases


def dataset_characteristics(examples, result, case_sensitive=True):
    forms = defaultdict(set)
    for ex in examples:
        for span in result.first[ex.query.id].spans:
            forms[normalize_surface(ex.query.surface(span), case_sensitive)].add(ex.query.id)
    repeated = total = 0
    for ex in examples:
        for span in result.first[ex.query.id].spans:
            total += 1
            if mention_key(ex.query.id, span) in result.detection.hard_spans:
                repeated += len(forms[normalize_surface(ex.query.surface(span), case_sensitive)]) > 1
    base = entity_metrics(examples, result.first)["f1"]
    final = entity_metrics(examples, result.final)["f1"]
    hard = len(result.detection.hard_spans)
    return dict(n_predicted_spans=total, n_hard_spans=hard, n_repeated_hard=repeated,
                hard_fraction=hard / total if total else None,
                repeated_hard_fraction=repeated / hard if hard else None,
                first_f1=base, final_f1=final, error_headroom=1 - base, gain=final - base,
                headroom_definition="1 - exact-match micro F1; not an error probability")


def screening_observations(examples, result, dataset_name=""):
    rows = []
    for ex in examples:
        gold = {s.key() for s in ex.gold}
        n = len(ex.query.tokens)
        bucket = "01-15" if n <= 15 else "16-25" if n <= 25 else "26-40" if n <= 40 else "41+"
        for s in result.first[ex.query.id].spans:
            rows.append(dict(dataset=dataset_name, query_id=ex.query.id, label=s.label,
                             start=s.start, end=s.end, sentence_tokens=n, length_bin=bucket,
                             confidence=s.score, low_flag=mention_key(ex.query.id, s) in result.detection.low_spans,
                             hard_flag=mention_key(ex.query.id, s) in result.detection.hard_spans,
                             wrong=s.key() not in gold))
    return rows


def strata_summary(observations, field):
    grouped = defaultdict(list)
    for row in observations:
        grouped[row[field]].append(row)
    result = []
    for name, rows in sorted(grouped.items()):
        flagged = [r for r in rows if r["low_flag"]]
        result.append(dict(stratum=name, n=len(rows), flagged_n=len(flagged),
                           mean_confidence=float(np.mean([r["confidence"] for r in rows])),
                           flagged_fraction=len(flagged) / len(rows),
                           error_among_flagged=float(np.mean([r["wrong"] for r in flagged])) if flagged else None))
    return result


def label_balanced_strata(observations, seed=42):
    """Equal-size sampling per (dataset, label), a documented frequency control."""
    groups = defaultdict(list)
    for row in observations:
        groups[(row["dataset"], row["label"])].append(row)
    if not groups:
        return []
    size = min(len(rows) for rows in groups.values())
    rng = np.random.default_rng(seed)
    balanced = []
    for key, rows in sorted(groups.items()):
        for i in rng.choice(len(rows), size=size, replace=False):
            balanced.append({**rows[int(i)], "dataset_label": ":".join(key)})
    return strata_summary(balanced, "dataset_label")


def jaccard(first, second):
    union = first | second
    return len(first & second) / len(union) if union else 1.0
