# utils.py
"""
Utility functions for CoCoNat
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Tuple, Dict
import numpy as np


def auto_threshold(
    queries_with_entities: List[Tuple[int, List[dict]]],
    kappa: float = 2.0,
) -> Tuple[float, float, float]:
    """
    Derive a data‑dependent confidence threshold.
    """
    confidences: list[float] = [ent["score"]
                                for _, ents in queries_with_entities
                                for ent in ents]
    arr = np.asarray(confidences)
    mean_conf = float(arr.mean())
    std_conf = float(arr.std())
    threshold = mean_conf - kappa * std_conf
    return threshold, mean_conf, std_conf


def subdivide_groups(
    groups: Dict[str, List[int]],
    max_size: int = 10,
) -> Dict[str, List[int]]:
    """
    Split oversized groups into roughly equal sub‑groups.
    """
    new_groups: dict[str, List[int]] = {}
    for entity, idxs in groups.items():
        n = len(idxs)
        if n <= max_size:
            new_groups[entity] = idxs
            continue

        num_parts = math.ceil(n / max_size)
        part_size = math.ceil(n / num_parts)
        for i in range(num_parts):
            start, end = i * part_size, (i + 1) * part_size
            new_groups[f"{entity}__SUB__{i + 1}"] = idxs[start:end]
    return new_groups


def gradual_ordering(records: List[Tuple[int, str, float]]) -> List[Tuple[int, str, float]]:
    """
    Arrange queries so that conflicting anchors are far apart.
    """
    n = len(records)

    # 1) Group by label
    per_label: defaultdict[str, List[Tuple[int, str, float]]] = defaultdict(list)
    for rec in records:
        per_label[rec[1]].append(rec)

    # 2) Pick one anchor (highest‑confidence) per label
    anchors: dict[str, Tuple[int, str, float]] = {
        lbl: max(recs, key=lambda r: r[2]) for lbl, recs in per_label.items()
    }
    labels_sorted = sorted(anchors, key=lambda L: anchors[L][2])

    if len(labels_sorted) == 1:
        anchor_pos = [0]
    else:
        anchor_pos = [int(i * (n - 1) / (len(labels_sorted) - 1))
                      for i in range(len(labels_sorted))]

    result: list | List[Tuple[int, str, float]] = [None] * n

    for lbl, pos in zip(labels_sorted, anchor_pos):
        result[pos] = anchors[lbl]

    # 3) Fill remaining slots label‑wise, alternating outward
    for lbl, pos in zip(labels_sorted, anchor_pos):
        others = [r for r in per_label[lbl] if r is not anchors[lbl]]
        others.sort(key=lambda r: r[2], reverse=True)

        if pos == 0:
            slots = range(1, n)
        elif pos == n - 1:
            slots = range(pos - 1, -1, -1)
        else:
            slots = list(range(pos + 1, n)) + list(range(pos - 1, -1, -1))
        free = (i for i in slots if result[i] is None)
        for s, item in zip(free, others):
            result[s] = item

    return [r for r in result if r is not None]
