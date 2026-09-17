"""Fit a scalar temperature on validation WORD-level NLL, then evaluate entity ECE."""

from __future__ import annotations

import math

import numpy as np

from .schema import spans_to_tags


def fit_temperature(examples, predictions, split="validation", bounds=(0.05, 20.0)):
    if split != "validation":
        raise ValueError("Temperature fitting is restricted to the validation split")
    logits, targets = [], []
    for ex in examples:
        ev = predictions[ex.query.id]
        if ev.logits is None:
            raise ValueError("Temperature fitting requires raw word logits")
        index = {tag: i for i, tag in enumerate(ev.tag_names)}
        tags = spans_to_tags(len(ex.query.tokens), ex.gold)
        targets.extend(index[tag] for tag in tags)
        logits.append(ev.logits)
    values = np.concatenate(logits).astype(np.float64)
    targets = np.asarray(targets, dtype=int)

    def nll(log_temperature):
        scaled = values / math.exp(log_temperature)
        maximum = scaled.max(axis=1)
        logsum = maximum + np.log(np.exp(scaled - maximum[:, None]).sum(axis=1))
        return float(np.mean(logsum - scaled[np.arange(len(targets)), targets]))

    # Golden-section search avoids an additional optimization dependency.
    lo, hi = map(math.log, bounds)
    if not lo < hi:
        raise ValueError("Temperature bounds must be ordered and positive")
    ratio = (math.sqrt(5) - 1) / 2
    c, d = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fc, fd = nll(c), nll(d)
    for _ in range(80):
        if fc <= fd:
            hi, d, fd = d, c, fc
            c = hi - ratio * (hi - lo)
            fc = nll(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + ratio * (hi - lo)
            fd = nll(d)
    candidates = [(nll(0.0), 0.0), (fc, c), (fd, d)]
    calibrated_nll, best = min(candidates)
    temperature = math.exp(best)
    return dict(temperature=temperature, raw_nll=nll(0.0), calibrated_nll=calibrated_nll,
                objective="validation_word_NLL_first_subtoken_including_O",
                fit_split=split, n_words=len(targets), bounds=list(bounds),
                near_bound=temperature < bounds[0] * 1.01 or temperature > bounds[1] * 0.99)
