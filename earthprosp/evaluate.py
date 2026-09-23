"""Metrics suited to positive-unlabeled, spatially clustered prospectivity labels.

We never have true negatives, so the headline numbers are *capture* metrics used in
mineral prospectivity mapping: what fraction of held-out known deposits falls inside
the top x% of the area ranked by the model (a "success-rate" curve). A random map
captures x% at x% area.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def capture_curve(score: np.ndarray, positive: np.ndarray, fracs=(0.01, 0.05, 0.1, 0.2)) -> dict:
    order = np.argsort(-score)
    pos_sorted = positive[order]
    cum = np.cumsum(pos_sorted) / max(pos_sorted.sum(), 1)
    n = len(score)
    out = {f"capture@{int(f * 100)}%": float(cum[max(int(np.ceil(f * n)) - 1, 0)]) for f in fracs}
    out["capture_auc"] = float(cum.mean())  # 0.5 = random, 1 = perfect
    out["roc_auc_pu"] = float(roc_auc_score(positive, score)) if 0 < positive.sum() < n else float("nan")
    return out


def composition_metrics(pred_comp: np.ndarray, target: np.ndarray, weight: np.ndarray | None = None) -> dict:
    """pred_comp, target: (n, K) rows summing to 1, evaluated on labeled positive cells."""
    w = np.ones(len(target)) if weight is None else weight
    top1 = (pred_comp.argmax(1) == target.argmax(1)).astype(float)
    ce = -(target * np.log(np.clip(pred_comp, 1e-9, 1))).sum(1)
    # Probability mass on the classes actually present (1 = all mass on correct classes).
    mass = (pred_comp * (target > 0)).sum(1)
    return {
        "top1_acc": float((top1 * w).sum() / w.sum()),
        "cross_entropy": float((ce * w).sum() / w.sum()),
        "mass_on_present": float((mass * w).sum() / w.sum()),
    }
