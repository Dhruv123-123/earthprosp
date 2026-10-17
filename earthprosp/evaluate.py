"""Metrics suited to positive-unlabeled, spatially clustered prospectivity labels.

We never have true negatives, so the headline numbers are *capture* metrics used in
mineral prospectivity mapping: what fraction of held-out known deposits falls inside
the top x% of the area ranked by the model (a "success-rate" curve). A random map
captures x% at x% area.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def capture_curve(score: np.ndarray, positive: np.ndarray, fracs=(0.01, 0.05, 0.1, 0.2),
                  weight: np.ndarray | None = None) -> dict:
    """Share of positives in the top f of cells by score. With `weight` (e.g. contained metal
    per cell), the share of total weight instead: a map that finds the giants scores higher
    than one that finds many small showings."""
    order = np.argsort(-score)
    pos_sorted = (positive if weight is None else np.where(positive, weight, 0.0))[order].astype(np.float64)
    cum = np.cumsum(pos_sorted) / max(pos_sorted.sum(), 1e-12)
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


def enrichment(score: np.ndarray, positive: np.ndarray, frac: float = 0.05) -> float:
    """Deposit density in the top `frac` of area relative to the average (1 = random)."""
    k = max(int(np.ceil(frac * len(score))), 1)
    top = np.argpartition(-score, k - 1)[:k]
    base = positive.mean()
    return float(positive[top].mean() / base) if base > 0 else float("nan")


def block_bootstrap_ci(score: np.ndarray, positive: np.ndarray, blocks: np.ndarray, metric: str = "capture_auc",
                       n_boot: int = 200, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Point estimate and (1 - alpha) CI of a capture metric, resampling whole spatial blocks.

    Deposits are clustered, so resampling cells understates the uncertainty; resampling
    blocks (e.g. H3 res-2 parents) keeps each cluster together."""
    rng = np.random.default_rng(seed)
    codes, uniq = _factorize(blocks)
    members = [np.where(codes == i)[0] for i in range(len(uniq))]
    point = capture_curve(score, positive)[metric]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        if 0 < positive[idx].sum() < len(idx):
            vals.append(capture_curve(score[idx], positive[idx])[metric])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if vals else (np.nan, np.nan)
    return float(point), float(lo), float(hi)


def _factorize(x):
    import pandas as pd
    codes, uniq = pd.factorize(np.asarray(x, dtype=object))
    return codes, uniq
