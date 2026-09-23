"""Hand-built per-cell features used by the baseline and as model covariates."""

from __future__ import annotations

import h3
import numpy as np

from .aef import CellBags


def bag_stats(cb: CellBags, chunk: int = 20000) -> np.ndarray:
    """Mean, std and 10/90th percentile of the pixel embeddings: (N, 256)."""
    out = np.zeros((len(cb.cells), 256), np.float32)
    for s in range(0, len(cb.cells), chunk):
        b = cb.bags[s:s + chunk].astype(np.float32)
        m = cb.mask[s:s + chunk]
        n = m.sum(1)
        mf = m[..., None]
        mean = (b * mf).sum(1) / n[:, None]
        std = np.sqrt((((b - mean[:, None]) ** 2) * mf).sum(1) / n[:, None])
        # Masked pixels -> +inf so they sort to the end; index by per-cell count.
        srt = np.sort(np.where(mf, b, np.inf), axis=1)
        rows = np.arange(len(b))
        p10 = srt[rows, np.floor(0.1 * (n - 1)).astype(int)]
        p90 = srt[rows, np.ceil(0.9 * (n - 1)).astype(int)]
        out[s:s + chunk] = np.concatenate([mean, std, p10, p90], 1)
    return out


def neighbourhood_context(cells: np.ndarray, emb: np.ndarray, rings=(2, 6)) -> np.ndarray:
    """Mean embedding over k-rings around each cell: (N, 64 * len(rings)).

    Mineral systems are regional (a porphyry belt, a greenstone belt, a basin margin),
    so the geology tens of km away is informative. Ring 2 ~ 10 km, ring 6 ~ 25 km
    radius at res 6. Cells missing from `cells` (ocean, outside AOI) are skipped.
    """
    lookup = {c: i for i, c in enumerate(cells)}
    out = np.zeros((len(cells), emb.shape[1] * len(rings)), np.float32)
    for j, k in enumerate(rings):
        for i, c in enumerate(cells):
            idx = [lookup[n] for n in h3.grid_disk(c, k) if n in lookup]
            v = emb[idx].mean(0)
            out[i, j * emb.shape[1]:(j + 1) * emb.shape[1]] = v / (np.linalg.norm(v) + 1e-8)
    return out
