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


_CTX = {}


def _ring_index(args):
    """Padded (n, ring_size) neighbour index for cells[s:e]; -1 = missing neighbour."""
    s, e, k = args
    cells, lookup = _CTX["cells"], _CTX["lookup"]
    nb = [[lookup.get(n, -1) for n in h3.grid_disk(c, k)] for c in cells[s:e]]
    idx = np.full((len(nb), max(len(r) for r in nb)), -1, np.int64)
    for i, r in enumerate(nb):
        idx[i, :len(r)] = r
    return s, idx


def neighbourhood_context(cells: np.ndarray, emb: np.ndarray, rings=(2, 6), workers: int | None = None) -> np.ndarray:
    """Mean embedding over k-rings around each cell: (N, 64 * len(rings)).

    Mineral systems are regional (a porphyry belt, a greenstone belt, a basin margin),
    so the geology tens of km away is informative. Ring 2 ~ 10 km, ring 6 ~ 25 km
    radius at res 6. Cells missing from `cells` (ocean, outside AOI) are skipped.
    Neighbour lookups (pure Python) run in `workers` forked processes for large inputs.
    """
    import multiprocessing as mp
    import os
    _CTX["cells"], _CTX["lookup"] = cells, {c: i for i, c in enumerate(cells)}
    out = np.zeros((len(cells), emb.shape[1] * len(rings)), np.float32)
    emb = np.asarray(emb, np.float32)
    workers = workers or min(os.cpu_count() or 1, 8)
    chunk = 50000
    for j, k in enumerate(rings):
        jobs = [(s, min(s + chunk, len(cells)), k) for s in range(0, len(cells), chunk)]
        if workers > 1 and len(jobs) > 1 and "fork" in mp.get_all_start_methods():
            with mp.get_context("fork").Pool(workers) as pool:
                parts = pool.imap_unordered(_ring_index, jobs)
                for s, idx in parts:
                    out[s:s + len(idx), j * emb.shape[1]:(j + 1) * emb.shape[1]] = _ring_mean(emb, idx)
        else:
            for job in jobs:
                s, idx = _ring_index(job)
                out[s:s + len(idx), j * emb.shape[1]:(j + 1) * emb.shape[1]] = _ring_mean(emb, idx)
    _CTX.clear()
    return out


def _ring_mean(emb, idx):
    valid = idx >= 0
    v = (emb[np.where(valid, idx, 0)] * valid[..., None]).sum(1) / valid.sum(1, keepdims=True)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
