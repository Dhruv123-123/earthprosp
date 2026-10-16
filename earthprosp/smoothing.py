"""District-scale smoothing of cell scores on the H3 grid.

Ore deposits cluster in districts and belts tens of km across, while a cell score is
noisy at the 5 km scale. Averaging a score over a k-ring is a prior that "a good cell has
good neighbours". It uses only predictions, never labels, so applying it to held-out
cells in CV is leak-free: the test score of a cell is smoothed with the *predicted*
scores of its neighbours.
"""

from __future__ import annotations

import h3
import numpy as np


def ring_index(cells, k: int, chunk: int = 100_000) -> np.ndarray:
    """(n, ring_size) neighbour indices into `cells` (-1 where a neighbour is missing)."""
    lookup = {c: i for i, c in enumerate(cells)}
    width = 3 * k * (k + 1) + 1
    idx = np.full((len(cells), width), -1, np.int64)
    for s in range(0, len(cells), chunk):
        for i, c in enumerate(cells[s:s + chunk]):
            r = [lookup.get(n, -1) for n in h3.grid_disk(c, k)]
            idx[s + i, :len(r)] = r
    return idx


def ring_mean(values: np.ndarray, idx: np.ndarray, self_weight: float = 0.0) -> np.ndarray:
    """Mean of `values` over each cell's ring (missing neighbours skipped). With
    self_weight w the result is w * own value + (1 - w) * ring mean."""
    valid = idx >= 0
    v = np.where(valid, values[np.where(valid, idx, 0)], 0.0).sum(1) / np.maximum(valid.sum(1), 1)
    if self_weight:
        v = self_weight * values + (1 - self_weight) * v
    return v.astype(np.float32)
