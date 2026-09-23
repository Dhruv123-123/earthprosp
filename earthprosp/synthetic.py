"""Synthetic cell bags with a planted mineral signal, for tests and smoke runs."""

from __future__ import annotations

import h3
import numpy as np

from . import commodities as C
from .aef import EMB_DIM, CellBags


def make(n_side: int = 40, bag: int = 16, pos_rate: float = 0.05, label_rate: float = 0.5,
         strength: float = 2.5, seed: int = 0):
    """Cells on a real H3 patch. A subset of cells are 'prospective'; their pixels carry
    a class-specific direction in embedding space. Only `label_rate` of prospective
    cells are labeled (PU setting)."""
    rng = np.random.default_rng(seed)
    origin = h3.latlng_to_cell(40.0, -117.0, 6)
    cells = np.array(sorted(h3.grid_disk(origin, n_side // 2)), dtype=object)
    n, K = len(cells), C.N_CLASSES
    signatures = rng.normal(size=(K, EMB_DIM))
    signatures /= np.linalg.norm(signatures, axis=1, keepdims=True)

    truly_pos = rng.random(n) < pos_rate
    cls = rng.integers(0, 4, n)  # only the first few classes appear
    bags = rng.normal(size=(n, bag, EMB_DIM)).astype(np.float32)
    anomalous = rng.random((n, bag)) < 0.3
    bags += (truly_pos[:, None, None] * anomalous[..., None]) * strength * signatures[cls][:, None, :]
    bags /= np.linalg.norm(bags, axis=-1, keepdims=True)
    mask = np.ones((n, bag), bool)
    mask[:, bag - bag // 4:] = rng.random((n, bag // 4)) < 0.5

    labeled = truly_pos & (rng.random(n) < label_rate)
    comp = np.zeros((n, K), np.float32)
    comp[np.arange(n), cls] = 1.0
    comp[~labeled] = 0
    return {
        "cb": CellBags(cells, bags.astype(np.float16), mask),
        "positive": labeled,
        "truly_positive": truly_pos,
        "conf": labeled.astype(np.float32),
        "composition": comp,
        "cls": cls,
    }
