"""Embedding change between two AlphaEarth years, per H3 cell.

AEF embeddings are unit vectors per pixel; the cell mean (renormalised) of two years
agrees closely where the land surface is stable, and drifts where it changed: new open
pits, tailings, clearing, urban growth, floods. For prospectivity this is a *leakage
diagnostic* more than a predictor. A model that scores cells highly because they changed
(that is, because a mine was built there) is mapping mining, not geology. The change
score lets the report measure how much of the top of the map is explained by recent
surface disturbance, and lets a run exclude it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_means(path: str) -> pd.DataFrame:
    z = np.load(path)
    return pd.DataFrame(z["mean"].astype(np.float32), index=pd.Index(z["cells"].astype(object), name="cell"))


def change(path_a: str, path_b: str) -> pd.DataFrame:
    """Cosine distance (0 = identical, 2 = opposite) per cell present in both years."""
    a, b = load_means(path_a), load_means(path_b)
    common = a.index.intersection(b.index)
    A = a.loc[common].to_numpy().copy()
    B = b.loc[common].to_numpy().copy()
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-8
    cos = (A * B).sum(1)
    return pd.DataFrame({"aef_change_cos": (1 - cos).astype(np.float32)}, index=common)


if __name__ == "__main__":
    import sys
    out = change(sys.argv[1], sys.argv[2])
    out.to_parquet(sys.argv[3])
    print(out.describe().T)
