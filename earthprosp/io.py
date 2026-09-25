"""Read the compact global prospectivity table (outputs/global/earthprosp_global_res6.parquet)."""

from __future__ import annotations

import h3
import numpy as np
import pandas as pd

from . import commodities as C


def read_compact(path: str = "outputs/global/earthprosp_global_res6.parquet", with_latlng: bool = False) -> pd.DataFrame:
    """Decode to floats: h3 (str), p_prospective, p_none, p_<class> (marginal), top_class."""
    q = pd.read_parquet(path)
    out = pd.DataFrame({"h3": [h3.int_to_str(int(v)) for v in q.h3]})
    p = q.p_prospective.to_numpy() / 1000.0
    out["p_prospective"], out["p_none"] = p, 1 - p
    share = q[[f"share_{c}" for c in C.CLASSES]].to_numpy(np.float64)
    share = share / np.maximum(share.sum(1, keepdims=True), 1)
    for i, c in enumerate(C.CLASSES):
        out[f"p_{c}"] = p * share[:, i]
    out["top_class"] = np.array(C.CLASSES)[q.top_class.to_numpy()]
    out["composition_trust"] = q.composition_trust.to_numpy() / 100.0
    out["known"] = q.known.to_numpy()
    if with_latlng:
        ll = np.array([h3.cell_to_latlng(h) for h in out.h3])
        out["lat"], out["lon"] = ll[:, 0], ll[:, 1]
    return out
