"""Write the committed, compact global table: every land H3 res-6 cell and its split.

    python scripts/export_compact.py --out outputs/global

Columns (decode with earthprosp.io.read_compact):
  h3 (uint64), p_prospective (uint16, per-mille), share_<class> (uint8, /255, the split
  conditional on a deposit), top_class (uint8 index into commodities.CLASSES),
  composition_trust (uint8 %), known (0 none, 1 producer cell, 2 world-class deposit cell)
"""

import argparse
import os
import sys

import h3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from earthprosp import commodities as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/global")
    a = ap.parse_args()
    df = pd.read_parquet(os.path.join(a.out, "global_split_res6.parquet"))
    q = pd.DataFrame({"h3": np.array([h3.str_to_int(h) for h in df.h3], dtype=np.uint64)})
    q["p_prospective"] = np.round(df.p_prospective.to_numpy() * 1000).astype(np.uint16)
    m = df[[f"p_{c}" for c in C.CLASSES]].to_numpy()
    share = m / np.maximum(m.sum(1, keepdims=True), 1e-9)
    for i, c in enumerate(C.CLASSES):
        q[f"share_{c}"] = np.round(share[:, i] * 255).astype(np.uint8)
    q["top_class"] = df.top_class.cat.codes.astype(np.uint8)
    q["composition_trust"] = np.round(df.composition_trust.to_numpy() * 100).astype(np.uint8)
    q["known"] = (df.known_positive.astype(np.uint8) + df.world_class_known.astype(np.uint8)).astype(np.uint8)
    path = os.path.join(a.out, "earthprosp_global_res6.parquet")
    q.to_parquet(path, compression="zstd", compression_level=19, index=False)
    print(f"{path}: {len(q):,} cells, {os.path.getsize(path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
