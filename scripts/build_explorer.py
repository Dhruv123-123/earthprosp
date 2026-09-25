"""Pack the res-4 global split into a single self-contained HTML explorer.

    python scripts/build_explorer.py --out outputs/global
"""

import argparse
import base64
import json
import os
import sys

import h3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from earthprosp import commodities as C  # noqa: E402

LABELS = {"Au": "Gold", "Ag": "Silver", "Cu": "Copper", "PbZn": "Lead-zinc", "Fe": "Iron", "Mn": "Manganese",
          "U": "Uranium", "WMoSn": "W-Mo-Sn", "NiCoPGECr": "Ni-Co-PGE-Cr", "LiBeTaNb": "Li-Be-Ta-Nb",
          "REE": "Rare earths", "Al": "Bauxite (Al)", "HgSbAs": "Hg-Sb-As", "BaF": "Barite-fluorite", "P": "Phosphate"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/global")
    ap.add_argument("--land", default="data/ne_110m_land.geojson")
    ap.add_argument("--skill", default="")
    a = ap.parse_args()
    g = pd.read_parquet(os.path.join(a.out, "global_split_res4.parquet"))
    ll = np.array([h3.cell_to_latlng(h) for h in g.h4])
    countries = sorted(g.country.unique())
    cidx = {c: i for i, c in enumerate(countries)}
    p = g.p_prospective.to_numpy()
    marg = g[[f"p_{c}" for c in C.CLASSES]].to_numpy()
    q = marg / np.maximum(marg.sum(1, keepdims=True), 1e-9)
    rec = np.zeros(len(g), dtype=[("lat", "<i2"), ("lon", "<i2"), ("p", "u1"), ("known", "u1"), ("c", "<u2"),
                                  *[(f"q{k}", "u1") for k in range(C.N_CLASSES)]])
    rec["lat"] = np.round(ll[:, 0] * 100)
    rec["lon"] = np.round(ll[:, 1] * 100)
    rec["p"] = np.round(np.clip(p, 0, 1) * 255)
    rec["known"] = np.clip(g.known.to_numpy(), 0, 255)
    rec["c"] = [cidx[c] for c in g.country]
    for k in range(C.N_CLASSES):
        rec[f"q{k}"] = np.round(q[:, k] * 255)
    data = base64.b64encode(rec.tobytes()).decode()

    land = json.load(open(a.land))
    rings = []
    for f in land["features"]:
        geom = f["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in polys:
            rings += [[[round(x, 1), round(y, 1)] for x, y in r] for r in poly]

    ct = pd.read_csv(os.path.join(a.out, "by_country.csv")).rename(columns={"Unnamed: 0": "country"})
    ct = ct[ct.cells >= 50]
    cols = ["country", "cells", "mean_p", "known_positive_cells", "p_prospective"] + [f"p_{c}" for c in C.CLASSES]
    table = json.loads(ct[cols].round(3).to_json(orient="records"))
    n6 = json.load(open(os.path.join(a.out, "run.json")))["n_cells"]

    html = open(os.path.join(os.path.dirname(__file__), "explorer_template.html")).read()
    for key, val in {
        "__CLASSES__": json.dumps(C.CLASSES), "__LABELS__": json.dumps([LABELS[c] for c in C.CLASSES]),
        "__COUNTRIES__": json.dumps(countries), "__COUNTRY_TABLE__": json.dumps(table, separators=(",", ":")),
        "__LAND__": json.dumps(rings, separators=(",", ":")), "__DATA__": data,
        "__NCELLS__": f"{n6:,}", "__SKILL__": a.skill,
    }.items():
        html = html.replace(key, val)
    path = os.path.join(a.out, "explorer.html")
    open(path, "w").write(html)
    print(f"{path}: {len(g):,} hexagons, {len(html) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
