"""Pack the v3 res-4 table into a single self-contained HTML explorer: commodity and
mineral-system layers, model disagreement, and an area-of-applicability mask.

    python scripts/build_explorer_v3.py --out outputs/v3
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
from earthprosp import mineral_systems as MS  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from build_explorer import LABELS  # noqa: E402

SYS_LABELS = {
    "porphyry_skarn_cu": "Porphyry and skarn Cu-Mo-Au", "epithermal_orogenic_au": "Epithermal and orogenic Au",
    "vms": "Volcanogenic massive sulfide", "sed_hosted_base": "Sediment-hosted Cu, Zn-Pb",
    "magmatic_mafic": "Magmatic Ni-Cu-PGE-Cr", "felsic_alkaline_rare": "Granite, pegmatite, carbonatite (Li, REE, Sn-W)",
    "chemical_sedimentary": "Chemical sedimentary (BIF, Mn, phosphate)", "surficial_residual": "Laterite and placer",
    "sandstone_u": "Sandstone and unconformity U",
}


def u8(x):
    return np.round(np.clip(np.nan_to_num(np.asarray(x, np.float64)), 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/v3")
    ap.add_argument("--land", default="data/raw/ne_110m_land.geojson")
    ap.add_argument("--skill", default="")
    a = ap.parse_args()
    g = pd.read_parquet(os.path.join(a.out, "global_v3_res4.parquet"))
    ll = np.array([h3.cell_to_latlng(h) for h in g.h4])
    countries = sorted(g.country.unique())
    cidx = {c: i for i, c in enumerate(countries)}
    p = g.p_prospective.to_numpy()
    marg = g[[f"p_{c}" for c in C.CLASSES]].to_numpy()
    q = marg / np.maximum(marg.sum(1, keepdims=True), 1e-9)
    sy = g[[f"sys_{s}" for s in MS.SYSTEMS]].to_numpy()
    sy = sy / np.maximum(sy.sum(1, keepdims=True), 1e-9)
    unc = g.uncertainty.to_numpy()
    unc = unc / max(np.quantile(unc, 0.999), 1e-9)
    rec = np.zeros(len(g), dtype=[("lat", "<i2"), ("lon", "<i2"), ("p", "u1"), ("known", "u1"), ("c", "<u2"),
                                  ("unc", "u1"), ("aoa", "u1"),
                                  *[(f"q{k}", "u1") for k in range(C.N_CLASSES)],
                                  *[(f"s{k}", "u1") for k in range(len(MS.SYSTEMS))]])
    rec["lat"] = np.round(ll[:, 0] * 100)
    rec["lon"] = np.round(ll[:, 1] * 100)
    rec["p"] = u8(p)
    rec["known"] = np.clip(g.known.to_numpy(), 0, 255)
    rec["c"] = [cidx[c] for c in g.country]
    rec["unc"] = u8(unc)
    rec["aoa"] = u8(g.in_aoa.to_numpy())
    for k in range(C.N_CLASSES):
        rec[f"q{k}"] = u8(q[:, k])
    for k in range(len(MS.SYSTEMS)):
        rec[f"s{k}"] = u8(sy[:, k])
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
    cols = (["country", "cells", "mean_p", "known_positive_cells", "p_prospective"]
            + [f"p_{c}" for c in C.CLASSES] + [f"sys_{s}" for s in MS.SYSTEMS if f"sys_{s}" in ct])
    table = json.loads(ct[cols].round(3).to_json(orient="records"))
    n6 = json.load(open(os.path.join(a.out, "run.json")))["n_cells"]

    html = open(os.path.join(os.path.dirname(__file__), "explorer_v3_template.html")).read()
    for key, val in {
        "__CLASSES__": json.dumps(C.CLASSES), "__LABELS__": json.dumps([LABELS[c] for c in C.CLASSES]),
        "__SYSTEMS__": json.dumps(MS.SYSTEMS), "__SYS_LABELS__": json.dumps([SYS_LABELS[s] for s in MS.SYSTEMS]),
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
