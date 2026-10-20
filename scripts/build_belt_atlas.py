"""Build the belt atlas page: every res-8 belt map, its test against the 36 km² map, and its
greenfield targets, in one HTML page with the map images beside it.

    python scripts/build_belt_atlas.py --out outputs/atlas

Reads outputs/downscale_<belt>/ (downscale.py --final). Writes index.html and maps/*.jpg.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageChops

BELTS = {  # folder suffix: (name, what is there)
    "wus": ("Great Basin and Arizona", "Carlin-type gold, porphyry copper, Mother Lode gold"),
    "andes": ("Central Andes", "Porphyry copper, IOCG, epithermal gold and silver"),
    "n_andes": ("Northern Andes", "Porphyry copper-gold, epithermal gold, Peruvian polymetallic"),
    "yilgarn": ("Yilgarn and Pilbara", "Archean orogenic gold, nickel sulphide, iron ore, lithium"),
    "abitibi": ("Abitibi and Superior", "Archean gold and VMS copper-zinc"),
    "copperbelt": ("Central African Copperbelt", "Sediment-hosted copper-cobalt"),
    "s_africa": ("Southern Africa", "Bushveld PGM and chrome, Witwatersrand gold, Kalahari manganese"),
    "tethyan_iran": ("Iranian Tethyan belt", "Porphyry copper, chromite, lead-zinc"),
    "tibet": ("Tibet and Yulong", "Collisional porphyry copper-molybdenum"),
    "kazakh": ("Kazakhstan and Tien Shan", "Porphyry copper, orogenic gold, uranium"),
    "mexico": ("Sierra Madre, Mexico", "Epithermal silver-gold, porphyry copper"),
    "bc_alaska": ("British Columbia and SE Alaska", "Porphyry copper-gold, VMS, orogenic gold"),
    "fennoscandia": ("Fennoscandian Shield", "Orogenic gold, VMS, iron, nickel"),
    "carajas_qf": ("Carajás and Quadrilátero Ferrífero", "IOCG copper, iron ore, orogenic gold"),
    "png": ("Papua New Guinea", "Porphyry copper-gold, epithermal gold"),
    # Frontier: few recorded deposits, mapped with the model pooled across the belts above.
    "birimian": ("West African Birimian", "Orogenic and paleoplacer gold, manganese"),
    "arabian_nubian": ("Arabian-Nubian Shield", "Orogenic gold, VMS copper-zinc"),
    "guiana": ("Guiana Shield", "Orogenic gold, bauxite"),
    "gobi": ("Gobi and South Mongolia", "Porphyry copper-gold, orogenic gold"),
}
FRONTIER = {"birimian", "arabian_nubian", "guiana", "gobi"}


def land_path(geojson, w=720, h=360):
    g = json.load(open(geojson))
    parts = []
    for f in g["features"]:
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            ring = poly[0]
            pts = [(round((x + 180) / 360 * w, 1), round((90 - y) / 180 * h, 1)) for x, y in ring[::2] or ring]
            if len(pts) > 2:
                parts.append("M" + "L".join(f"{x},{y}" for x, y in pts) + "Z")
    return "".join(parts)


def crop_jpg(src, dst, width=1100):
    im = Image.open(src).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    box = ImageChops.difference(im, bg).getbbox()
    if box:
        im = im.crop(box)
    im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    im.save(dst, quality=84, optimize=True)
    return im.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/atlas")
    ap.add_argument("--land", default="data/raw/ne_110m_land.geojson")
    ap.add_argument("--template", default="scripts/belt_atlas_template.html")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "maps"), exist_ok=True)
    bp = "outputs/bootstrap_downscale/bootstrap_vs_coarse.csv"
    boot = pd.read_csv(bp) if os.path.exists(bp) else pd.DataFrame(columns=["region", "model"])
    tp = os.path.join(a.out, "greenfield_targets_all_regions.csv")
    sysmap = (pd.read_csv(tp).set_index("h3").likely_system.to_dict()
              if os.path.exists(tp) and "likely_system" in pd.read_csv(tp, nrows=1) else {})
    belts = []
    for key, (name, deposits) in BELTS.items():
        d = f"outputs/downscale_{key}"
        if not os.path.exists(os.path.join(d, "map_res8.png")):
            continue
        meta = json.load(open(os.path.join(d, "downscale.json")))
        s = pd.read_csv(os.path.join(d, "downscale_summary.csv")) if os.path.exists(
            os.path.join(d, "downscale_summary.csv")) else None
        stat = {}
        if s is not None:
            for _, r in s[s.target == "all"].iterrows():
                stat[r.model] = {"auc": round(float(r.capture_auc), 3),
                                 "district": None if pd.isna(r.within_district_auc) else round(float(r.within_district_auc), 3),
                                 "top5": round(float(r["capture@5%"]), 3)}
        w, h = crop_jpg(os.path.join(d, "map_res8.png"), os.path.join(a.out, "maps", f"{key}.jpg"))
        t = pd.read_csv(os.path.join(d, "greenfield_targets_res8.csv")).head(20)
        x0, y0, x1, y1 = meta["bbox"]
        belts.append({"key": key, "name": name, "deposits": deposits, "bbox": meta["bbox"],
                      "frontier": key in FRONTIER,
                      "ci": {r.model: [r.diff_vs_coarse, r.lo90, r.hi90] for r in boot[boot.region == key].itertuples()}, "map_model": meta.get("map_model", "fine"),
                      "cells": meta["n_res8"], "pos": meta["n_pos8"], "stat": stat, "img": f"maps/{key}.jpg",
                      "iw": w, "ih": h,
                      "targets": [{"lat": round(r.lat, 3), "lon": round(r.lon, 3), "score": round(r.blend, 4),
                                   "h3": r.h3, "system": sysmap.get(r.h3, "").replace("_", " "), "stable": round(float(getattr(r, "seed_top1pct_share", np.nan)), 2)
                                   if hasattr(r, "seed_top1pct_share") else None} for r in t.itertuples()]})
    order = {k: i for i, k in enumerate(BELTS)}
    belts.sort(key=lambda b: order[b["key"]])
    won = [b for b in belts if "fine" in b["stat"] and "coarse" in b["stat"] and not b["frontier"]]
    n_win = sum(b["stat"]["fine"]["auc"] > b["stat"]["coarse"]["auc"] for b in won)
    n_win_d = sum((b["stat"]["fine"]["district"] or 0) > (b["stat"]["coarse"]["district"] or 0) for b in won)
    med = lambda k, m: float(np.median([b["stat"][m][k] for b in won if b["stat"][m][k] is not None]))  # noqa: E731
    summary = {"n": len(won), "n_win": n_win, "n_win_district": n_win_d,
               "auc_fine": round(med("auc", "fine"), 3) if won else None,
               "auc_coarse": round(med("auc", "coarse"), 3) if won else None,
               "district_fine": round(med("district", "fine"), 3) if won else None,
               "district_coarse": round(med("district", "coarse"), 3) if won else None,
               "cells": sum(b["cells"] for b in belts), "n_core": sum(not b["frontier"] for b in belts)}
    page = open(a.template).read()
    page = page.replace("__LAND__", land_path(a.land)).replace("__DATA__", json.dumps({"belts": belts, "summary": summary}))
    page = page.replace("__NCORE__", str(summary["n_core"])).replace("__NFRONT__", str(len(belts) - summary["n_core"]))
    page = page.replace("__NBELTS__", str(len(belts))).replace("__NCELLS__", f"{summary['cells'] / 1e6:.1f}")
    open(os.path.join(a.out, "index.html"), "w").write(page)
    print(f"{len(belts)} belts, {summary}")


if __name__ == "__main__":
    main()
