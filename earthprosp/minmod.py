"""Flatten the DARPA CriticalMAAS MinMod mineral-site dump into one table.

MinMod (github.com/DARPA-CRITICALMAAS/ta2-minmod-data, data/mineral-sites/) republishes
USGS MRDS and ~25 other USGS compilations (porphyry Cu, sediment-hosted Cu/Zn-Pb/Au,
VMS, Ni-Cr-PGE, carbonatites, laterites, REE, phosphate, potash, major deposits, ARDF),
state surveys, Mudd & Jowitt global Cu / Ni / Zn-Pb resource compilations, the world
Sn-W deposit database, and NI 43-101 report extractions, all in one JSON schema. It is
the most complete open mineral-site collection reachable without the USGS host.

One output row per (source, record_id): coordinates, country, site type (MRDS dev
status), ordered commodity list, deposit-type candidates, and the largest reported ore
tonnage. Several teams (umn, sri, usc, ...) extracted overlapping sources; the first
team in TEAM_ORDER wins on duplicates.

    python -m earthprosp.minmod /path/to/ta2-minmod-data data/minmod_sites.parquet
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from . import commodities as C

TEAM_ORDER = ("umn", "usc", "sri", "inferlink", "usc-stu", "bvu", "caknoblock", "jhammars",
              "ggraham", "glederer")

# Element symbols used by non-MRDS sources -> class.
_SYMBOL = {
    "Au": "Au", "Ag": "Ag", "Cu": "Cu", "Pb": "PbZn", "Zn": "PbZn", "Fe": "Fe", "Mn": "Mn", "U": "U",
    "W": "WMoSn", "Mo": "WMoSn", "Sn": "WMoSn", "Bi": "WMoSn", "Ni": "NiCoPGECr", "Co": "NiCoPGECr",
    "Pt": "NiCoPGECr", "Pd": "NiCoPGECr", "PGE": "NiCoPGECr", "PGM": "NiCoPGECr", "Cr": "NiCoPGECr",
    "Li": "LiBeTaNb", "Be": "LiBeTaNb", "Ta": "LiBeTaNb", "Nb": "LiBeTaNb", "Cs": "LiBeTaNb",
    "REE": "REE", "Ce": "REE", "La": "REE", "Nd": "REE", "Y": "REE", "Th": "REE", "Dy": "REE",
    "Al": "Al", "Hg": "HgSbAs", "Sb": "HgSbAs", "As": "HgSbAs", "Ba": "BaF", "F": "BaF", "P": "P",
}


def _point(wkt):
    if not isinstance(wkt, str):
        return np.nan, np.nan
    m = re.match(r"\s*POINT\s*\(\s*([-0-9.eE]+)\s+([-0-9.eE]+)\s*\)", wkt)
    if not m:
        return np.nan, np.nan
    return float(m.group(2)), float(m.group(1))


def classify_name(name: str) -> str | None:
    if not name:
        return None
    n = name.strip()
    cls = C.classify(n) or _SYMBOL.get(n)
    if cls:
        return cls
    from .labels import text_to_vector
    v = text_to_vector(n)
    return C.CLASSES[int(v.argmax())] if v.sum() > 0 else None


def _site_rows(path: str, team: str):
    try:
        data = json.load(open(path))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    if isinstance(data, dict):
        data = [data]
    for r in data:
        loc = r.get("location_info") or {}
        lat, lon = _point(loc.get("location"))
        country = ((loc.get("country") or [{}])[0] or {}).get("observed_name")
        commods, ore_max = [], np.nan
        for m in r.get("mineral_inventory") or []:
            name = ((m.get("commodity") or {}).get("observed_name") or "").strip()
            if name and name not in commods:
                commods.append(name)
            ore = (m.get("ore") or {}).get("value")
            if isinstance(ore, (int, float)) and np.isfinite(ore):
                ore_max = ore if not np.isfinite(ore_max) else max(ore_max, ore)
        dtc = r.get("deposit_type_candidate") or []
        dtypes = [d.get("observed_name") for d in dtc if isinstance(d, dict) and d.get("observed_name")]
        yield {
            "team": team,
            "source": r.get("source_id", ""),
            "record_id": str(r.get("record_id", "")),
            "name": r.get("name"),
            "lat": lat, "lon": lon, "country": country,
            "site_type": r.get("site_type"),
            "commodities": "|".join(commods),
            "deposit_types": "|".join(dtypes),
            "ore_max": ore_max,
        }


def load(repo_dir: str) -> pd.DataFrame:
    root = os.path.join(repo_dir, "data", "mineral-sites")
    rows = []
    for team in sorted(os.listdir(root), key=lambda t: TEAM_ORDER.index(t) if t in TEAM_ORDER else 99):
        for f in sorted(glob.glob(os.path.join(root, team, "**", "*.json"), recursive=True)):
            rows.extend(_site_rows(f, team))
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(["source", "record_id"], keep="first")
    df = df.dropna(subset=["lat", "lon"])
    df = df[df.lat.between(-90, 90) & df.lon.between(-180, 180)]
    return df.reset_index(drop=True)


def class_vectors(df: pd.DataFrame) -> np.ndarray:
    """(n, K) commodity vectors with MRDS rank weights (1, 0.5, 0.25)."""
    vec = np.zeros((len(df), C.N_CLASSES), np.float32)
    cache: dict[str, str | None] = {}
    for i, s in enumerate(df["commodities"].fillna("")):
        rank = 0
        seen = set()
        for name in s.split("|"):
            if name not in cache:
                cache[name] = classify_name(name)
            cls = cache[name]
            if cls and cls not in seen and rank < 3:
                vec[i, C.CLASS_INDEX[cls]] += C.POSITION_WEIGHT[rank]
                seen.add(cls)
                rank += 1
    return vec


if __name__ == "__main__":
    d = load(sys.argv[1])
    d.to_parquet(sys.argv[2], index=False)
    print(f"{len(d):,} sites")
    print(d.source.value_counts().head(40).to_string())
    print(d.site_type.value_counts().to_string())
