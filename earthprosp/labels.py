"""Deposit labels from USGS MRDS, aggregated to H3 cells.

For each cell we produce:
  * `composition` (K,)  - normalised commodity-class mix of the known deposits in the
                          cell, weighted by development status and commodity rank;
  * `confidence`        - max development-status weight of any deposit in the cell
                          (1 = producer, 0.3 = occurrence), used as the positive weight;
  * `n_records`         - count of *all* MRDS records (including aggregates), a proxy
                          for how much the cell has been looked at (survey effort).

MRDS is ~88% US records and was last updated in 2011-ish; outside the US it is sparse
and biased to historic producers. See docs/DESIGN.md for other label sources.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import commodities as C
from . import grid

MRDS_URL = "https://mrdata.usgs.gov/mrds/mrds-csv.zip"


def load_mrds(path: str = MRDS_URL) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False,
                     usecols=["dep_id", "latitude", "longitude", "country",
                              "commod1", "commod2", "commod3", "dev_stat", "model"])
    return df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


def deposit_vectors(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-record class vector (n, K) and development-status weight (n,)."""
    vec = np.zeros((len(df), C.N_CLASSES), np.float32)
    cols = [df[c].to_numpy() for c in ("commod1", "commod2", "commod3")]
    for i in range(len(df)):
        for rank, col in enumerate(cols):
            for tok in C.tokens(col[i]):
                cls = C.classify(tok)
                if cls is not None:
                    vec[i, C.CLASS_INDEX[cls]] += C.POSITION_WEIGHT[rank]
    w = df["dev_stat"].map(C.DEV_STAT_WEIGHT).fillna(0.3).to_numpy(np.float32)
    return vec, w


def cell_labels(df: pd.DataFrame, res: int = grid.DEFAULT_RES) -> pd.DataFrame:
    """One row per H3 cell that contains at least one MRDS record."""
    vec, w = deposit_vectors(df)
    cells = grid.latlng_to_cells(df.latitude.to_numpy(), df.longitude.to_numpy(), res)
    is_target = vec.sum(1) > 0
    comp = pd.DataFrame(vec * w[:, None], columns=C.CLASSES)
    comp["cell"] = cells
    comp["conf"] = np.where(is_target, w, 0.0)
    comp["n_records"] = 1
    g = comp.groupby("cell").agg({**{c: "sum" for c in C.CLASSES}, "conf": "max", "n_records": "sum"})
    tot = g[C.CLASSES].sum(axis=1)
    g[C.CLASSES] = g[C.CLASSES].div(tot.where(tot > 0, 1.0), axis=0)
    g["positive"] = tot > 0
    return g


def align(cells: np.ndarray, labels: pd.DataFrame) -> dict[str, np.ndarray]:
    """Align cell labels to a cell array; cells with no record get zeros (unlabeled)."""
    lab = labels.reindex(pd.Index(cells, name="cell"))
    return {
        "positive": lab["positive"].fillna(False).to_numpy(bool),
        "conf": lab["conf"].fillna(0.0).to_numpy(np.float32),
        "composition": lab[C.CLASSES].fillna(0.0).to_numpy(np.float32),
        "n_records": lab["n_records"].fillna(0).to_numpy(np.int32),
    }


# ---------------------------------------------------------------------------
# Global label sources (USGS global deposit compilations)
# ---------------------------------------------------------------------------
# Rough long-run metal prices, used only to turn grades into a value-weighted
# commodity split for grade-tonnage deposits. Units: USD per (grade unit x tonne ore):
# % grades -> price_per_t / 100, g/t grades -> price_per_g.
_VALUE_PER_GRADE = {
    "cugrd": ("Cu", 9000 / 100), "mogrd": ("WMoSn", 40000 / 100), "cogrd": ("NiCoPGECr", 30000 / 100),
    "zngrd": ("PbZn", 2700 / 100), "pbgrd": ("PbZn", 2100 / 100),
    "augrd": ("Au", 80.0), "aggrd": ("Ag", 1.0),
}

# Major-deposits commodity codes -> class.
_MAJOR_CODE = {
    "Cu": "Cu", "Au": "Au", "Ag": "Ag", "Fe": "Fe", "Pb": "PbZn", "Zn": "PbZn",
    "phosphate": "P", "P": "P", "PGE": "NiCoPGECr", "Pt": "NiCoPGECr", "Pd": "NiCoPGECr",
    "Cr": "NiCoPGECr", "Ni": "NiCoPGECr", "Co": "NiCoPGECr", "Mo": "WMoSn", "Sn": "WMoSn",
    "W": "WMoSn", "Bi": "WMoSn", "REE": "REE", "Th": "REE", "Al": "Al", "Mn": "Mn",
    "Nb": "LiBeTaNb", "Ta": "LiBeTaNb", "Li": "LiBeTaNb", "Be": "LiBeTaNb", "Cs": "LiBeTaNb",
    "U": "U", "barite": "BaF", "fluorspar": "BaF", "fluorite": "BaF", "F": "BaF",
    "Hg": "HgSbAs", "Sb": "HgSbAs", "As": "HgSbAs",
}


def _records(lat, lon, vec, conf, source) -> pd.DataFrame:
    df = pd.DataFrame(vec, columns=C.CLASSES)
    df["latitude"], df["longitude"], df["conf"], df["source"] = lat, lon, conf, source
    ok = df.latitude.between(-90, 90) & df.longitude.between(-180, 180)
    return df[ok].reset_index(drop=True)


def mrds_records(df: pd.DataFrame) -> pd.DataFrame:
    vec, w = deposit_vectors(df)
    keep = vec.sum(1) > 0
    return _records(df.latitude.to_numpy()[keep], df.longitude.to_numpy()[keep], vec[keep], w[keep], "mrds")


def major_deposit_records(folder: str) -> pd.DataFrame:
    """USGS OFR 2005-1294, 'Major mineral deposits of the world' (~3,200 sites)."""
    dep = pd.read_csv(f"{folder}/deposit.csv", encoding="latin1")
    com = pd.read_csv(f"{folder}/commodity.csv", encoding="latin1")
    com["rank"] = com.groupby("gid").cumcount()
    vec = np.zeros((len(dep), C.N_CLASSES), np.float32)
    row_of = {g: i for i, g in enumerate(dep.gid)}
    for g, v, r in zip(com.gid, com.value, com["rank"]):
        cls = _MAJOR_CODE.get(str(v).strip())
        if cls is not None and g in row_of:
            vec[row_of[g], C.CLASS_INDEX[cls]] += C.POSITION_WEIGHT[min(r, 2)]
    keep = vec.sum(1) > 0
    return _records(dep.latitude.to_numpy()[keep], dep.longitude.to_numpy()[keep], vec[keep], 1.0, "major")


def grade_tonnage_records(csv: str, primary: str, source: str) -> pd.DataFrame:
    """USGS grade-tonnage compilations (porphyry Cu, sediment-hosted Cu, VMS).

    Composition = share of in-ground metal value per class, from grades x rough prices;
    deposits without grades fall back to the deposit type's primary class."""
    d = pd.read_csv(csv, encoding="latin1", low_memory=False).dropna(subset=["latitude", "longitude"])
    vec = np.zeros((len(d), C.N_CLASSES), np.float32)
    for col, (cls, value) in _VALUE_PER_GRADE.items():
        if col in d:
            vec[:, C.CLASS_INDEX[cls]] += np.nan_to_num(pd.to_numeric(d[col], errors="coerce").to_numpy()) * value
    empty = vec.sum(1) == 0
    vec[empty, C.CLASS_INDEX[primary]] = 1.0
    return _records(d.latitude.to_numpy(), d.longitude.to_numpy(), vec, 1.0, source)


def global_records(mrds_path: str = "data/mrds.csv", usgs_dir: str = "data/usgs_global") -> pd.DataFrame:
    parts = [
        mrds_records(load_mrds(mrds_path)),
        major_deposit_records(f"{usgs_dir}/ofr20051294"),
        grade_tonnage_records(f"{usgs_dir}/porcu/main.csv", "Cu", "porcu"),
        grade_tonnage_records(f"{usgs_dir}/sedcu/main.csv", "Cu", "sedcu"),
        grade_tonnage_records(f"{usgs_dir}/vms/main.csv", "Cu", "vms"),
    ]
    return pd.concat(parts, ignore_index=True)


def cell_labels_from_records(rec: pd.DataFrame, res: int = grid.DEFAULT_RES) -> pd.DataFrame:
    """Aggregate normalised per-record class vectors to cells (confidence-weighted)."""
    vec = rec[C.CLASSES].to_numpy(np.float32)
    vec = vec / vec.sum(1, keepdims=True)
    cells = grid.latlng_to_cells(rec.latitude.to_numpy(), rec.longitude.to_numpy(), res)
    comp = pd.DataFrame(vec * rec["conf"].to_numpy()[:, None], columns=C.CLASSES)
    comp["cell"], comp["conf"], comp["n_records"] = cells, rec["conf"].to_numpy(), 1
    comp["major"] = (rec["source"] != "mrds").to_numpy()
    g = comp.groupby("cell").agg({**{c: "sum" for c in C.CLASSES}, "conf": "max",
                                   "n_records": "sum", "major": "any"})
    tot = g[C.CLASSES].sum(axis=1)
    g[C.CLASSES] = g[C.CLASSES].div(tot, axis=0)
    g["positive"] = True
    return g


# ---------------------------------------------------------------------------
# v3 label set: one global standard without MRDS
# ---------------------------------------------------------------------------
# Free-text commodity names (USGS mineral facilities, OSM `resource=*`) -> class.
_TEXT_RULES: list[tuple[str, str]] = [
    (r"\bgold\b|\bau\b", "Au"), (r"silver|\bag\b", "Ag"), (r"copper|\bcu\b", "Cu"),
    (r"\blead\b|\bzinc\b|\bpb\b|\bzn\b|galena|sphalerite", "PbZn"),
    (r"iron ore|iron\b|magnetite|hematite|haematite|\bfe\b", "Fe"), (r"manganese|\bmn\b", "Mn"),
    (r"uranium|\bu3o8\b", "U"), (r"tungsten|molybdenum|\btin\b|cassiterite|wolfram|bismuth", "WMoSn"),
    (r"nickel|cobalt|platinum|palladium|\bpgm?s?\b|chrom", "NiCoPGECr"),
    (r"lithium|beryl|tantal|niobium|columbium|cesium|spodumene", "LiBeTaNb"),
    (r"rare.earth|\bree\b|monazite|thorium|yttrium", "REE"), (r"bauxite|alumin", "Al"),
    (r"mercury|antimony|arsenic", "HgSbAs"), (r"barite|baryte|fluorspar|fluorite", "BaF"),
    (r"phosphat", "P"),
]


def text_to_vector(text) -> np.ndarray:
    """Commodity vector from free text ("Copper, cobalt"); first match weighs most."""
    import re
    vec = np.zeros(C.N_CLASSES, np.float32)
    if text is None or (isinstance(text, float) and text != text):
        return vec
    parts = [p.strip().lower().replace("_", " ") for p in re.split(r"[,;/]", str(text)) if p.strip()]
    seen, rank = set(), 0
    for p in parts:
        for pat, cls in _TEXT_RULES:
            if re.search(pat, p):
                if cls not in seen and rank < 3:
                    vec[C.CLASS_INDEX[cls]] += C.POSITION_WEIGHT[rank]
                    seen.add(cls)
                    rank += 1
                break
    return vec


def mineral_facility_records(path: str = "data/raw/wri/minfac/minfac.shp") -> pd.DataFrame:
    """USGS Mineral Facilities of the World (non-US): operating *mines* only.

    Plants, smelters and refineries are dropped (they sit where the power and ports are,
    not on the ore), as are oil/gas, coal and industrial minerals (no class)."""
    import geopandas as gpd
    g = gpd.read_file(path)
    mine = g["fac_type"].fillna("").str.contains("Mine", case=False)
    g = g[mine].reset_index(drop=True)
    vec = np.stack([text_to_vector(t) for t in g["commodity"]])
    keep = vec.sum(1) > 0
    lat = pd.to_numeric(g["latitude"], errors="coerce").to_numpy()
    lon = pd.to_numeric(g["longitude"], errors="coerce").to_numpy()
    from . import mineral_systems as MS
    rec = _records(lat[keep], lon[keep], vec[keep], 1.0, "minfac")
    rec["precision"] = g.loc[keep, "precision"].to_numpy()[rec.index] if len(rec) == keep.sum() else None
    rec["system"] = [MS.assign(v) for v in rec[C.CLASSES].to_numpy()]
    return rec


def major_deposit_records_typed(folder: str) -> pd.DataFrame:
    """Major deposits with a mineral-system label (model text first, then rules)."""
    from . import mineral_systems as MS
    rec = major_deposit_records(folder)
    dep = pd.read_csv(f"{folder}/deposit.csv", encoding="latin1")
    dep = dep.dropna(subset=["latitude", "longitude"])
    # major_deposit_records drops rows without a class; realign on coordinates.
    key = lambda la, lo: (np.round(la, 4), np.round(lo, 4))  # noqa: E731
    meta = {key(a, b): (t, m, d) for a, b, t, m, d in
            zip(dep.latitude, dep.longitude, dep.dep_type, dep.type_detail, dep.model_name)}
    systems = []
    for la, lo, v in zip(rec.latitude, rec.longitude, rec[C.CLASSES].to_numpy()):
        t, m, d = meta.get(key(la, lo), (None, None, None))
        text = " ".join(str(x) for x in (m, d) if isinstance(x, str)) or None
        systems.append(MS.assign(v, t, text))
    rec["system"] = systems
    return rec


def osm_mine_records(path: str = "data/overture_mines.parquet", min_km2: float = 0.0) -> pd.DataFrame:
    """OSM mine/quarry polygons whose `resource` tag names a target commodity.

    Independent of the USGS compilations (mapped by OSM volunteers from imagery and
    local knowledge), so it is used as an *evaluation* set, not for training."""
    d = pd.read_parquet(path)
    d = d[d["bbox_km2"] >= min_km2]
    vec = np.stack([text_to_vector(t) for t in d["resource"].fillna("")])
    keep = vec.sum(1) > 0
    rec = _records(d.lat.to_numpy()[keep], d.lon.to_numpy()[keep], vec[keep], 1.0, "osm")
    rec["bbox_km2"] = d["bbox_km2"].to_numpy()[keep]
    return rec


def global_records_v3(raw_dir: str = "data/raw", mrds_path: str = "data/mrds.csv") -> pd.DataFrame:
    """Training records with one global standard: world-class deposits + operating mines.

    MRDS producers are added when the MRDS csv is present (its host is not always
    reachable); propensity weighting downstream keeps the dense US from dominating."""
    import os
    parts = [major_deposit_records_typed(f"{raw_dir}/wri/major/ofr20051294"),
             mineral_facility_records(f"{raw_dir}/wri/minfac/minfac.shp")]
    if os.path.exists(mrds_path):
        m = mrds_records(load_mrds(mrds_path))
        parts.append(m[m["conf"] >= 1.0])
    rec = pd.concat(parts, ignore_index=True)
    if "system" not in rec:
        rec["system"] = None
    return rec
