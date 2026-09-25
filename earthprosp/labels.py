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
