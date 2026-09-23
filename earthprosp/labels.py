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
