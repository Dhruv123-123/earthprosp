"""Streaming, resumable global aggregation of AEF overview pixels onto H3 cells.

Per-cell pixel bags do not fit in memory globally (~4.5 M cells x 32 x 64), so the global
run keeps only *mergeable* per-cell statistics: pixel count, sum and sum of squares of
the 64-d embedding. Each tile is reduced to partial sums in a worker process. Partials
are written to shard files (so an interrupted run resumes where it stopped) and then
merged into per-cell mean / std.
"""

from __future__ import annotations

import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import aef, grid


def _tile_partial(args):
    path, crs, utm_zone, level, res = args
    for attempt in range(3):
        try:
            px = aef.read_tile_overview(path, crs, utm_zone, level)
            break
        except Exception as exc:  # transient HTTP failure
            if attempt == 2:
                return path, None, str(exc)
            time.sleep(2 * (attempt + 1))
    if len(px.lat) == 0:
        return path, (np.empty(0, object), np.empty((0, 64), np.float32),
                      np.empty((0, 64), np.float32), np.empty(0, np.int32)), None
    cells = grid.latlng_to_cells(px.lat, px.lon, res)
    uniq, inv = np.unique(cells, return_inverse=True)
    s = np.zeros((len(uniq), 64), np.float64)
    ss = np.zeros((len(uniq), 64), np.float64)
    np.add.at(s, inv, px.emb)
    np.add.at(ss, inv, px.emb.astype(np.float64) ** 2)
    n = np.bincount(inv, minlength=len(uniq)).astype(np.int32)
    return path, (uniq, s.astype(np.float32), ss.astype(np.float32), n), None


def _reduce(cells, s, ss, n):
    codes, uniq = pd.factorize(cells)
    S = np.zeros((len(uniq), 64), np.float64)
    SS = np.zeros((len(uniq), 64), np.float64)
    np.add.at(S, codes, s)
    np.add.at(SS, codes, ss)
    N = np.bincount(codes, weights=n, minlength=len(uniq))
    return np.asarray(uniq, dtype=object), S.astype(np.float32), SS.astype(np.float32), N.astype(np.int32)


def run(tiles: pd.DataFrame, shard_dir: str, level: int = 6, res: int = grid.DEFAULT_RES,
        n_shards: int = 60, workers: int = 16) -> None:
    os.makedirs(shard_dir, exist_ok=True)
    tiles = tiles.sort_values(["utm_zone", "path"]).reset_index(drop=True)
    parts = np.array_split(np.arange(len(tiles)), n_shards)
    with ProcessPoolExecutor(workers) as ex:
        for k, idx in enumerate(parts):
            out = os.path.join(shard_dir, f"shard_{k:03d}.npz")
            if os.path.exists(out):
                continue
            t0 = time.time()
            jobs = [(r.path, r.crs, r.utm_zone, level, res) for r in tiles.iloc[idx].itertuples()]
            acc, failed = [], []
            for path, part, err in ex.map(_tile_partial, jobs, chunksize=4):
                if part is None:
                    failed.append(path)
                elif len(part[0]):
                    acc.append(part)
            if acc:
                cells, s, ss, n = _reduce(np.concatenate([a[0] for a in acc]), np.concatenate([a[1] for a in acc]),
                                          np.concatenate([a[2] for a in acc]), np.concatenate([a[3] for a in acc]))
            else:
                cells, s, ss, n = np.empty(0, object), np.empty((0, 64), np.float32), np.empty((0, 64), np.float32), np.empty(0, np.int32)
            np.savez(out, cells=cells.astype(str), s=s, ss=ss, n=n, failed=np.array(failed, dtype=str))
            print(f"[global] shard {k + 1}/{n_shards}: {len(idx)} tiles, {len(cells):,} cells, "
                  f"{len(failed)} failed, {time.time() - t0:.0f}s", flush=True)


def merge(shard_dir: str, min_pixels: int = 4) -> pd.DataFrame | tuple:
    """Merge shards -> (cells, mean (N,64) unit-norm, std (N,64), count, failed_paths).

    Streams one shard at a time into float32 accumulators: concatenating all shards (and
    float64 sums) needs ~10 GB at global scale. Most cells live in one or two shards, so
    float32 sums lose nothing that matters."""
    files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))
    cell_lists = []
    for f in files:
        with np.load(f) as z:
            cell_lists.append(z["cells"])
    codes_all, uniq = pd.factorize(np.concatenate(cell_lists))
    bounds = np.cumsum([0] + [len(c) for c in cell_lists])
    del cell_lists
    S = np.zeros((len(uniq), 64), np.float32)
    SS = np.zeros((len(uniq), 64), np.float32)
    N = np.zeros(len(uniq), np.int64)
    failed = []
    for k, f in enumerate(files):
        codes = codes_all[bounds[k]:bounds[k + 1]]
        with np.load(f) as z:
            np.add.at(S, codes, z["s"])
            np.add.at(SS, codes, z["ss"])
            np.add.at(N, codes, z["n"])
            failed.extend(z["failed"].tolist())
    keep = N >= min_pixels
    cells, S, SS, N = np.asarray(uniq)[keep], S[keep], SS[keep], N[keep]
    S /= N[:, None]                       # S is now the mean
    SS /= N[:, None]
    SS -= S ** 2
    np.sqrt(np.maximum(SS, 0, out=SS), out=SS)
    S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-8
    return cells, S, SS, N.astype(np.int32), np.array(failed, dtype=str)
