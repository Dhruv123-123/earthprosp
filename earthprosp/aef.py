"""Read AlphaEarth Foundations (AEF) annual embeddings and aggregate them onto H3 cells.

Data source: the public COG mirror of GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL on
Source Cooperative (s3://us-west-2.opendata.source.coop/tge-labs/aef/, CC-BY 4.0,
"The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and
Google DeepMind."). Each COG is 8192x8192 px at 10 m, 64 int8 bands, in one UTM zone.

We never touch the 10 m pixels for a global run. Each COG's overview levels are the
*unit-normalised mean* of the 10 m embeddings beneath them, so overview level 6
(64x64 px at 1.28 km) is already a spatially pooled embedding; reading it costs a
few HTTP range requests per file (~34k files per year globally).

Two quirks handled here:
  * the COGs are "bottom-up" (origin bottom-left, positive y resolution), so pixel
    row r has northing y0 + (r + 0.5) * res with y0 the *southern* edge;
  * int8 -> float de-quantisation is sign(v) * (v / 127.5)^2, with -128 = nodata.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import grid

INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
S3_PREFIX = "s3://us-west-2.opendata.source.coop/"
HTTPS_PREFIX = "https://data.source.coop/"
EMB_DIM = 64
NODATA = -128


def dequantize(raw: np.ndarray) -> np.ndarray:
    v = raw.astype(np.float32)
    out = np.sign(v) * (v / 127.5) ** 2
    out[raw == NODATA] = np.nan
    return out


def to_https(path: str) -> str:
    return path.replace(S3_PREFIX, HTTPS_PREFIX) if path.startswith(S3_PREFIX) else path


def load_index(path_or_url: str = INDEX_URL, year: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(
        path_or_url,
        columns=["path", "year", "utm_zone", "crs",
                 "wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north"],
    )
    if year is not None:
        df = df[df.year == year]
    return df.reset_index(drop=True)


def tiles_in_bbox(index: pd.DataFrame, west: float, south: float, east: float, north: float) -> pd.DataFrame:
    m = ((index.wgs84_west < east) & (index.wgs84_east > west)
         & (index.wgs84_south < north) & (index.wgs84_north > south))
    return index[m].reset_index(drop=True)


@dataclass
class TilePixels:
    lat: np.ndarray  # (n,)
    lon: np.ndarray  # (n,)
    emb: np.ndarray  # (n, 64) float32, unit-norm


def read_tile_overview(path: str, crs: str, utm_zone: str, overview_level: int = 6) -> TilePixels:
    """Read one COG at an overview level and return valid pixels with WGS84 coords.

    Pixels outside the tile's own UTM zone are dropped so that the overlap between
    neighbouring zones is not double counted.
    """
    import rasterio
    from pyproj import Transformer

    url = "/vsicurl/" + to_https(path)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff",
                      GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="1"):
        with rasterio.open(url, overview_level=overview_level) as src:
            raw = src.read()
            t = src.transform
    emb = dequantize(raw)  # (64, H, W)
    _, h, w = emb.shape
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    # Bottom-up: t.e > 0 and t.f is the southern edge; the affine handles both cases.
    x = t.c + (cols + 0.5) * t.a
    y = t.f + (rows + 0.5) * t.e
    valid = ~np.isnan(emb[0])
    if not valid.any():
        return TilePixels(np.empty(0), np.empty(0), np.empty((0, EMB_DIM), np.float32))
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(x[valid], y[valid])
    zone = int(utm_zone[:-1])
    zw, ze = -180 + 6 * (zone - 1), -180 + 6 * zone
    keep = (lon >= zw) & (lon < ze)
    e = emb[:, valid].T[keep]
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-8
    return TilePixels(lat[keep], lon[keep], e.astype(np.float32))


def read_tiles(tiles: pd.DataFrame, overview_level: int = 6, workers: int = 16) -> TilePixels:
    def one(row):
        try:
            return read_tile_overview(row.path, row.crs, row.utm_zone, overview_level)
        except Exception as exc:  # network hiccup on one tile should not kill a global run
            print(f"[aef] failed {row.path}: {exc}")
            return None

    os.environ.setdefault("CURL_CA_BUNDLE", os.environ.get("REQUESTS_CA_BUNDLE", ""))
    with ThreadPoolExecutor(workers) as ex:
        parts = [p for p in ex.map(one, tiles.itertuples()) if p is not None and len(p.lat)]
    if not parts:
        return TilePixels(np.empty(0), np.empty(0), np.empty((0, EMB_DIM), np.float32))
    return TilePixels(np.concatenate([p.lat for p in parts]),
                      np.concatenate([p.lon for p in parts]),
                      np.concatenate([p.emb for p in parts]))


@dataclass
class CellBags:
    """Per-cell bags of AEF pixel embeddings (the model input)."""
    cells: np.ndarray  # (N,) H3 ids
    bags: np.ndarray   # (N, B, 64) float16
    mask: np.ndarray   # (N, B) bool, True = real pixel

    def save(self, path: str) -> None:
        np.savez_compressed(path, cells=self.cells.astype(str), bags=self.bags, mask=self.mask)

    @classmethod
    def load(cls, path: str) -> "CellBags":
        z = np.load(path, allow_pickle=False)
        return cls(z["cells"].astype(object), z["bags"], z["mask"])

    def mean_embedding(self) -> np.ndarray:
        b = self.bags.astype(np.float32)
        m = self.mask[..., None]
        mu = (b * m).sum(1) / np.maximum(m.sum(1), 1)
        return mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-8)


def pixels_to_bags(px: TilePixels, res: int = grid.DEFAULT_RES, bag_size: int = 32,
                   min_pixels: int = 4, seed: int = 0) -> CellBags:
    """Group pixels by H3 cell. Cells with fewer than `min_pixels` valid pixels are
    dropped (mostly coastline slivers); larger cells are randomly subsampled."""
    rng = np.random.default_rng(seed)
    cells = grid.latlng_to_cells(px.lat, px.lon, res)
    order = np.argsort(cells, kind="stable")
    cells_sorted = cells[order]
    uniq, start, counts = np.unique(cells_sorted, return_index=True, return_counts=True)
    keep = counts >= min_pixels
    uniq, start, counts = uniq[keep], start[keep], counts[keep]
    bags = np.zeros((len(uniq), bag_size, EMB_DIM), np.float16)
    mask = np.zeros((len(uniq), bag_size), bool)
    for i, (s, c) in enumerate(zip(start, counts)):
        idx = order[s:s + c]
        if c > bag_size:
            idx = rng.choice(idx, bag_size, replace=False)
        bags[i, :len(idx)] = px.emb[idx]
        mask[i, :len(idx)] = True
    return CellBags(uniq.astype(object), bags, mask)
