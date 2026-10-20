"""Stream global geographic COG mosaics onto H3 cells (land cover, DEM, Sentinel-2 bands).

Same pattern as global_agg (AEF): every tile is read at a coarse overview level in a
worker process, each pixel is turned into a few per-pixel quantities, and those are
reduced to per-cell partial sums written in shards (resumable), then merged.

Products (all public, anonymous S3, EPSG:4326 tiles):

* `worldcover` ESA WorldCover 10 m 2021 v200, read at 1/32 (~300 m): per-cell fraction of
  each of the 11 classes. Built-up and bare fractions are the confounds the v2 model
  tripped on (cities and bare rock scoring like mines); cropland and tree cover say how
  much of the surface the embedding can actually see.
* `dem` Copernicus GLO-90, read at 1/4 (~360 m): elevation mean / std / min / max and
  slope mean / std. Relief and slope are first-order controls on exposure and erosion
  level.
* `s2` ESA WorldCover Sentinel-2 2020 annual median composites (RGBNIR + SWIR), read at
  1/8 (~150 m): classic alteration band ratios on bare pixels (NDVI < 0.25):
  clay / AlOH-MgOH (B11/B12), ferric iron (B4/B2), ferrous iron (B12/B8 + B3/B4),
  gossan (B11/B4), plus the bare-pixel fraction (van der Meer et al. 2014;
  Sabins 1999). These are the explicit mineralogy layers the embedding lacks.
"""

from __future__ import annotations

import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

import h3
import numpy as np
import pandas as pd

WC_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 70: "snow",
              80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}

PRODUCTS = {
    "worldcover": dict(
        bucket="https://esa-worldcover.s3.eu-central-1.amazonaws.com", prefix="v200/2021/map/",
        overview_level=4, columns=[f"wc_{v}" for v in WC_CLASSES.values()] + ["wc_valid"], extra=()),
    "dem": dict(
        bucket="https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com", prefix="",
        overview_level=1, columns=["dem_n", "dem_sum", "dem_sumsq", "slope_sum", "slope_sumsq"],
        extra=("dem_min", "dem_max")),
    "s2": dict(
        bucket="https://esa-worldcover-s2.s3.eu-central-1.amazonaws.com", prefix="swir/2020/",
        overview_level=2, columns=["s2_n", "s2_bare_n", "s2_clay", "s2_ferric", "s2_ferrous", "s2_gossan",
                                   "s2_clay_sq", "s2_ferric_sq", "s2_ndvi"], extra=()),
}


def _env():
    import rasterio
    return rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                        GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="1", GDAL_CACHEMAX=256)


def _read(url, level, bands=None):
    import rasterio
    with _env(), rasterio.open("/vsicurl/" + url, overview_level=level) as src:
        a = src.read(bands) if bands else src.read()
        return a, src.transform, src.nodata


def _pixel_latlon(transform, h, w):
    cols = np.arange(w) + 0.5
    rows = np.arange(h) + 0.5
    lon = transform.c + cols * transform.a
    lat = transform.f + rows * transform.e
    return np.repeat(lat, w), np.tile(lon, h)


def _cells(lat, lon, res=6):
    return np.array([h3.latlng_to_cell(a, b, res) for a, b in zip(lat.tolist(), lon.tolist())], dtype=object)


def _per_pixel(product, url, level):
    """(lat, lon, values (n, k), extra_min_max or None) for one tile."""
    if product == "worldcover":
        a, t, nd = _read(url, level)
        a = a[0]
        lat, lon = _pixel_latlon(t, *a.shape)
        v = a.ravel()
        ok = v != 0
        cols = [(v == c).astype(np.float32) for c in WC_CLASSES]
        cols.append(np.ones_like(cols[0]))
        return lat[ok], lon[ok], np.stack(cols, 1)[ok], None
    if product == "dem":
        a, t, nd = _read(url, level)
        z = a[0].astype(np.float64)
        bad = ~np.isfinite(z) | (z < -1000)
        if nd is not None:
            bad |= z == nd
        lat, lon = _pixel_latlon(t, *z.shape)
        # Slope in degrees with a latitude-aware east-west spacing.
        dy = abs(t.e) * 110_574.0
        dx = abs(t.a) * 111_320.0 * np.cos(np.radians(lat.reshape(z.shape)))
        zf = np.where(bad, np.nan, z)
        gy, gx = np.gradient(zf)
        slope = np.degrees(np.arctan(np.hypot(gx / np.maximum(dx, 1.0), gy / dy)))
        ok = (~bad).ravel() & np.isfinite(slope).ravel()
        zz, ss = z.ravel()[ok], slope.ravel()[ok]
        vals = np.stack([np.ones_like(zz), zz, zz ** 2, ss, ss ** 2], 1)
        return lat[ok], lon[ok], vals, np.stack([zz, zz], 1)
    if product == "s2":
        swir, t, _ = _read(url, level)
        rgb_url = url.replace("/swir/", "/rgbnir/").replace("_SWIR.tif", "_S2RGBNIR.tif")
        rgb, t2, _ = _read(rgb_url, level)
        h = min(swir.shape[1], rgb.shape[1])
        w = min(swir.shape[2], rgb.shape[2])
        # Reflectance: RGBNIR is uint16 x 1e4; SWIR is uint8 with an (undocumented) scale of
        # ~1/250 that matches B11/B4 of desert pixels. Same-file ratios are scale-free.
        b11, b12 = swir[0, :h, :w].astype(np.float32) / 250, swir[1, :h, :w].astype(np.float32) / 250
        b4, b3, b2, b8 = (rgb[i, :h, :w].astype(np.float32) / 1e4 for i in range(4))  # B04, B03, B02, B08
        valid = (swir[0, :h, :w] < 255) & (swir[1, :h, :w] < 255) & (b8 > 0) & (b4 > 0) & (b2 > 0) & (b12 > 0)
        lat, lon = _pixel_latlon(t, h, w)
        ndvi = (b8 - b4) / np.maximum(b8 + b4, 1e-4)
        bare = valid & (ndvi < 0.25)
        eps = 1e-3
        clay = b11 / (b12 + eps)
        ferric = b4 / (b2 + eps)
        ferrous = b12 / (b8 + eps) + b3 / (b4 + eps)
        gossan = b11 / (b4 + eps)
        bf = bare.astype(np.float32)
        vals = np.stack([np.ones_like(bf), bf, clay * bf, ferric * bf, ferrous * bf, gossan * bf,
                         clay ** 2 * bf, ferric ** 2 * bf, ndvi], -1)[valid]
        return lat[valid.ravel()], lon[valid.ravel()], vals.reshape(-1, vals.shape[-1]), None
    raise ValueError(product)


def tile_partial(args):
    product, url, level, res = args
    for attempt in range(3):
        try:
            lat, lon, vals, mm = _per_pixel(product, url, level)
            break
        except Exception as exc:  # transient HTTP failure or a missing companion file
            if attempt == 2:
                return url, None, f"{exc.__class__.__name__}: {exc}"
            time.sleep(2 * (attempt + 1))
    if len(lat) == 0:
        return url, (np.empty(0, object), np.empty((0, vals.shape[1]), np.float64), None), None
    codes, uniq = pd.factorize(_cells(lat, lon, res))
    S = np.zeros((len(uniq), vals.shape[1]), np.float64)
    np.add.at(S, codes, vals)
    MM = None
    if mm is not None:
        df = pd.DataFrame({"c": codes, "mn": mm[:, 0], "mx": mm[:, 1]}).groupby("c")
        MM = np.stack([df["mn"].min().to_numpy(), df["mx"].max().to_numpy()], 1)
    return url, (np.asarray(uniq, object), S, MM), None


def list_tiles(product: str, limit: int | None = None) -> list[str]:
    """All tile URLs of a product from the public S3 listing."""
    import re
    import urllib.parse
    import urllib.request
    p = PRODUCTS[product]
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": p["prefix"], "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(p["bucket"] + "/?" + urllib.parse.urlencode(q), timeout=60) as r:
            body = r.read().decode()
        keys += re.findall(r"<Key>([^<]+)</Key>", body)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", body)
        if not m or (limit and len(keys) >= limit * 20):
            break
        token = m.group(1)
    if product == "dem":
        keys = [k for k in keys if k.endswith("_DEM.tif") and "/AUXFILES/" not in k]
    elif product == "worldcover":
        keys = [k for k in keys if k.endswith("_Map.tif")]
    elif product == "s2":
        keys = [k for k in keys if k.endswith("_SWIR.tif")]
    urls = [f"{p['bucket']}/{k}" for k in sorted(keys)]
    return urls[:limit] if limit else urls


def tile_overlaps(url: str, bbox) -> bool:
    """Whether a tile (named by its lower-left corner, e.g. N36W117; at most 3 degrees square)
    can overlap bbox = (lon0, lat0, lon1, lat1)."""
    import re
    m = re.search(r"([NS])(\d{2})(?:_00)?_?([EW])(\d{3})", os.path.basename(url))
    if not m:
        return True
    lat = int(m.group(2)) * (1 if m.group(1) == "N" else -1)
    lon = int(m.group(4)) * (1 if m.group(3) == "E" else -1)
    x0, y0, x1, y1 = bbox
    return lon < x1 and lon + 3 > x0 and lat < y1 and lat + 3 > y0


def run(product: str, urls: list[str], shard_dir: str, n_shards: int = 40, workers: int = 8, res: int = 6,
        level: int | None = None):
    os.makedirs(shard_dir, exist_ok=True)
    level = PRODUCTS[product]["overview_level"] if level is None else level
    parts = np.array_split(np.arange(len(urls)), n_shards)
    with ProcessPoolExecutor(workers) as ex:
        for k, idx in enumerate(parts):
            out = os.path.join(shard_dir, f"shard_{k:03d}.npz")
            if os.path.exists(out) or len(idx) == 0:
                continue
            t0 = time.time()
            acc, failed = [], []
            for url, part, err in ex.map(tile_partial, [(product, urls[i], level, res) for i in idx], chunksize=2):
                if part is None:
                    failed.append(f"{url}\t{err}")
                elif len(part[0]):
                    acc.append(part)
            cells, S, MM = _reduce(acc)
            np.savez(out, cells=cells.astype(str), S=S, MM=MM if MM is not None else np.empty((0, 2)),
                     failed=np.array(failed, dtype=str))
            print(f"[{product}] shard {k + 1}/{n_shards}: {len(idx)} tiles, {len(cells):,} cells, "
                  f"{len(failed)} failed, {time.time() - t0:.0f}s", flush=True)


def _reduce(acc):
    if not acc:
        return np.empty(0, object), np.empty((0, 1)), None
    cells = np.concatenate([a[0] for a in acc])
    S = np.concatenate([a[1] for a in acc])
    codes, uniq = pd.factorize(cells)
    out = np.zeros((len(uniq), S.shape[1]), np.float64)
    np.add.at(out, codes, S)
    MM = None
    if acc[0][2] is not None:
        mm = np.concatenate([a[2] for a in acc])
        df = pd.DataFrame({"c": codes, "mn": mm[:, 0], "mx": mm[:, 1]}).groupby("c")
        MM = np.stack([df["mn"].min().to_numpy(), df["mx"].max().to_numpy()], 1)
    return np.asarray(uniq, object), out, MM


def merge(product: str, shard_dir: str) -> pd.DataFrame:
    """Per-cell features for a product, indexed by H3 id."""
    zs = [np.load(f, allow_pickle=False) for f in sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))]
    acc = [(z["cells"].astype(object), z["S"], z["MM"] if len(z["MM"]) else None) for z in zs if len(z["cells"])]
    cells, S, MM = _reduce(acc)
    cols = PRODUCTS[product]["columns"]
    raw = pd.DataFrame(S, index=pd.Index(cells, name="h3"), columns=cols)
    out = pd.DataFrame(index=raw.index)
    if product == "worldcover":
        n = raw["wc_valid"].clip(lower=1)
        for c in cols[:-1]:
            out[c] = raw[c] / n
    elif product == "dem":
        n = raw["dem_n"].clip(lower=1)
        out["dem_mean"] = raw.dem_sum / n
        out["dem_std"] = np.sqrt(np.maximum(raw.dem_sumsq / n - out.dem_mean ** 2, 0))
        out["slope_mean"] = raw.slope_sum / n
        out["slope_std"] = np.sqrt(np.maximum(raw.slope_sumsq / n - out.slope_mean ** 2, 0))
        out["dem_min"], out["dem_max"] = MM[:, 0], MM[:, 1]
        out["dem_relief"] = out.dem_max - out.dem_min
    elif product == "s2":
        n, nb = raw["s2_n"].clip(lower=1), raw["s2_bare_n"]
        out["s2_bare_frac"] = nb / n
        out["s2_ndvi"] = raw.s2_ndvi / n
        nbc = nb.clip(lower=1)
        for c in ("clay", "ferric", "ferrous", "gossan"):
            out[f"s2_{c}"] = np.where(nb > 0, raw[f"s2_{c}"] / nbc, np.nan)
        for c in ("clay", "ferric"):
            out[f"s2_{c}_std"] = np.where(nb > 1, np.sqrt(np.maximum(raw[f"s2_{c}_sq"] / nbc - out[f"s2_{c}"] ** 2, 0)),
                                          np.nan)
    return out.astype(np.float32)
