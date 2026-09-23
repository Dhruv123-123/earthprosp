"""Global discrete grid (H3) helpers and spatial blocking for cross-validation."""

from __future__ import annotations

import h3
import numpy as np

# Resolution 6: ~36 km^2 hexagons (~3.7 km edge). ~4.1M cells cover the land
# surface. Fine enough to resolve a district, coarse enough that "does this cell
# host a known deposit" is a meaningful label.
DEFAULT_RES = 6


def latlng_to_cells(lat: np.ndarray, lon: np.ndarray, res: int = DEFAULT_RES) -> np.ndarray:
    return np.array([h3.latlng_to_cell(a, b, res) for a, b in zip(lat, lon)], dtype=object)


def cell_centers(cells) -> tuple[np.ndarray, np.ndarray]:
    ll = np.array([h3.cell_to_latlng(c) for c in cells])
    return ll[:, 0], ll[:, 1]


def spatial_blocks(cells, block_res: int = 3) -> np.ndarray:
    """Parent cell at a coarse resolution, used as the CV group.

    Res 3 parents are ~12,000 km^2: large enough that a held-out block is not
    trivially predictable from deposits a few km away in the training set.
    """
    return np.array([h3.cell_to_parent(c, block_res) for c in cells], dtype=object)


def in_country(cells, geojson_path: str, iso_a3: str) -> np.ndarray:
    """Boolean mask of cells whose centre falls inside a Natural Earth country polygon."""
    import json

    import shapely
    from shapely.geometry import shape

    with open(geojson_path) as f:
        feats = json.load(f)["features"]
    geom = shapely.union_all([shape(ft["geometry"]) for ft in feats
                              if ft["properties"].get("ADM0_A3") == iso_a3])
    lat, lon = cell_centers(cells)
    return shapely.contains_xy(geom, lon, lat)
