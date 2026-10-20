"""Turn a cell map into a list of separate prospects.

Neighbouring top cells usually describe one anomaly. Greedy non-maximum suppression keeps
the best cell, drops every candidate within `min_km` of it, and repeats.
"""

import numpy as np


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nms(lat, lon, score, min_km=10.0, n=50):
    """Indices of up to n cells, best first, no two closer than min_km."""
    lat, lon, score = map(np.asarray, (lat, lon, score))
    order = np.argsort(-score, kind="stable")
    alive = np.ones(len(order), bool)
    keep = []
    for pos, i in enumerate(order):
        if not alive[pos]:
            continue
        keep.append(int(i))
        if len(keep) == n:
            break
        rest = order[pos + 1:]
        alive[pos + 1:] &= haversine_km(lat[i], lon[i], lat[rest], lon[rest]) >= min_km
    return np.array(keep, int)


def footprint(lat, lon, centre_idx, km):
    """Number of candidate cells within km of each kept centre."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    return np.array([int((haversine_km(lat[i], lon[i], lat, lon) < km).sum()) for i in centre_idx])
