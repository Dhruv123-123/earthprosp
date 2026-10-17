import json
import os
import subprocess
import sys

import h3
import numpy as np
import pandas as pd

from earthprosp import commodities as C
from earthprosp import mineral_systems as MS

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_build_explorer_v3_packs_every_hexagon(tmp_path):
    rng = np.random.default_rng(0)
    h4 = sorted(h3.grid_disk(h3.latlng_to_cell(-23.0, -69.0, 4), 3))
    g = pd.DataFrame({"h4": h4, "p_prospective": rng.random(len(h4)) * 0.2, "score_rank": rng.random(len(h4)),
                      "uncertainty": rng.random(len(h4)) * 0.05, "known": rng.integers(0, 3, len(h4)),
                      "in_aoa": rng.random(len(h4)), "country": "Chile"})
    for c in C.CLASSES:
        g[f"p_{c}"] = rng.random(len(h4)) * 0.01
    for s in MS.SYSTEMS:
        g[f"sys_{s}"] = rng.random(len(h4))
    g.to_parquet(tmp_path / "global_v3_res4.parquet")
    ct = pd.DataFrame({"country": ["Chile"], "cells": [len(h4)], "mean_p": [0.1], "known_positive_cells": [3],
                       "p_prospective": [5.0], **{f"p_{c}": [0.1] for c in C.CLASSES},
                       **{f"sys_{s}": [0.2] for s in MS.SYSTEMS}})
    ct.to_csv(tmp_path / "by_country.csv", index=False)
    json.dump({"n_cells": 1234}, open(tmp_path / "run.json", "w"))
    land = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {
        "type": "Polygon", "coordinates": [[[-75, -30], [-65, -30], [-65, -15], [-75, -15], [-75, -30]]]}}]}
    json.dump(land, open(tmp_path / "land.geojson", "w"))
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_explorer_v3.py"), "--out", str(tmp_path),
                    "--land", str(tmp_path / "land.geojson")], check=True)
    html = open(tmp_path / "explorer.html").read()
    assert "__" not in html.split("<script>")[1].split("const B64")[0]  # every placeholder filled
    assert "Porphyry and skarn" in html and "1,234" in html
