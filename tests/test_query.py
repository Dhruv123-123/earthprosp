import h3
import pandas as pd

from earthprosp import query


def _table():
    cells = [h3.latlng_to_cell(-22.3, -68.9, 4), h3.latlng_to_cell(10.0, 10.0, 4)]
    return pd.DataFrame({"h4": cells, "p_prospective": [0.3, 0.01], "score_rank": [0.99, 0.2],
                         "uncertainty": [0.05, 0.01], "in_aoa": [1.0, 0.0], "p_Cu": [0.2, 0.0],
                         "p_Au": [0.05, 0.001], "sys_porphyry_skarn_cu": [0.8, 0.1],
                         "sys_vms": [0.1, 0.5]}).set_index("h4")


def test_point_and_area():
    df = _table()
    r = query.point(df, 4, -22.3, -68.9)
    assert r["found"] and list(r["commodities"])[0] == "Cu" and list(r["mineral_systems"])[0] == "porphyry_skarn_cu"
    assert not query.point(df, 4, 50.0, 50.0)["found"]
    top = query.area(df, 4, (-75, -30, -60, -15), top=5, system="porphyry_skarn_cu")
    assert len(top) == 1 and abs(top.score.iloc[0] - 0.24) < 1e-6
