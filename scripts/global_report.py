"""Maps and summary tables from outputs/global/global_split_res6.parquet.

    python scripts/global_report.py --out outputs/global
"""

import argparse
import json
import os
import sys

import h3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from earthprosp import commodities as C  # noqa: E402

CMAP = "tab20"


def raster(df, col, deg=0.1, how="mean"):
    ny, nx = int(180 / deg), int(360 / deg)
    iy = np.clip(((90 - df.lat.to_numpy()) / deg).astype(int), 0, ny - 1)
    ix = np.clip(((df.lon.to_numpy() + 180) / deg).astype(int), 0, nx - 1)
    flat = iy * nx + ix
    v = df[col].to_numpy(np.float64)
    s = np.bincount(flat, weights=v, minlength=ny * nx)
    n = np.bincount(flat, minlength=ny * nx)
    img = np.full(ny * nx, np.nan)
    img[n > 0] = s[n > 0] / n[n > 0]
    return img.reshape(ny, nx)


def country_of(df, geojson):
    import shapely
    from shapely.geometry import shape

    with open(geojson) as f:
        feats = json.load(f)["features"]
    geoms = [shape(ft["geometry"]) for ft in feats]
    names = np.array([ft["properties"]["ADMIN"] for ft in feats] + ["(none)"], dtype=object)
    tree = shapely.STRtree(geoms)
    pts = shapely.points(df.lon.to_numpy(), df.lat.to_numpy())
    pi, gi = tree.query(pts, predicate="within")
    out = np.full(len(df), len(geoms))
    out[pi] = gi
    return names[out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/global")
    ap.add_argument("--countries-geojson", default="data/ne_countries.geojson")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    df = pd.read_parquet(os.path.join(a.out, "global_split_res6.parquet"))
    ext = [-180, 180, -90, 90]
    df = df[df.lat > -60]  # Antarctica: AEF covers it, but no labels and no mining; drop from maps/tables

    # 1. prospectivity + dominant class
    fig, axes = plt.subplots(2, 1, figsize=(20, 19), constrained_layout=True)
    img = raster(df, "p_prospective")
    ax = axes[0]
    ax.set_facecolor("#0b1a2a")
    vmax = float(np.nanquantile(img, 0.99))
    im = ax.imshow(img, extent=ext, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.5, label="prospectivity score (rank; nnPU, not calibrated); scale clipped at 99th pct")
    ax.set_title("Global mineral prospectivity (any of 15 commodity classes), AlphaEarth 2024, H3 res 6", fontsize=14)
    ax = axes[1]
    ax.set_facecolor("#0b1a2a")
    cls_idx = df.top_class.cat.codes.to_numpy()
    tmp = df.assign(ci=cls_idx)
    # dominant class per pixel = argmax of mean marginals in the pixel
    marg = np.stack([raster(df, f"p_{c}") for c in C.CLASSES], -1)
    top = np.nanargmax(np.nan_to_num(marg, nan=-1), -1).astype(float)
    top[np.isnan(img)] = np.nan
    cmap = plt.get_cmap(CMAP)
    colors = cmap(np.arange(C.N_CLASSES) % 20)
    rgb = colors[np.nan_to_num(top, nan=0).astype(int), :3]
    alpha = np.clip(np.nan_to_num(img) / np.nanquantile(img, 0.97), 0.08, 1)[..., None]
    rgb = rgb * alpha + np.array([0.043, 0.102, 0.165]) * (1 - alpha)
    rgb[np.isnan(img)] = [0.043, 0.102, 0.165]
    ax.imshow(rgb, extent=ext, interpolation="nearest")
    for i, c in enumerate(C.CLASSES):
        ax.scatter([], [], color=colors[i], label=c, s=60)
    ax.legend(loc="lower left", ncol=5, fontsize=10, facecolor="white")
    ax.set_title("Dominant commodity class (brightness = prospectivity)", fontsize=14)
    for ax in axes:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 84)
    fig.savefig(os.path.join(a.out, "global_map.png"), dpi=110)
    plt.close(fig)

    # 2. per-class small multiples (marginal p * q_k)
    fig, axes = plt.subplots(5, 3, figsize=(24, 22), constrained_layout=True)
    for ax, (i, c) in zip(axes.flat, enumerate(C.CLASSES)):
        m = marg[..., i]
        ax.set_facecolor("#0b1a2a")
        ax.imshow(m, extent=ext, cmap="inferno", vmin=0, vmax=np.nanquantile(m, 0.995), interpolation="nearest")
        ax.set_title(f"P({c})", fontsize=13)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 84)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(os.path.join(a.out, "global_by_commodity.png"), dpi=80)
    plt.close(fig)

    # 3. country tables: expected number of prospective res-6 cells per class
    df["country"] = country_of(df, a.countries_geojson)
    agg = df.groupby("country")[["p_prospective"] + [f"p_{c}" for c in C.CLASSES]].sum()
    agg["cells"] = df.groupby("country").size()
    agg["mean_p"] = agg.p_prospective / agg.cells
    agg["known_positive_cells"] = df.groupby("country").known_positive.sum()
    agg = agg.drop(index="(none)", errors="ignore").sort_values("p_prospective", ascending=False)
    agg.round(4).to_csv(os.path.join(a.out, "by_country.csv"))
    top = {c: agg[f"p_{c}"].sort_values(ascending=False).head(8).round(0).to_dict() for c in C.CLASSES}
    with open(os.path.join(a.out, "top_countries_by_class.json"), "w") as f:
        json.dump(top, f, indent=1)

    # 4. greenfield candidates: high marginal, no known deposit within ~25 km (ring 6)
    known = set(df.h3[df.known_positive])
    rows = []
    for c in C.CLASSES:
        cand = df.nlargest(4000, f"p_{c}")
        cand = cand[~cand.known_positive]
        keep = [not any(n in known for n in h3.grid_disk(h, 6)) for h in cand.h3]
        cand = cand[keep].head(25)
        for r in cand.itertuples():
            rows.append({"class": c, "h3": r.h3, "lat": round(r.lat, 3), "lon": round(r.lon, 3),
                         "country": r.country, "p_class": round(getattr(r, f"p_{c}"), 3),
                         "p_prospective": round(r.p_prospective, 3)})
    gc = pd.DataFrame(rows)
    # Flag candidates near large cities: urban land looks "disturbed" to AEF, like mines do.
    places = "data/ne_populated_places.geojson"
    if os.path.exists(places) and len(gc):
        from sklearn.neighbors import BallTree
        feats = json.load(open(places))["features"]
        big = [(f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0]) for f in feats
               if (f["properties"].get("pop_max") or 0) >= 250_000]
        tree = BallTree(np.radians(np.array(big)), metric="haversine")
        d = tree.query(np.radians(gc[["lat", "lon"]].to_numpy()), k=1)[0][:, 0] * 6371
        gc["km_to_city_250k"] = np.round(d, 1)
        gc["urban_flag"] = d < 30
    gc.to_csv(os.path.join(a.out, "greenfield_candidates.csv"), index=False)

    # 5. res-4 aggregation for an interactive map
    df["h4"] = [h3.cell_to_parent(h, 4) for h in df.h3]
    cols = ["p_prospective"] + [f"p_{c}" for c in C.CLASSES]
    g = df.groupby("h4")[cols].mean()
    g["known"] = df.groupby("h4").known_positive.sum()
    g["country"] = df.groupby("h4").country.agg(lambda x: x.value_counts().index[0])
    g.round(4).reset_index().to_parquet(os.path.join(a.out, "global_split_res4.parquet"), index=False)
    print(agg.head(25)[["cells", "mean_p", "p_prospective", "known_positive_cells"]].round(2).to_string())
    print(json.dumps({c: list(v)[:5] for c, v in top.items()}, indent=0))


if __name__ == "__main__":
    main()
