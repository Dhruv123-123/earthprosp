"""Maps and tables for the v3 global run.

    python scripts/report_v3.py --out outputs/v3

Writes: v3_map.png (presence; dominant mineral system), v3_uncertainty.png (bag spread
and area of applicability), v3_systems.png (one panel per system), by_country.csv,
greenfield_candidates.csv (with system and top drivers), global_v3_res4.parquet.
"""

import argparse
import json
import os
import sys

import h3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from earthprosp import commodities as C  # noqa: E402
from earthprosp import mineral_systems as MS  # noqa: E402
from global_report import country_of, raster  # noqa: E402

BG = np.array([0.043, 0.102, 0.165])
SYS_COLORS = ["#e6550d", "#fdd0a2", "#31a354", "#3182bd", "#756bb1", "#e7298a", "#a6761d", "#66c2a5", "#ffff33"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/v3")
    ap.add_argument("--countries-geojson", default="data/raw/ne_50m_admin_0_countries.geojson")
    ap.add_argument("--change", nargs=2, default=["data/global_2018.npz", "data/global_2024.npz"],
                    help="two AEF years for the surface-change leakage diagnostic (skipped if missing)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    df = pd.read_parquet(os.path.join(a.out, "global_v3_res6.parquet"))
    df = df[df.lat > -60]
    ext = [-180, 180, -90, 90]

    img = raster(df, "score_rank")
    fig, axes = plt.subplots(2, 1, figsize=(20, 19), constrained_layout=True)
    ax = axes[0]
    ax.set_facecolor(BG)
    im = ax.imshow(img, extent=ext, cmap="magma", vmin=0.5, vmax=1.0, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.5, label="prospectivity percentile among land cells")
    ax.set_title("v3 global mineral prospectivity: AlphaEarth 2024 + gravity, lithosphere, faults, plates, "
                 "land cover, terrain, Sentinel-2 (H3 res 6)", fontsize=13)
    ax = axes[1]
    ax.set_facecolor(BG)
    sysm = np.stack([raster(df, f"sys_{s}") for s in MS.SYSTEMS], -1)
    top = np.nanargmax(np.nan_to_num(sysm, nan=-1), -1)
    cols = np.array([to_rgb(c) for c in SYS_COLORS])
    rgb = cols[top]
    alpha = np.clip((np.nan_to_num(img) - 0.6) / 0.35, 0.05, 1)[..., None]
    rgb = rgb * alpha + BG * (1 - alpha)
    rgb[np.isnan(img)] = BG
    ax.imshow(rgb, extent=ext, interpolation="nearest")
    for i, s in enumerate(MS.SYSTEMS):
        ax.scatter([], [], color=SYS_COLORS[i], label=s.replace("_", " "), s=60)
    ax.legend(loc="lower left", ncol=3, fontsize=10, facecolor="white")
    ax.set_title("Most likely mineral system (brightness = prospectivity)", fontsize=14)
    for ax in axes:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 84)
    fig.savefig(os.path.join(a.out, "v3_map.png"), dpi=100)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(20, 19), constrained_layout=True)
    u = raster(df, "uncertainty")
    axes[0].set_facecolor(BG)
    im = axes[0].imshow(u, extent=ext, cmap="viridis", vmin=0, vmax=np.nanquantile(u, 0.99), interpolation="nearest")
    fig.colorbar(im, ax=axes[0], shrink=0.5, label="std of presence score across PU bags")
    axes[0].set_title("Model uncertainty (disagreement between bagged models)", fontsize=14)
    df["out_aoa"] = (~df.in_aoa).astype(float)
    o = raster(df, "out_aoa")
    axes[1].set_facecolor(BG)
    im = axes[1].imshow(o, extent=ext, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im, ax=axes[1], shrink=0.5, label="fraction of cells outside the area of applicability")
    axes[1].set_title("Where the model extrapolates (Meyer & Pebesma dissimilarity index above the training threshold)",
                      fontsize=14)
    for ax in axes:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 84)
    fig.savefig(os.path.join(a.out, "v3_uncertainty.png"), dpi=90)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(24, 14), constrained_layout=True)
    pres = raster(df, "p_prospective")
    for ax, (i, s) in zip(axes.flat, enumerate(MS.SYSTEMS)):
        m = sysm[..., i] * pres
        ax.set_facecolor(BG)
        ax.imshow(m, extent=ext, cmap="inferno", vmin=0, vmax=np.nanquantile(m, 0.995), interpolation="nearest")
        ax.set_title(f"P(deposit) x P({s.replace('_', ' ')} | deposit)", fontsize=12)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 84)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(os.path.join(a.out, "v3_systems.png"), dpi=80)
    plt.close(fig)

    df["country"] = country_of(df, a.countries_geojson)
    marg = [f"p_{c}" for c in C.CLASSES]
    agg = df.groupby("country")[["p_prospective"] + marg].sum()
    for s_ in MS.SYSTEMS:  # expected prospective cells per mineral system
        agg[f"sys_{s_}"] = (df.p_prospective * df[f"sys_{s_}"]).groupby(df.country).sum()
    agg["cells"] = df.groupby("country").size()
    agg["mean_p"] = agg.p_prospective / agg.cells
    agg["known_positive_cells"] = df.groupby("country").known_positive.sum()
    agg["share_in_aoa"] = df.groupby("country").in_aoa.mean()
    agg = agg.drop(index="(none)", errors="ignore").sort_values("p_prospective", ascending=False)
    agg.round(4).to_csv(os.path.join(a.out, "by_country.csv"))

    # Greenfield: top cells per system with no mineral record of any kind within ~25 km.
    known = set(df.h3[df.any_record])
    drivers_path = os.path.join(a.out, "top1pct_drivers.parquet")
    drv = pd.read_parquet(drivers_path).set_index("h3") if os.path.exists(drivers_path) else None
    rows = []
    for s in MS.SYSTEMS:
        score = df.p_prospective * df[f"sys_{s}"]
        cand = df.assign(_s=score).nlargest(5000, "_s")
        cand = cand[~cand.any_record & cand.in_aoa]
        keep = [not any(n in known for n in h3.grid_disk(h, 6)) for h in cand.h3]
        for r in cand[keep].head(20).itertuples():
            row = {"system": s, "h3": r.h3, "lat": round(r.lat, 3), "lon": round(r.lon, 3), "country": r.country,
                   "p_prospective": round(r.p_prospective, 3), "p_system": round(getattr(r, f"sys_{s}"), 3),
                   "uncertainty": round(r.uncertainty, 3), "top_class": r.top_class}
            if drv is not None and r.h3 in drv.index:
                d = drv.loc[r.h3]
                row.update({f"driver{j}": d[f"driver{j}"] for j in (1, 2, 3)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "greenfield_candidates.csv"), index=False)

    df["h4"] = [h3.cell_to_parent(h, 4) for h in df.h3]
    cols = ["p_prospective", "score_rank", "uncertainty"] + marg + [f"sys_{s}" for s in MS.SYSTEMS]
    g = df.groupby("h4")[cols].mean()
    g["known"] = df.groupby("h4").known_positive.sum()
    g["in_aoa"] = df.groupby("h4").in_aoa.mean()
    g["country"] = df.groupby("h4").country.agg(lambda x: x.value_counts().index[0])
    g.round(4).reset_index().to_parquet(os.path.join(a.out, "global_v3_res4.parquet"), index=False)
    print(agg.head(25)[["cells", "mean_p", "p_prospective", "known_positive_cells", "share_in_aoa"]].round(3).to_string())
    with open(os.path.join(a.out, "top_countries_by_system.json"), "w") as f:
        sys_country = df.assign(**{f"e_{s}": df.p_prospective * df[f"sys_{s}"] for s in MS.SYSTEMS}) \
            .groupby("country")[[f"e_{s}" for s in MS.SYSTEMS]].sum()
        json.dump({s: sys_country[f"e_{s}"].drop(index="(none)", errors="ignore").nlargest(8).round(0).to_dict()
                   for s in MS.SYSTEMS}, f, indent=1)
    if all(os.path.exists(p) for p in a.change):
        change_diagnostic(df, a)


def change_diagnostic(df, a):
    """How much of the top of the map is recent surface change (pits, tailings, clearing)?"""
    from scipy.stats import spearmanr

    from earthprosp import aef_change
    ch = aef_change.change(*a.change)["aef_change_cos"]
    d = df[["h3", "p_prospective", "known_positive"]].assign(change=ch.reindex(df.h3.to_numpy()).to_numpy())
    d = d.dropna(subset=["change"])
    hi_change = d.change >= d.change.quantile(0.95)
    out = {"years": a.change, "cells": int(len(d)),
           "spearman_score_vs_change": float(spearmanr(d.p_prospective, d.change).statistic)}
    for f in (0.01, 0.05):
        top = d.p_prospective >= d.p_prospective.quantile(1 - f)
        out[f"top{int(f * 100)}pct_share_in_top5pct_change"] = float((top & hi_change).sum() / top.sum())
        out[f"top{int(f * 100)}pct_unlabelled_share_in_top5pct_change"] = float(
            (top & hi_change & ~d.known_positive).sum() / max((top & ~d.known_positive).sum(), 1))
    out["known_positive_share_in_top5pct_change"] = float((hi_change & d.known_positive).sum() / d.known_positive.sum())
    json.dump(out, open(os.path.join(a.out, "change_diagnostic.json"), "w"), indent=2)
    print("surface-change diagnostic:", out)


if __name__ == "__main__":
    main()
