"""Train on all labeled cells and write the per-cell mineral split (+ a quick-look map).

    python scripts/predict_map.py --bags data/westus_2024.npz --train-country USA --out outputs/westus

Output parquet columns: h3, lat, lon, p_prospective, p_none, p_<class>..., top_class.
Cells in `--bags` outside the training country are still predicted (extrapolation).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from earthprosp import aef, features, grid, labels  # noqa: E402
from earthprosp import commodities as C  # noqa: E402
from earthprosp.train import TrainConfig, fit, predict  # noqa: E402


def plot(df: pd.DataFrame, path: str, known=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    ax = axes[0]
    sc = ax.scatter(df.lon, df.lat, c=df.p_prospective, s=0.6, cmap="magma", marker="h", linewidths=0)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="P(prospective)")
    if known is not None:
        ax.scatter(known.lon, known.lat, s=0.3, c="cyan", alpha=0.25, linewidths=0, label="producer cells")
        ax.legend(loc="lower left", markerscale=10)
    ax.set_title("Prospectivity (any target commodity)")
    ax = axes[1]
    idx = df.top_class.map(C.CLASS_INDEX).to_numpy()
    alpha = np.clip(df.p_prospective.to_numpy() / np.quantile(df.p_prospective, 0.98), 0.05, 1)
    colors = cmap(idx % 20)
    colors[:, 3] = alpha
    ax.scatter(df.lon, df.lat, c=colors, s=0.6, marker="h", linewidths=0)
    for i, c in enumerate(C.CLASSES):
        ax.scatter([], [], color=cmap(i % 20), label=c, s=30)
    ax.legend(loc="lower left", ncol=3, fontsize=8)
    ax.set_title("Dominant commodity class (opacity = prospectivity)")
    for ax in axes:
        ax.set_aspect(1 / np.cos(np.radians(df.lat.mean())))
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", required=True)
    ap.add_argument("--mrds", default="data/mrds.csv")
    ap.add_argument("--train-country", default=None)
    ap.add_argument("--countries-geojson", default="data/ne_countries.geojson")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--pretrain-epochs", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cb = aef.CellBags.load(a.bags)
    lab = labels.align(cb.cells, labels.cell_labels(labels.load_mrds(a.mrds)))
    ctx = features.neighbourhood_context(cb.cells, cb.mean_embedding())
    cov = ((ctx - ctx.mean(0)) / (ctx.std(0) + 1e-6)).astype(np.float32)
    tr = (grid.in_country(cb.cells, a.countries_geojson, a.train_country)
          if a.train_country else np.ones(len(cb.cells), bool))

    cfg = TrainConfig(epochs=a.epochs, pretrain_epochs=a.pretrain_epochs, verbose=True)
    model = fit(cb.bags[tr], cb.mask[tr], lab["positive"][tr], lab["conf"][tr], lab["composition"][tr],
                cov[tr], cfg, unlabeled_bags=cb.bags[~tr], unlabeled_mask=cb.mask[~tr], unlabeled_cov=cov[~tr])
    split = predict(model, cb.bags, cb.mask, cov)

    lat, lon = grid.cell_centers(cb.cells)
    df = pd.DataFrame({"h3": cb.cells.astype(str), "lat": lat, "lon": lon,
                       "p_prospective": 1 - split[:, 0], "p_none": split[:, 0]})
    for i, c in enumerate(C.CLASSES):
        df[f"p_{c}"] = split[:, i + 1]
    df["top_class"] = np.array(C.CLASSES)[split[:, 1:].argmax(1)]
    df["in_training_region"] = tr
    df["known_positive"] = lab["positive"]
    df.to_parquet(os.path.join(a.out, "split.parquet"), index=False)
    plot(df, os.path.join(a.out, "map.png"), known=df[lab["conf"] >= 1.0])
    print(df.describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
