"""Render a res-8 map (downscale.py output) as a raster image.

    python scripts/render_res8.py outputs/downscale_wus
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main(out_dir, cell_deg=0.012):
    df = pd.read_parquet(os.path.join(out_dir, "map_res8.parquet"))
    top = pd.read_csv(os.path.join(out_dir, "greenfield_targets_res8.csv"))
    x0, x1, y0, y1 = df.lon.min(), df.lon.max(), df.lat.min(), df.lat.max()
    nx, ny = int((x1 - x0) / cell_deg) + 1, int((y1 - y0) / cell_deg) + 1
    ix = ((df.lon - x0) / cell_deg).astype(int).to_numpy()
    iy = ((df.lat - y0) / cell_deg).astype(int).to_numpy()
    img = np.full((ny, nx), np.nan, np.float32)
    np.fmax.at(img, (iy, ix), df.blend.to_numpy(np.float32))  # brightest cell per pixel
    known = np.zeros((ny, nx), bool)
    k = df.known_positive.to_numpy()
    known[iy[k], ix[k]] = True
    aspect = 1 / np.cos(np.radians((y0 + y1) / 2))
    fig, ax = plt.subplots(figsize=(10, 10 * (y1 - y0) * aspect / (x1 - x0) + 0.6), dpi=170)
    ax.set_facecolor("#0b0b0b")
    im = ax.imshow(img, origin="lower", extent=(x0, x1, y0, y1), cmap="magma", vmin=0.6, vmax=1.0,
                   interpolation="nearest", aspect=aspect)
    yy, xx = np.nonzero(known)
    ax.scatter(x0 + (xx + 0.5) * cell_deg, y0 + (yy + 0.5) * cell_deg, s=0.4, c="#38bdf8", linewidths=0,
               label="known deposit cells")
    ax.scatter(top.lon, top.lat, s=40, facecolors="none", edgecolors="#4ade80", linewidths=1.0,
               label="top 50 greenfield cells (no record within 3 km)")
    ax.legend(loc="lower left", fontsize=8, markerscale=4, framealpha=0.85)
    fig.colorbar(im, ax=ax, shrink=0.6, label="prospectivity rank (fine model x global district)")
    ax.set_title("Res-8 (~0.7 km²) prospectivity", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "map_res8.png"))


if __name__ == "__main__":
    main(sys.argv[1])
