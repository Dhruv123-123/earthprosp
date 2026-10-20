"""Build the global feature caches run_v3 needs (AEF context features, covariates), so the
expensive one-off steps can run while other jobs still stream rasters."""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import run_v3  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stats", default="data/global_2024.npz")
ap.add_argument("--data", default="data")
ap.add_argument("--features-cache", default="data/global_features.npz")
a = ap.parse_args()
t0 = time.time()
cells, X = run_v3.aef_features(a.stats, a.features_cache)
run_v3.log(f"AEF features {X.shape} ({time.time() - t0:.0f}s)")
X, cols = run_v3.feature_matrix(cells, X, a.data)
run_v3.log(f"features {X.shape}, NaN share {np.isnan(X[::50, 256:]).mean():.2%} ({time.time() - t0:.0f}s)")
