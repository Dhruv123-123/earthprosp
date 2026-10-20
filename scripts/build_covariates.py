"""Add covariate families to an existing per-cell covariate cache, in place.

    python scripts/build_covariates.py data/covariates_4489547.parquet [family ...]

Families already in the cache are skipped, so this only computes what is new.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from earthprosp import covariates  # noqa: E402

path, fams = sys.argv[1], sys.argv[2:] or None
cells = pd.read_parquet(path, columns=["grav_bouguer"]).index.to_numpy()  # h3 is the index
cov = covariates.build(cells, families=fams, cache=path)
print(cov.shape)
