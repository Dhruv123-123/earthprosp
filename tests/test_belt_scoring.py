import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import bootstrap_downscale  # noqa: E402
import pool_belts  # noqa: E402

from earthprosp import evaluate  # noqa: E402


def test_pool_score_against_uniform_cells():
    rng = np.random.default_rng(0)
    s = np.concatenate([rng.random(1000), np.full(10, 2.0)])  # 10 positives above every background cell
    pos = np.r_[np.zeros(1000, bool), np.ones(10, bool)]
    uni = ~pos
    r = pool_belts.score(s, pos, uni)
    assert r["capture_auc"] == 1.0 and r["top1"] == 1.0 and r["n_pos"] == 10
    r = pool_belts.score(rng.random(1010), pos, uni)  # random scores sit near 0.5
    assert 0.3 < r["capture_auc"] < 0.7


def test_bootstrap_capture_auc_matches_evaluate():
    rng = np.random.default_rng(1)
    s, y = rng.random(5000), rng.random(5000) < 0.02
    ref = evaluate.capture_curve(s, y)["capture_auc"]
    assert abs(bootstrap_downscale.capture_auc(s, y) - ref) < 1e-9
