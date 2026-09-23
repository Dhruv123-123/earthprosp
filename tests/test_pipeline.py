import numpy as np
import pandas as pd
import torch

from earthprosp import aef, commodities as C, evaluate, features, grid, labels, synthetic
from earthprosp.model import ProspectivityModel, nnpu_loss
from earthprosp.train import TrainConfig, fit, predict, spatial_folds


def test_dequantize_matches_spec():
    raw = np.array([-128, -127, 0, 64, 127], np.int8)
    out = aef.dequantize(raw)
    assert np.isnan(out[0])
    np.testing.assert_allclose(out[1:], [-(127 / 127.5) ** 2, 0, (64 / 127.5) ** 2, (127 / 127.5) ** 2], rtol=1e-6)


def test_pixels_to_bags_groups_by_cell():
    lat = np.array([40.0, 40.0001, 40.0002, 40.0003, 10.0])
    lon = np.array([-117.0, -117.0001, -117.0, -117.0002, 10.0])
    emb = np.eye(64, dtype=np.float32)[:5]
    cb = aef.pixels_to_bags(aef.TilePixels(lat, lon, emb), bag_size=8, min_pixels=2)
    assert len(cb.cells) == 1  # the lone pixel at (10, 10) is dropped
    assert cb.mask[0].sum() == 4


def test_cell_labels_composition():
    df = pd.DataFrame({
        "latitude": [40.0, 40.0001, 10.0],
        "longitude": [-117.0, -117.0001, 10.0],
        "commod1": ["Gold", "Copper", "Sand and Gravel"],
        "commod2": ["Silver", None, None],
        "commod3": [None, None, None],
        "dev_stat": ["Producer", "Occurrence", "Producer"],
    })
    lab = labels.cell_labels(df)
    assert lab.positive.sum() == 1  # the sand & gravel cell is not a target
    row = lab[lab.positive].iloc[0]
    np.testing.assert_allclose(row[C.CLASSES].sum(), 1.0, rtol=1e-6)
    assert row["Au"] > row["Ag"] > 0 and row["Cu"] > 0
    assert row["conf"] == 1.0


def test_split_sums_to_one():
    m = ProspectivityModel(C.N_CLASSES, d=16)
    b = torch.randn(5, 8, 64)
    mask = torch.ones(5, 8, dtype=torch.bool)
    s = m.split(b, mask)
    assert s.shape == (5, C.N_CLASSES + 1)
    torch.testing.assert_close(s.sum(-1), torch.ones(5))


def test_nnpu_nonnegative_branch():
    logit = torch.full((10,), 5.0, requires_grad=True)
    pos = torch.zeros(10, dtype=torch.bool)
    pos[0] = True
    loss, risk = nnpu_loss(logit, pos, torch.ones(10), prior=0.5)
    loss.backward()
    assert torch.isfinite(loss)


def test_end_to_end_recovers_planted_signal():
    d = synthetic.make(n_side=40, pos_rate=0.1, strength=5.0, seed=1)
    cb = d["cb"]
    groups = grid.spatial_blocks(cb.cells, block_res=4)
    folds = spatial_folds(groups, k=3)
    test = folds[0]
    train = np.setdiff1d(np.arange(len(cb.cells)), test)
    cfg = TrainConfig(d=32, epochs=60, batch=256, prior=0.1, seed=2)
    m = fit(cb.bags[train], cb.mask[train], d["positive"][train], d["conf"][train], d["composition"][train], cfg=cfg)
    split = predict(m, cb.bags[test], cb.mask[test])
    np.testing.assert_allclose(split.sum(1), 1.0, rtol=1e-5)
    score = 1 - split[:, 0]
    met = evaluate.capture_curve(score, d["truly_positive"][test])
    assert met["capture_auc"] > 0.75, met
    tp = d["truly_positive"][test]
    comp = split[tp, 1:] / split[tp, 1:].sum(1, keepdims=True)
    assert (comp.argmax(1) == d["cls"][test][tp]).mean() > 0.5  # 4 planted classes -> 0.25 by chance


def test_features_shapes():
    d = synthetic.make(n_side=6)
    cb = d["cb"]
    assert features.bag_stats(cb).shape == (len(cb.cells), 256)
    ctx = features.neighbourhood_context(cb.cells, cb.mean_embedding(), rings=(1, 2))
    assert ctx.shape == (len(cb.cells), 128)


def test_pretrain_runs_and_changes_weights():
    from earthprosp.model import CellEncoder
    from earthprosp.pretrain import pretrain

    d = synthetic.make(n_side=8)
    enc = CellEncoder(d=16)
    before = enc.adapter.proj.weight.detach().clone()
    b = torch.as_tensor(d["cb"].bags, dtype=torch.float32)
    m = torch.as_tensor(d["cb"].mask)
    pretrain(enc, b, m, epochs=1, batch=64, verbose=False)
    assert not torch.allclose(before, enc.adapter.proj.weight)
