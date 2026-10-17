import numpy as np

from earthprosp import aef_change


def test_change_is_zero_for_identical_years_and_positive_for_changed_cells(tmp_path):
    rng = np.random.default_rng(0)
    m = rng.normal(size=(5, 64)).astype(np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    cells = np.array([f"86000000{i}fffffff" for i in range(5)])
    m2 = m.copy()
    m2[0] = -m[0]
    np.savez(tmp_path / "a.npz", cells=cells, mean=m.astype(np.float16))
    np.savez(tmp_path / "b.npz", cells=cells[::-1], mean=m2[::-1].astype(np.float16))
    c = aef_change.change(str(tmp_path / "a.npz"), str(tmp_path / "b.npz"))["aef_change_cos"]
    assert abs(c[cells[0]] - 2) < 1e-2
    assert np.allclose(c[list(cells[1:])], 0, atol=1e-2)
