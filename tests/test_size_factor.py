import numpy as np

from histoomnist.data.size_factor import compute_size_factor, log_size_factor


def test_size_factor_mean_one():
    counts = np.asarray([[1, 1], [2, 2], [4, 4]], dtype=np.float32)
    sf, valid = compute_size_factor(counts)
    assert valid.tolist() == [True, True, True]
    assert np.isclose(sf.mean(), 1.0)
    assert np.all(np.isfinite(log_size_factor(sf)))
