import numpy as np
from scipy import sparse

from histoomnist.data.size_factor import compute_size_factor, log_size_factor
from histoomnist.data.spot_table import load_spot_table


def test_size_factor_mean_one():
    counts = np.asarray([[1, 1], [2, 2], [4, 4]], dtype=np.float32)
    sf, valid = compute_size_factor(counts)
    assert valid.tolist() == [True, True, True]
    assert np.isclose(sf.mean(), 1.0)
    assert np.all(np.isfinite(log_size_factor(sf)))


def test_loaded_size_factor_keeps_total_count_valid_mask(tmp_path):
    features = np.ones((2, 3), dtype=np.float32)
    counts = sparse.csr_matrix(np.asarray([[0, 0], [2, 2]], dtype=np.float32))
    sf = np.asarray([1.0e-8, 1.0], dtype=np.float32)

    features_path = tmp_path / "features.npy"
    counts_path = tmp_path / "counts.npz"
    sf_path = tmp_path / "size_factor.npy"
    np.save(features_path, features)
    sparse.save_npz(counts_path, counts)
    np.save(sf_path, sf)

    table = load_spot_table(
        "toy",
        features_path=features_path,
        counts_path=counts_path,
        size_factor_path=sf_path,
        min_total_counts=1.0,
    )
    assert table.valid_mask.tolist() == [False, True]
