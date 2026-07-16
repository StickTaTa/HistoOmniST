import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from histoomnist.inference.count_scale import (
    add_count_scale_columns,
    add_count_scale_programs,
    mean_one_sf_from_log,
    reconstruct_count_scale,
)


def test_mean_one_sf_from_log_is_stable_and_centered() -> None:
    normalized_log_sf, sf = mean_one_sf_from_log(np.array([1000.0, 1001.0, 999.0]))

    assert np.isfinite(normalized_log_sf).all()
    assert np.isfinite(sf).all()
    assert np.isclose(sf.mean(), 1.0, atol=1.0e-6)
    assert np.allclose(np.exp(normalized_log_sf), sf, atol=1.0e-6)


def test_reconstruct_count_scale_multiplies_back_transformed_rate() -> None:
    rate_log1p = np.log1p(np.array([[2.0, 4.0], [3.0, 5.0]], dtype=np.float32))
    sf = np.array([0.5, 1.5], dtype=np.float32)

    count, count_log1p = reconstruct_count_scale(rate_log1p, sf)

    expected = np.array([[1.0, 2.0], [4.5, 7.5]], dtype=np.float32)
    assert np.allclose(count, expected)
    assert np.allclose(count_log1p, np.log1p(expected))


def test_reconstruct_count_scale_projects_negative_rate_predictions_to_zero() -> None:
    rate_log1p = np.array([[-0.01, np.log1p(2.0)]], dtype=np.float32)
    sf = np.array([1.5], dtype=np.float32)

    count, count_log1p = reconstruct_count_scale(rate_log1p, sf)

    expected = np.array([[0.0, 3.0]], dtype=np.float32)
    assert np.allclose(count, expected)
    assert np.allclose(count_log1p, np.log1p(expected))


def test_programs_use_log1p_reconstructed_counts_not_rate_columns() -> None:
    frame = pd.DataFrame(
        {
            "pred_sf": [0.5, 1.5],
            "gene_A": np.log1p([2.0, 3.0]),
            "gene_B": np.log1p([4.0, 5.0]),
        }
    )
    frame = add_count_scale_columns(
        frame,
        ["A", "B"],
        source_rate_prefix="gene_",
    )
    frame, used = add_count_scale_programs(
        frame,
        ["A", "B"],
        {"pair": ["A", "B"]},
    )

    expected = frame[["count_log1p_A", "count_log1p_B"]].mean(axis=1)
    assert used == {"pair": ["A", "B"]}
    assert np.allclose(frame["program_pair"], expected)
    assert not np.allclose(frame["program_pair"], frame[["gene_A", "gene_B"]].mean(axis=1))


def test_release_panel_column_generation_does_not_fragment_the_frame() -> None:
    genes = [f"G{index}" for index in range(28)]
    frame = pd.DataFrame(
        {
            "pred_sf": [0.5, 1.5],
            **{f"gene_{gene}": np.log1p([2.0, 3.0]) for gene in genes},
        }
    )
    for index in range(30):
        frame[f"metadata_{index}"] = index

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PerformanceWarning)
        frame = add_count_scale_columns(frame, genes, source_rate_prefix="gene_")
        frame, used = add_count_scale_programs(frame, genes, {"all": genes})

    assert used == {"all": genes}
    assert not any(isinstance(item.message, PerformanceWarning) for item in caught)
