import numpy as np

from histoomnist.features.patch_features import RGB_FEATURE_NAMES, hipt256_feature_names, rgb_stats_features


def test_rgb_stats_feature_shape_and_tissue_fraction():
    images = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    images[0, ...] = 255
    images[1, ..., 0] = 120
    images[1, ..., 1] = 40
    images[1, ..., 2] = 80

    features = rgb_stats_features(images)

    assert features.shape == (2, len(RGB_FEATURE_NAMES))
    assert np.all(np.isfinite(features))
    assert features[0, RGB_FEATURE_NAMES.index("tissue_fraction")] == 0.0
    assert features[1, RGB_FEATURE_NAMES.index("tissue_fraction")] == 1.0


def test_hipt256_feature_names_are_stable():
    assert hipt256_feature_names(3) == ["hipt256_cls_000", "hipt256_cls_001", "hipt256_cls_002"]

