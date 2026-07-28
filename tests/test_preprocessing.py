import numpy as np
import pytest

from sigtrade import SignatureTransformer, add_basepoint, lead_lag, time_augment
from sigtrade._backend import n_features, signature


def test_time_augmentation_appends_a_local_clock():
    path = np.array([[2.0], [3.0], [5.0]])
    augmented = time_augment(path)
    np.testing.assert_allclose(augmented, [[2.0, 0.0], [3.0, 0.5], [5.0, 1.0]])


def test_basepoint_prepends_origin():
    path = np.array([[2.0, 3.0], [5.0, 7.0]])
    np.testing.assert_allclose(add_basepoint(path), [[0.0, 0.0], [2.0, 3.0], [5.0, 7.0]])


def test_lead_lag_embedding_has_expected_order():
    path = np.array([[1.0], [4.0], [6.0]])
    embedded = lead_lag(path)
    np.testing.assert_allclose(embedded, [[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [6.0, 4.0], [6.0, 6.0]])


def test_transformer_preprocessing_matches_direct_signature():
    series = np.array([[1.0], [3.0], [2.0], [5.0]])
    tr = SignatureTransformer(
        window=3,
        depth=2,
        backend="numpy",
        basepoint=True,
        time_augmentation=True,
        lead_lag_transform=True,
    )
    out = tr.fit_transform(series)
    direct_path = lead_lag(time_augment(add_basepoint(series[-3:])))
    assert out[-1] == pytest.approx(signature(direct_path, depth=2, backend="numpy"))
    assert out.shape[1] == n_features(4, 2)


def test_factorial_rescaling_applies_per_signature_level():
    series = np.array([[0.0], [1.0], [3.0]])
    plain = SignatureTransformer(window=3, depth=3, backend="numpy").fit_transform(series)
    scaled = SignatureTransformer(window=3, depth=3, backend="numpy", rescale=True).fit_transform(series)
    assert scaled[-1] == pytest.approx(plain[-1] * np.array([1.0, 2.0, 6.0]))


@pytest.mark.parametrize("kwargs", [{"window": 0}, {"window": 1.5}, {"depth": 0}, {"depth": True}])
def test_transformer_rejects_invalid_window_and_depth(kwargs):
    with pytest.raises(ValueError):
        SignatureTransformer(backend="numpy", **kwargs).fit(np.ones((3, 1)))
