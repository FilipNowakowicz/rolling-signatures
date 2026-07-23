import numpy as np
import pytest

from sigtrade import SignatureTransformer
from sigtrade._backend import n_features, signature


@pytest.fixture
def series():
    rng = np.random.default_rng(0)
    return rng.standard_normal((30, 2)).cumsum(axis=0)


def test_output_shape(series):
    tr = SignatureTransformer(window=5, depth=2)
    out = tr.fit_transform(series)
    assert out.shape == (30, n_features(2, 2))


def test_matches_direct_signature_call(series):
    window, depth = 5, 2
    tr = SignatureTransformer(window=window, depth=depth)
    out = tr.fit_transform(series)

    t = 17
    expected = signature(series[t + 1 - window : t + 1], depth=depth)
    assert out[t] == pytest.approx(expected)


def test_causal_alignment(series):
    """The row at time t must depend only on data up to and including t --
    computing it from a truncated prefix must give the same answer."""
    window, depth = 5, 2
    tr = SignatureTransformer(window=window, depth=depth)
    full_out = tr.fit_transform(series)

    for t in (0, 1, 4, 5, 10, 29):
        prefix_out = tr.fit_transform(series[: t + 1])
        assert prefix_out[t] == pytest.approx(full_out[t]), f"lookahead leak at t={t}"


def test_get_feature_names_out(series):
    tr = SignatureTransformer(window=5, depth=2).fit(series)
    names = tr.get_feature_names_out()
    assert list(names) == ["sig_1", "sig_2", "sig_11", "sig_12", "sig_21", "sig_22"]
