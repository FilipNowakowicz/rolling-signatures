import numpy as np
import pytest

from rollsig import SignatureTransformer
from rollsig._backend import n_features, n_log_features, signature


@pytest.fixture
def series():
    rng = np.random.default_rng(0)
    return rng.standard_normal((30, 2)).cumsum(axis=0)


def test_output_shape(series):
    tr = SignatureTransformer(window=5, depth=2, backend="numpy")
    out = tr.fit_transform(series)
    assert out.shape == (30, n_features(2, 2))


def test_matches_direct_signature_call(series):
    window, depth = 5, 2
    tr = SignatureTransformer(window=window, depth=depth, backend="numpy")
    out = tr.fit_transform(series)

    t = 17
    expected = signature(series[t + 1 - window : t + 1], depth=depth, backend="numpy")
    assert out[t] == pytest.approx(expected)


def test_causal_alignment(series):
    """The row at time t must depend only on data up to and including t --
    computing it from a truncated prefix must give the same answer."""
    window, depth = 5, 2
    tr = SignatureTransformer(window=window, depth=depth, backend="numpy")
    full_out = tr.fit_transform(series)

    for t in (0, 1, 4, 5, 10, 29):
        prefix_out = tr.fit_transform(series[: t + 1])
        assert prefix_out[t] == pytest.approx(full_out[t]), f"lookahead leak at t={t}"


def test_get_feature_names_out(series):
    tr = SignatureTransformer(window=5, depth=2, backend="numpy").fit(series)
    names = tr.get_feature_names_out()
    assert list(names) == ["sig_1", "sig_2", "sig_11", "sig_12", "sig_21", "sig_22"]


def test_log_signature_output_shape_and_names(series):
    pytest.importorskip("roughpy")
    tr = SignatureTransformer(window=5, depth=3, output="log_signature").fit(series)
    out = tr.transform(series)
    assert out.shape == (30, n_log_features(2, 3))
    assert list(tr.get_feature_names_out()) == [
        "logsig_1",
        "logsig_2",
        "logsig_[1,2]",
        "logsig_[1,[1,2]]",
        "logsig_[2,[1,2]]",
    ]


def test_log_signature_causal_alignment(series):
    """Same lookahead guarantee as the plain signature (test_causal_alignment),
    checked for the log-signature output path too."""
    pytest.importorskip("roughpy")
    window, depth = 5, 2
    tr = SignatureTransformer(window=window, depth=depth, output="log_signature")
    full_out = tr.fit_transform(series)

    for t in (0, 1, 4, 5, 10, 29):
        prefix_out = tr.fit_transform(series[: t + 1])
        assert prefix_out[t] == pytest.approx(full_out[t]), f"lookahead leak at t={t}"


def test_rescale_rejected_with_log_signature_output(series):
    with pytest.raises(ValueError):
        SignatureTransformer(backend="numpy", output="log_signature", rescale=True).fit(series)
