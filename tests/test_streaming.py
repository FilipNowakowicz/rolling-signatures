"""The streaming engine, checked against the thing it replaces.

Every claim v0.3 makes is a claim of *equality* with a from-scratch
recomputation, so almost everything here is a comparison against the v0.1
batch path -- which is itself pinned against RoughPy, `iisignature` and the
property-based oracles in `test_backend.py`.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from sigtrade import SignatureTransformer, signature
from sigtrade.algebra import n_features, tensor_inverse, tensor_multiply
from sigtrade.preprocessing import preprocess_path
from sigtrade.streaming import (
    StreamingSignature,
    _block_inverse,
    _block_signature,
    _PathEncoding,
    nested_suffix_signatures,
    rolling_signature,
    suffix_signature,
)

PREPROCESSING = [
    {},
    {"time_augmentation": True},
    {"lead_lag_transform": True},
    {"time_augmentation": True, "lead_lag_transform": True},
]


@pytest.fixture
def series():
    rng = np.random.default_rng(0)
    return rng.standard_normal((60, 2)).cumsum(axis=0) * 0.1


@pytest.mark.parametrize("options", PREPROCESSING)
def test_encoding_reproduces_the_preprocessed_increments(options):
    """The load-bearing assumption of the whole module.

    Streaming is only O(1) because preprocessing turns each raw increment into
    a fixed-size block of preprocessed increments, independent of the rest of
    the window. If `preprocess_path` ever stopped decomposing that way, every
    other test here would still pass on the shared code path -- so this checks
    the decomposition directly against the batch preprocessing.
    """
    rng = np.random.default_rng(1)
    window = rng.standard_normal((8, 2))
    encoding = _PathEncoding(2, options.get("time_augmentation", False), options.get("lead_lag_transform", False))
    dt = 1.0 / (len(window) - 1)

    expected = np.diff(preprocess_path(window, **options), axis=0)
    blocks = [vector for delta in np.diff(window, axis=0) for vector in encoding.block(delta, dt)]
    assert np.array(blocks) == pytest.approx(expected)
    assert encoding.channels == expected.shape[1]


@pytest.mark.parametrize("options", PREPROCESSING)
@pytest.mark.parametrize("window, depth", [(1, 2), (2, 2), (3, 1), (5, 3), (20, 2)])
def test_streaming_matches_the_batch_transformer(series, options, window, depth):
    batch = SignatureTransformer(
        window=window, depth=depth, backend="numpy", method="batch", **options
    ).fit_transform(series)
    streamed = rolling_signature(series, window=window, depth=depth, refresh_every=None, **options)
    assert streamed == pytest.approx(batch, abs=1e-10)


@given(series=arrays(dtype=float, shape=(25, 2), elements=st.floats(-2, 2, allow_nan=False)))
@settings(max_examples=30, deadline=None)
def test_streaming_matches_the_batch_transformer_on_random_series(series):
    batch = SignatureTransformer(
        window=7, depth=3, backend="numpy", method="batch", time_augmentation=True, lead_lag_transform=True
    ).fit_transform(series)
    streamed = rolling_signature(
        series, window=7, depth=3, time_augmentation=True, lead_lag_transform=True, refresh_every=None
    )
    assert streamed == pytest.approx(batch, abs=1e-9)


def test_streaming_is_causal(series):
    """Same guarantee as `test_transformer.py::test_causal_alignment`, and here
    it is structural: at tick t the engine has not been shown X[t+1:] at all."""
    engine = StreamingSignature(5, 2, dim=2, refresh_every=None)
    full = engine.extend(series)
    for t in (0, 1, 4, 5, 10, 59):
        engine.reset()
        assert engine.extend(series[: t + 1])[t] == pytest.approx(full[t]), f"lookahead leak at t={t}"


def test_block_inverse_matches_the_general_group_inverse():
    """The hot path uses the closed form exp(v)^-1 = exp(-v) rather than the
    `tensor_inverse` recursion. They must agree, or the speed came from being
    wrong."""
    encoding = _PathEncoding(2, time_augmentation=True, lead_lag_transform=True)
    block = encoding.block(np.array([0.3, -0.7]), dt=0.05)
    dim, depth = encoding.channels, 3
    closed_form = _block_inverse(block, dim, depth)
    general = tensor_inverse(_block_signature(block, dim, depth), dim, depth)
    for got, want in zip(closed_form, general):
        assert got == pytest.approx(want, abs=1e-12)


def test_reset_clears_the_stream(series):
    engine = StreamingSignature(5, 2, dim=2, refresh_every=None)
    first = engine.extend(series[:20])
    engine.reset()
    assert engine.n_updates_ == 0
    assert engine.signature == pytest.approx(np.zeros(n_features(2, 2)))
    assert engine.extend(series[:20]) == pytest.approx(first)


def test_auto_refresh_re_anchors_once_per_window(series):
    engine = StreamingSignature(10, 2, dim=2, refresh_every="auto")
    engine.extend(series)
    # 60 ticks, window 10: 9 recomputes filling the first window are skipped
    # here (no time augmentation, so the fill streams), leaving the re-anchors.
    assert engine.n_recomputes_ == pytest.approx(len(series) // 10, abs=1)


def test_refresh_does_not_change_the_answer(series):
    for refresh in (None, "auto", 3):
        streamed = rolling_signature(series, window=8, depth=3, refresh_every=refresh, time_augmentation=True)
        batch = SignatureTransformer(
            window=8, depth=3, backend="numpy", method="batch", time_augmentation=True
        ).fit_transform(series)
        assert streamed == pytest.approx(batch, abs=1e-10), f"refresh_every={refresh}"


def test_drift_stays_bounded_without_re_anchoring():
    """A regression guard on the number `benchmarks/streaming/` reports.

    Repeated group operations accumulate error a recomputation never incurs.
    Over 2,000 ticks at depth 3 the disagreement stays far below anything a
    downstream model could notice; if a change to the algebra made it worse by
    orders of magnitude, this catches it.
    """
    window, depth, ticks = 50, 3, 2000
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((ticks, 1)) * 1e-3).cumsum(axis=0)

    engine = StreamingSignature(window, depth, 1, time_augmentation=True, refresh_every=None)
    streamed = engine.extend(X)[-1]
    exact = signature(preprocess_path(X[-window:], time_augmentation=True), depth, backend="numpy")
    relative = np.max(np.abs(streamed - exact)) / np.max(np.abs(exact))
    assert relative < 1e-11

    anchored = StreamingSignature(window, depth, 1, time_augmentation=True, refresh_every="auto")
    relative_anchored = np.max(np.abs(anchored.extend(X)[-1] - exact)) / np.max(np.abs(exact))
    assert relative_anchored < relative


def test_suffix_signature_recovers_a_suffix(series):
    depth = 3
    whole = signature(series, depth, backend="numpy")
    prefix = signature(series[:21], depth, backend="numpy")
    expected = signature(series[20:], depth, backend="numpy")
    assert suffix_signature(whole, prefix, 2, depth) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("options", PREPROCESSING)
def test_nested_suffix_signatures_match_independent_computation(series, options):
    depth = 3
    got = nested_suffix_signatures(series, (60, 30, 15, 1), depth, **options)
    assert set(got) == {60, 30, 15, 1}
    for length, value in got.items():
        expected = signature(preprocess_path(series[-length:], **options), depth, backend="numpy")
        assert value == pytest.approx(expected, abs=1e-10), f"length={length}"


def test_nested_suffix_signatures_reject_an_over_long_window(series):
    with pytest.raises(ValueError, match="exceeds"):
        nested_suffix_signatures(series, (len(series) + 1,), 2)


def test_streaming_rejects_the_wrong_point_shape():
    engine = StreamingSignature(5, 2, dim=3)
    with pytest.raises(ValueError, match="3 channels"):
        engine.update([1.0, 2.0])


@pytest.mark.parametrize("kwargs", [{"window": 0}, {"depth": 0}, {"dim": 0}, {"refresh_every": -1}])
def test_streaming_validates_its_parameters(kwargs):
    with pytest.raises(ValueError):
        StreamingSignature(**{"window": 5, "depth": 2, "dim": 1, **kwargs})


def test_transformer_defaults_to_streaming_and_agrees_with_batch(series):
    auto = SignatureTransformer(window=9, depth=3, lead_lag_transform=True).fit(series)
    assert auto.method_ == "streaming"
    batch = SignatureTransformer(window=9, depth=3, lead_lag_transform=True, method="batch").fit(series)
    assert batch.method_ == "batch"
    assert auto.transform(series) == pytest.approx(batch.transform(series), abs=1e-9)


def test_transformer_falls_back_to_batch_when_streaming_cannot_apply(series):
    pytest.importorskip("roughpy")
    for options in ({"basepoint": True}, {"output": "log_signature"}):
        estimator = SignatureTransformer(window=6, depth=2, **options).fit(series)
        assert estimator.method_ == "batch"


@pytest.mark.parametrize("options", [{"basepoint": True}, {"output": "log_signature"}])
def test_transformer_rejects_explicit_streaming_it_cannot_honour(series, options):
    with pytest.raises(ValueError, match="does not support"):
        SignatureTransformer(window=6, depth=2, method="streaming", **options).fit(series)


def test_transformer_rejects_an_unknown_method(series):
    with pytest.raises(ValueError, match="method must be"):
        SignatureTransformer(method="fast").fit(series)


def test_rescale_applies_to_the_streaming_output(series):
    streamed = SignatureTransformer(window=8, depth=3, rescale=True, method="streaming").fit_transform(series)
    batch = SignatureTransformer(
        window=8, depth=3, rescale=True, method="batch", backend="numpy"
    ).fit_transform(series)
    assert streamed == pytest.approx(batch, abs=1e-9)


def test_chens_identity_composes_the_two_ends_of_a_window(series):
    """The identity the engine implements, stated once on its own terms:
    S(a.b) = S(a) * S(b), so a window is the product of its parts and the
    streaming update is that product rearranged."""
    depth, dim = 3, 2
    left, right = series[:30], series[29:]
    from sigtrade._backend import _signature_levels_numpy

    product = tensor_multiply(
        _signature_levels_numpy(left, depth), _signature_levels_numpy(right, depth), dim, depth
    )
    joined = signature(series, depth, backend="numpy")
    assert np.concatenate(product[1:]) == pytest.approx(joined, abs=1e-10)
