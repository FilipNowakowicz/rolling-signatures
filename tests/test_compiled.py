"""The compiled streaming kernel against its oracles.

`rollsig._compiled` is the same algorithm as the numpy engine written as loops
over flat arrays, so the numpy engine is the reference and has to agree with it
to floating-point noise. On top of that the kernel is held to the identities
that do not depend on the numpy code at all: Chen's identity, the shuffle
identity, the group laws, and agreement with every batch backend.
"""

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from rollsig import SignatureTransformer, signature
from rollsig import streaming as streaming_module
from rollsig.algebra import n_features, tensor_exp, tensor_inverse, tensor_multiply, to_levels, from_levels
from rollsig.preprocessing import preprocess_path
from rollsig.streaming import StreamingSignature, rolling_signature

pytest.importorskip("numba")
from rollsig import _compiled  # noqa: E402

PREPROCESSING = [
    {},
    {"time_augmentation": True},
    {"lead_lag_transform": True},
    {"time_augmentation": True, "lead_lag_transform": True},
]


@pytest.fixture
def series():
    rng = np.random.default_rng(1)
    return rng.standard_normal((90, 2)).cumsum(axis=0) * 0.1


def _relative(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


@pytest.mark.parametrize("options", PREPROCESSING)
@pytest.mark.parametrize("depth", [1, 2, 3, 4])
@pytest.mark.parametrize("refresh", ["auto", None, 5])
def test_compiled_matches_the_numpy_engine(series, options, depth, refresh):
    kwargs = dict(window=12, depth=depth, refresh_every=refresh, **options)
    compiled = rolling_signature(series, engine="numba", **kwargs)
    reference = rolling_signature(series, engine="python", **kwargs)
    assert _relative(compiled, reference) < 1e-12


@pytest.mark.parametrize("options", PREPROCESSING)
@pytest.mark.parametrize("backend", ["numpy", "iisignature", "roughpy"])
def test_compiled_matches_every_batch_backend(series, options, backend):
    pytest.importorskip(backend)
    streamed = rolling_signature(series, window=15, depth=3, engine="numba", **options)
    batch = SignatureTransformer(window=15, depth=3, backend=backend, method="batch", **options).fit_transform(series)
    assert _relative(streamed, batch) < 1e-11


@given(
    series=arrays(dtype=float, shape=(40, 2), elements=st.floats(-3, 3, allow_nan=False)),
    window=st.integers(2, 15),
    options=st.sampled_from(PREPROCESSING),
)
@settings(max_examples=40, deadline=None)
def test_compiled_matches_the_numpy_engine_on_random_series(series, window, options):
    compiled = rolling_signature(series, window=window, depth=3, engine="numba", **options)
    reference = rolling_signature(series, window=window, depth=3, engine="python", **options)
    assert compiled == pytest.approx(reference, rel=1e-9, abs=1e-9)


def test_update_and_extend_agree_and_count_alike(series):
    options = {"time_augmentation": True, "lead_lag_transform": True}
    one_by_one = StreamingSignature(10, 3, 2, engine="numba", **options)
    rows = np.array([one_by_one.update(point) for point in series])
    at_once = StreamingSignature(10, 3, 2, engine="numba", **options)
    assert at_once.extend(series) == pytest.approx(rows, abs=1e-14)
    reference = StreamingSignature(10, 3, 2, engine="python", **options)
    reference.extend(series)
    assert one_by_one.n_recomputes_ == at_once.n_recomputes_ == reference.n_recomputes_
    assert one_by_one.n_updates_ == at_once.n_updates_ == reference.n_updates_ == len(series)


def test_chens_identity_on_the_compiled_output(series):
    """S(whole) = S(first part) * S(second part), all three from the kernel."""
    dim, depth, split = 2, 3, 30
    n_windows = len(series)
    whole = rolling_signature(series[:n_windows], window=n_windows, depth=depth, engine="numba")[-1]
    first = rolling_signature(series[:split], window=n_windows, depth=depth, engine="numba")[-1]
    second = rolling_signature(series[split - 1 :], window=n_windows, depth=depth, engine="numba")[-1]
    product = tensor_multiply(to_levels(first, dim, depth), to_levels(second, dim, depth), dim, depth)
    assert from_levels(product) == pytest.approx(whole, abs=1e-12)


def _word_index(word, channels):
    offset = n_features(channels, len(word) - 1) if len(word) > 1 else 0
    index = 0
    for letter in word:
        index = index * channels + (letter - 1)
    return offset + index


def _shuffles(u, v):
    for positions in itertools.combinations(range(len(u) + len(v)), len(u)):
        chosen = set(positions)
        u_iter, v_iter = iter(u), iter(v)
        yield tuple(next(u_iter) if k in chosen else next(v_iter) for k in range(len(u) + len(v)))


@pytest.mark.parametrize("u,v", [((1,), (2,)), ((1,), (2, 3)), ((4,), (4, 1)), ((2,), (3, 3))])
def test_shuffle_identity_on_the_compiled_output(series, u, v):
    """S^u S^v = sum over shuffles w of u and v of S^w, at a window the kernel has slid."""
    options = {"time_augmentation": True, "lead_lag_transform": True}
    sig = rolling_signature(series[:, :1], window=20, depth=3, engine="numba", **options)[-1]
    assert sig[_word_index(u, 4)] * sig[_word_index(v, 4)] == pytest.approx(
        sum(sig[_word_index(w, 4)] for w in _shuffles(u, v)), abs=1e-12
    )


def test_kernel_operations_match_the_algebra_and_obey_the_group_laws():
    rng = np.random.default_rng(2)
    channels, depth = 3, 4
    offs = _compiled.level_offsets(channels, depth)
    work = np.zeros((2, int(offs[-1])))
    base = np.concatenate([[1.0], rng.standard_normal(int(offs[-1]) - 1) * 0.3])
    v = rng.standard_normal(channels)
    levels = to_levels(base[1:], channels, depth)
    exp = tensor_exp(v, depth)

    right = base.copy()
    _compiled._right(right, v, 1.0, work, offs, depth, channels)
    assert right[1:] == pytest.approx(from_levels(tensor_multiply(levels, exp, channels, depth)), abs=1e-13)

    left = base.copy()
    _compiled._left(left, v, 1.0, work, offs, depth, channels)
    assert left[1:] == pytest.approx(from_levels(tensor_multiply(exp, levels, channels, depth)), abs=1e-13)

    # exp(v) exp(-v) = 1: multiplying and then dividing returns the start, on either side.
    _compiled._right(right, v, -1.0, work, offs, depth, channels)
    _compiled._left(left, v, -1.0, work, offs, depth, channels)
    assert right == pytest.approx(base, abs=1e-12)
    assert left == pytest.approx(base, abs=1e-12)
    # and exp(-v) is the general group inverse of exp(v).
    inverse = tensor_inverse(exp, channels, depth)
    assert from_levels(inverse) == pytest.approx(from_levels(tensor_exp(-v, depth)), abs=1e-13)


def test_reset_restarts_the_compiled_stream(series):
    engine = StreamingSignature(8, 2, 2, engine="numba", time_augmentation=True)
    first = engine.extend(series)
    engine.reset()
    assert engine.n_updates_ == 0 and engine.n_recomputes_ == 0
    assert engine.extend(series) == pytest.approx(first, abs=0)


def test_a_one_point_window_works_with_the_compiled_engine(series):
    out = rolling_signature(series, window=1, depth=2, engine="numba")
    assert out == pytest.approx(np.zeros_like(out))


def test_compiled_rejects_the_wrong_point_shape():
    with pytest.raises(ValueError, match="expected a point"):
        StreamingSignature(5, 2, 2, engine="numba").update([1.0, 2.0, 3.0])


def test_drift_with_re_anchoring_stays_at_the_reported_level():
    """The README's figure: re-anchored drift of order 1e-14, at every depth."""
    window, ticks, checkpoint = 250, 6000, 250
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((ticks, 1)) * 1e-3).cumsum(axis=0)
    options = {"time_augmentation": True, "lead_lag_transform": True}
    for depth in (2, 3, 4):
        engine = StreamingSignature(window, depth, 1, engine="numba", **options)
        worst = 0.0
        for tick, point in enumerate(X):
            streamed = engine.update(point)
            if tick >= window and tick % checkpoint == 0:
                exact = signature(preprocess_path(X[tick + 1 - window : tick + 1], **options), depth, backend="numpy")
                worst = max(worst, _relative(streamed, exact))
        assert worst < 1e-13, f"depth {depth}: {worst:.2e}"


def test_unanchored_compiled_drift_is_no_worse_than_the_numpy_engines():
    window, depth, ticks = 50, 3, 2000
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((ticks, 1)) * 1e-3).cumsum(axis=0)
    exact = signature(preprocess_path(X[-window:], time_augmentation=True), depth, backend="numpy")
    errors = {}
    for engine in ("python", "numba"):
        streamed = rolling_signature(X, window, depth, time_augmentation=True, refresh_every=None, engine=engine)[-1]
        errors[engine] = _relative(streamed, exact)
    assert errors["numba"] < 1e-11
    assert errors["numba"] < 10 * errors["python"] + 1e-15


def test_engine_selection_and_fallback(monkeypatch):
    with pytest.raises(ValueError, match="engine"):
        StreamingSignature(5, 2, 1, engine="cython")
    assert StreamingSignature(5, 2, 1, engine="python")._kernel is None
    assert StreamingSignature(5, 2, 1, engine="auto")._kernel is _compiled
    # Without Numba: "auto" quietly uses numpy, an explicit request says why it cannot.
    monkeypatch.setattr(streaming_module, "_load_compiled", lambda: None)
    fallback = StreamingSignature(5, 2, 1)
    assert fallback._kernel is None
    fallback.extend(np.arange(8.0)[:, None])
    with pytest.raises(ImportError, match="Numba"):
        StreamingSignature(5, 2, 1, engine="numba")
