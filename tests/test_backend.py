import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from rollsig._backend import (
    _log_signature_expanded,
    _log_signature_full_numpy,
    _words,
    log_signature,
    n_features,
    n_log_features,
    signature,
)


@pytest.fixture
def path():
    rng = np.random.default_rng(0)
    return rng.standard_normal((10, 2)).cumsum(axis=0)


def test_level_one_is_net_displacement(path):
    sig = signature(path, depth=1, backend="numpy")
    assert sig == pytest.approx(path[-1] - path[0])


def test_diagonal_level_two_is_half_squared_displacement(path):
    sig = signature(path, depth=2, backend="numpy")
    s1 = sig[:2]
    s11, s22 = sig[2], sig[5]
    assert s11 == pytest.approx(0.5 * s1[0] ** 2)
    assert s22 == pytest.approx(0.5 * s1[1] ** 2)


def test_shuffle_identity(path):
    sig = signature(path, depth=2, backend="numpy")
    s1, s2 = sig[0], sig[1]
    s11, s12, s21, s22 = sig[2], sig[3], sig[4], sig[5]
    assert s1 * s2 == pytest.approx(s12 + s21)
    assert s1 * s1 == pytest.approx(2 * s11)
    assert s2 * s2 == pytest.approx(2 * s22)


def test_short_path_is_zero():
    sig = signature(np.zeros((1, 2)), depth=3, backend="numpy")
    assert sig == pytest.approx(np.zeros(n_features(2, 3)))


def test_backends_agree(path):
    pytest.importorskip("iisignature")
    sig_rp = signature(path, depth=3, backend="roughpy")
    sig_ii = signature(path, depth=3, backend="iisignature")
    assert sig_rp == pytest.approx(sig_ii, abs=1e-8)


def test_numpy_reference_agrees_with_roughpy(path):
    pytest.importorskip("roughpy")
    sig_rp = signature(path, depth=3, backend="roughpy")
    sig_np = signature(path, depth=3, backend="numpy")
    assert sig_rp == pytest.approx(sig_np, abs=1e-8)


def test_chens_identity_for_numpy_reference(path):
    left, right = path[:6], path[5:]
    depth, dim = 3, path.shape[1]
    left_sig = signature(left, depth=depth, backend="numpy")
    right_sig = signature(right, depth=depth, backend="numpy")
    joined_sig = signature(np.vstack((left, right[1:])), depth=depth, backend="numpy")

    left_levels = [np.ones(1)]
    right_levels = [np.ones(1)]
    offset = 0
    for level in range(1, depth + 1):
        size = dim**level
        left_levels.append(left_sig[offset : offset + size])
        right_levels.append(right_sig[offset : offset + size])
        offset += size
    expected = np.concatenate(
        [
            sum(
                (np.kron(left_levels[i], right_levels[level - i]) for i in range(level + 1)),
                start=np.zeros(dim**level),
            )
            for level in range(1, depth + 1)
        ]
    )
    assert joined_sig == pytest.approx(expected)


def _resample_collinear(path: np.ndarray, insertions: dict[int, float]) -> np.ndarray:
    """Insert extra points along existing segments of `path`.

    Each inserted point is a convex combination of the segment's endpoints,
    so it lies exactly on the line already being traversed -- this changes
    only how densely the trajectory is sampled, not the trajectory itself.
    """
    points = [path[0]]
    for i in range(len(path) - 1):
        if i in insertions:
            points.append(path[i] + insertions[i] * (path[i + 1] - path[i]))
        points.append(path[i + 1])
    return np.array(points)


@given(
    path=arrays(dtype=float, shape=(5, 2), elements=st.floats(-5, 5, allow_nan=False)),
    insertions=st.dictionaries(
        keys=st.integers(0, 3), values=st.floats(0.05, 0.95, allow_nan=False), max_size=3
    ),
)
@settings(max_examples=50)
def test_signature_invariant_under_time_reparametrization(path, insertions):
    """Resampling the same trajectory more densely doesn't change its signature
    (docs/notes.html s5.1): the signature depends on the path as a geometric
    trace, not on how many points were used to record it. Exact for
    piecewise-linear paths, since two collinear increments commute in the
    tensor algebra -- exp(a) * exp(b) = exp(a + b) when a and b are scalar
    multiples of the same vector -- so subdividing a straight segment leaves
    every level of the signature unchanged rather than merely close."""
    depth = 3
    dense = _resample_collinear(path, insertions)
    sig_sparse = signature(path, depth=depth, backend="numpy")
    sig_dense = signature(dense, depth=depth, backend="numpy")
    assert sig_dense == pytest.approx(sig_sparse, abs=1e-8)


@pytest.mark.parametrize(
    "dim, depth, expected",
    [(2, 1, 2), (2, 2, 3), (2, 3, 5), (2, 4, 8), (3, 2, 6), (3, 3, 14)],
)
def test_n_log_features_matches_witt_formula(dim, depth, expected):
    assert n_log_features(dim, depth) == expected


def test_log_signature_level_one_matches_signature(path):
    """log(1+x) = x - x^2/2 + ... has the same degree-1 term as x itself,
    so the first `dim` log-signature coordinates equal S1 (see s1.3)."""
    pytest.importorskip("roughpy")
    dim = path.shape[1]
    sig = signature(path, depth=3, backend="numpy")
    logsig = log_signature(path, depth=3, backend="roughpy")
    assert logsig[:dim] == pytest.approx(sig[:dim])


def test_log_signature_short_path_is_zero():
    pytest.importorskip("roughpy")
    logsig = log_signature(np.zeros((1, 2)), depth=3, backend="roughpy")
    assert logsig == pytest.approx(np.zeros(n_log_features(2, 3)))


def test_log_signature_rejects_unknown_backend(path):
    with pytest.raises(ValueError):
        log_signature(path, depth=2, backend="numpy")


@pytest.mark.parametrize("backend", ["roughpy", "iisignature"])
def test_log_signature_backends_expand_to_the_numpy_oracle(path, backend):
    """Both backends really do return log-signatures.

    Each one's coordinates live in its own basis of the free Lie algebra, so
    they can't be compared elementwise. Expanding both back into canonical
    tensor-word coordinates gives common ground -- and there they must agree
    with the numpy oracle, which computes log(S) as a power series with no
    basis involved at all.
    """
    pytest.importorskip(backend)
    expanded = _log_signature_expanded(path, depth=3, backend=backend)
    assert expanded == pytest.approx(_log_signature_full_numpy(path, depth=3), abs=1e-8)


def test_log_signature_backends_disagree_from_level_three(path):
    """A regression test on a known, documented incompatibility.

    roughpy and iisignature agree through level 2 but order and sign their
    level-3 brackets differently (docs/notes-orvp.html s6.2). This is not a
    bug to be fixed; it's a constraint the benchmark has to respect -- fit
    and predict must use the same backend. If a future version of either
    library silently changed convention, this test would catch it.
    """
    pytest.importorskip("iisignature")
    dim, depth = path.shape[1], 3
    rp = log_signature(path, depth=depth, backend="roughpy")
    ii = log_signature(path, depth=depth, backend="iisignature")
    through_level_two = dim + n_log_features(dim, 2) - dim
    assert rp[:through_level_two] == pytest.approx(ii[:through_level_two], abs=1e-8)
    assert rp[through_level_two:] != pytest.approx(ii[through_level_two:], abs=1e-8)


def test_log_signature_iisignature_labels_match_feature_count(path):
    from rollsig._backend import _log_labels_iisignature

    pytest.importorskip("iisignature")
    labels = _log_labels_iisignature(path.shape[1], 3)
    assert len(labels) == n_log_features(path.shape[1], 3)
    assert labels[: path.shape[1]] == [str(i) for i in range(1, path.shape[1] + 1)]


def test_log_signature_matches_bracket_expansion_of_full_tensor_log(path):
    """Cross-check the roughpy Hall-basis coefficients against the numpy
    oracle at the three depth-<=3 words worked out by hand in s4.5: the
    bracket [1,2]'s coefficient is the raw word(1,2) coefficient, and
    [1,[1,2]] / [2,[1,2]] are word(1,1,2) and word(2,1,2)/2 respectively."""
    pytest.importorskip("roughpy")
    depth = 3
    logsig = log_signature(path, depth=depth, backend="roughpy")
    log_full = dict(zip(_words(2, depth), _log_signature_full_numpy(path, depth)))
    assert logsig[2] == pytest.approx(log_full[(1, 2)], abs=1e-8)
    assert logsig[3] == pytest.approx(log_full[(1, 1, 2)], abs=1e-8)
    assert logsig[4] == pytest.approx(log_full[(2, 1, 2)] / 2, abs=1e-8)


def _shuffles(u, v):
    """All interleavings of u and v that preserve each one's internal order."""
    for u_positions in itertools.combinations(range(len(u) + len(v)), len(u)):
        u_positions = set(u_positions)
        u_iter, v_iter, word = iter(u), iter(v), []
        for k in range(len(u) + len(v)):
            word.append(next(u_iter) if k in u_positions else next(v_iter))
        yield tuple(word)


@given(
    path=arrays(dtype=float, shape=(6, 2), elements=st.floats(-5, 5, allow_nan=False)),
    u=st.tuples(st.integers(1, 2)),
    v=st.tuples(st.integers(1, 2), st.integers(1, 2)),
)
@settings(max_examples=50)
def test_log_signature_is_primitive_under_shuffle(path, u, v):
    """The defining property of a Lie element (docs/notes.html s4.2): unlike
    a grouplike signature (test_shuffle_identity, S^u S^v = sum_shuffle S^w),
    a *primitive* element sums to zero over every nontrivial shuffle."""
    log_full = dict(zip(_words(2, 3), _log_signature_full_numpy(path, depth=3)))
    total = sum(log_full[w] for w in _shuffles(u, v))
    assert total == pytest.approx(0.0, abs=1e-6)
