"""The tensor algebra's own correctness oracles.

`test_backend.py` checks that signatures are right. This checks that the ring
they live in behaves like a ring -- which is what `sigtrade.streaming` leans
on, so a silent failure here would show up as a plausible-looking rolling
feature rather than as an exception.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from sigtrade._backend import _log_signature_full_numpy, _signature_levels_numpy, signature
from sigtrade.algebra import (
    _words,
    dilate,
    from_levels,
    n_features,
    tensor_exp,
    tensor_identity,
    tensor_inverse,
    tensor_log,
    tensor_multiply,
    tensor_norm,
    to_levels,
)

DEPTH = 3
DIM = 2


@pytest.fixture
def path():
    rng = np.random.default_rng(0)
    return rng.standard_normal((10, DIM)).cumsum(axis=0)


def _levels(path, depth=DEPTH):
    return _signature_levels_numpy(np.asarray(path, dtype=float), depth)


def test_identity_is_the_unit(path):
    levels = _levels(path)
    unit = tensor_identity(DIM, DEPTH)
    for product in (
        tensor_multiply(levels, unit, DIM, DEPTH),
        tensor_multiply(unit, levels, DIM, DEPTH),
    ):
        for got, want in zip(product, levels):
            assert got == pytest.approx(want)


def test_levels_and_flat_round_trip(path):
    flat = signature(path, DEPTH, backend="numpy")
    assert from_levels(to_levels(flat, DIM, DEPTH)) == pytest.approx(flat)
    assert to_levels(flat, DIM, DEPTH)[0] == pytest.approx(np.ones(1))


def test_to_levels_rejects_the_wrong_length():
    with pytest.raises(ValueError):
        to_levels(np.zeros(n_features(DIM, DEPTH) + 1), DIM, DEPTH)


def test_exponential_is_the_signature_of_one_straight_segment():
    """exp(v) is precisely S of the segment from 0 to v -- the atom every other
    construction here is built from (docs/notes.html s3.3)."""
    v = np.array([0.7, -1.3])
    segment = np.vstack((np.zeros(DIM), v))
    assert from_levels(tensor_exp(v, DEPTH)) == pytest.approx(signature(segment, DEPTH, backend="numpy"))


@given(path=arrays(dtype=float, shape=(7, DIM), elements=st.floats(-3, 3, allow_nan=False)))
@settings(max_examples=50)
def test_inverse_is_two_sided(path):
    """Grouplike elements form a group, not merely a monoid with a left inverse."""
    levels = _levels(path)
    inverse = tensor_inverse(levels, DIM, DEPTH)
    unit = tensor_identity(DIM, DEPTH)
    for product in (
        tensor_multiply(levels, inverse, DIM, DEPTH),
        tensor_multiply(inverse, levels, DIM, DEPTH),
    ):
        for got, want in zip(product, unit):
            assert got == pytest.approx(want, abs=1e-9)


@given(path=arrays(dtype=float, shape=(7, DIM), elements=st.floats(-3, 3, allow_nan=False)))
@settings(max_examples=50)
def test_inverse_is_the_signature_of_the_reversed_path(path):
    """The group inverse has a path-level meaning: undo the path by walking it
    backwards. This is what makes `streaming._block_inverse`'s closed form
    (reverse the segments, negate them) legitimate rather than a coincidence."""
    inverse = tensor_inverse(_levels(path), DIM, DEPTH)
    reversed_signature = signature(path[::-1], DEPTH, backend="numpy")
    assert from_levels(inverse) == pytest.approx(reversed_signature, abs=1e-9)


def test_inverse_matches_the_antipode_formula(path):
    """The Hopf-algebra antipode, coordinate by coordinate: the coefficient of a
    word `w` in S^-1 is (-1)^|w| times the coefficient of `w` reversed in S."""
    inverse = dict(zip(_words(DIM, DEPTH), from_levels(tensor_inverse(_levels(path), DIM, DEPTH))))
    forward = dict(zip(_words(DIM, DEPTH), signature(path, DEPTH, backend="numpy")))
    for word, value in inverse.items():
        assert value == pytest.approx((-1) ** len(word) * forward[word[::-1]], abs=1e-9)


def test_inverse_rejects_a_non_unit_scalar():
    levels = tensor_identity(DIM, DEPTH)
    levels[0] = np.full(1, 2.0)
    with pytest.raises(ValueError):
        tensor_inverse(levels, DIM, DEPTH)


def test_log_matches_the_full_tensor_log_signature(path):
    """`tensor_log` is the same computation `_log_signature_full_numpy` performs
    on a signature, so the oracle in test_backend.py transitively covers it."""
    got = from_levels(tensor_log(_levels(path), DIM, DEPTH))
    assert got == pytest.approx(_log_signature_full_numpy(path, DEPTH), abs=1e-9)


@given(
    vector=arrays(dtype=float, shape=(DIM,), elements=st.floats(-2, 2, allow_nan=False)),
)
@settings(max_examples=50)
def test_log_and_exp_invert_each_other(vector):
    round_tripped = tensor_log(tensor_exp(vector, DEPTH), DIM, DEPTH)
    assert round_tripped[1] == pytest.approx(vector, abs=1e-9)
    # exp of a single vector is grouplike with a purely level-1 logarithm: the
    # higher levels of log(exp(v)) must vanish.
    for level in round_tripped[2:]:
        assert level == pytest.approx(np.zeros_like(level), abs=1e-9)


@given(
    path=arrays(dtype=float, shape=(6, DIM), elements=st.floats(-3, 3, allow_nan=False)),
    scales=arrays(dtype=float, shape=(DIM,), elements=st.floats(0.1, 5, allow_nan=False)),
)
@settings(max_examples=50)
def test_dilate_matches_scaling_the_path(path, scales):
    """Scaling channels of the path is a diagonal map on the signature; this is
    what lets `nested_suffix_signatures` fix up a renormalised time channel."""
    scaled = signature(path * scales, DEPTH, backend="numpy")
    assert from_levels(dilate(_levels(path), scales, DIM, DEPTH)) == pytest.approx(scaled, abs=1e-9)


def test_norm_ignores_the_scalar_part():
    levels = tensor_identity(DIM, DEPTH)
    assert tensor_norm(levels) == 0.0
    levels[2][0] = -4.0
    assert tensor_norm(levels) == 4.0
