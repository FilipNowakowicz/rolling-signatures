"""Narrow interface over signature-computation backends.

Everything else in rollsig calls only `signature()` and `n_features()`.
Swapping or adding a backend means touching this file alone.
"""

from __future__ import annotations

import functools
from typing import Literal

import numpy as np

from rollsig.algebra import (
    _words,
    n_features,
    tensor_exp,
    tensor_identity,
    tensor_log,
    tensor_multiply,
)

Backend = Literal["roughpy", "iisignature", "numpy"]


def _mobius(n: int) -> int:
    if n == 1:
        return 1
    result, remaining, p = 1, n, 2
    while p * p <= remaining:
        if remaining % p == 0:
            remaining //= p
            if remaining % p == 0:
                return 0
            result = -result
        p += 1
    if remaining > 1:
        result = -result
    return result


def n_log_features(dim: int, depth: int) -> int:
    """Dimension of the depth-`depth` free Lie algebra on `dim` generators.

    Witt's formula: level n contributes (1/n) * sum_{d | n} mobius(d) *
    dim**(n/d) independent Lie words. This is the size of `log_signature()`'s
    output -- much smaller than `n_features()` because the shuffle identities
    (docs/notes.html s1.7) make most tensor coordinates dependent.
    """
    return sum(
        sum(_mobius(d) * dim ** (level // d) for d in range(1, level + 1) if level % d == 0) // level
        for level in range(1, depth + 1)
    )


def _roughpy_signature(path: np.ndarray, depth: int):
    """Build the RoughPy context and grouplike signature object for `path`."""
    import roughpy as rp

    dim = path.shape[1]
    increments = np.diff(path, axis=0)
    ctx = rp.get_context(width=dim, depth=depth, coeffs=rp.DPReal)
    stream = rp.LieIncrementStream.from_increments(increments, ctx=ctx)
    sig = stream.signature(rp.RealInterval(0, len(increments)))
    return ctx, sig


def _signature_roughpy(path: np.ndarray, depth: int) -> np.ndarray:
    import roughpy as rp

    dim = path.shape[1]
    ctx, sig = _roughpy_signature(path, depth)
    # NB: TensorKey built with ctx= silently mis-indexes repeated-letter words
    # (e.g. (1,1)); width=/depth= indexes correctly and doesn't itself warn.
    return np.array(
        [
            sig[rp.TensorKey(list(word), width=dim, depth=depth)].to_float()
            for word in _words(dim, depth)
        ],
        dtype=float,
    )


def _log_labels_roughpy(dim: int, depth: int) -> list[str]:
    """Hall-basis bracket labels, in the order `log_signature()` returns them."""
    import roughpy as rp

    ctx = rp.get_context(width=dim, depth=depth, coeffs=rp.DPReal)
    basis = ctx.lie_basis
    return [str(basis.index_to_key(i)) for i in range(basis.dimension)]


def _log_signature_roughpy(path: np.ndarray, depth: int) -> np.ndarray:
    ctx, sig = _roughpy_signature(path, depth)
    logsig = ctx.to_logsignature(sig)
    basis = ctx.lie_basis
    # NB: LieBasis.key_to_index and Lie.__getitem__(key) both silently return
    # wrong values for this basis (see docs/notes.html s4.5, RoughPy 0.3.0) --
    # matching values up by the string form of each key, read off through
    # iteration, is the only way found to line them up reliably.
    by_label = {str(item.key()): item.value().to_float() for item in logsig}
    return np.array(
        [by_label.get(str(basis.index_to_key(i)), 0.0) for i in range(basis.dimension)],
        dtype=float,
    )


def _signature_iisignature(path: np.ndarray, depth: int) -> np.ndarray:
    import iisignature

    return np.asarray(iisignature.sig(path, depth), dtype=float)


@functools.lru_cache(maxsize=None)
def _iisignature_prepared(dim: int, depth: int):
    """Cached `iisignature.prepare` handle.

    `prepare` builds the Lyndon basis and the expansion tables for a given
    (dim, depth) and costs far more than a single log-signature; the ORVP
    benchmark calls `log_signature` hundreds of thousands of times at a
    fixed (dim, depth), so this must be memoised rather than rebuilt.
    """
    import iisignature

    return iisignature.prepare(dim, depth)


def _log_labels_iisignature(dim: int, depth: int) -> list[str]:
    """Lyndon-basis bracket labels, in `iisignature`'s own order."""
    import iisignature

    return list(iisignature.basis(_iisignature_prepared(dim, depth)))


def _log_signature_iisignature(path: np.ndarray, depth: int) -> np.ndarray:
    import iisignature

    dim = path.shape[1]
    return np.asarray(iisignature.logsig(path, _iisignature_prepared(dim, depth)), dtype=float)


def _signature_levels_numpy(path: np.ndarray, depth: int) -> list[np.ndarray]:
    """Truncated signature as levels 0..depth (level 0 is the scalar 1).

    Multiplies the truncated tensor exponentials of every path increment --
    Chen's identity as an algorithm (docs/notes.html s1.6, s3.3). The algebra
    itself lives in `rollsig.algebra`; this is the loop over increments that
    turns a path into an element of it.
    """
    dim = path.shape[1]
    levels = tensor_identity(dim, depth)
    for increment in np.diff(path, axis=0):
        levels = tensor_multiply(levels, tensor_exp(increment, depth), dim, depth)
    return levels


def _signature_numpy(path: np.ndarray, depth: int) -> np.ndarray:
    """Reference signature for a piecewise-linear path.

    This deliberately simple implementation is a correctness oracle for
    tests and small examples, not a production backend.
    """
    return np.concatenate(_signature_levels_numpy(path, depth)[1:])


def _log_signature_full_numpy(path: np.ndarray, depth: int) -> np.ndarray:
    """Full-tensor-coordinate log-signature -- a correctness oracle, not a backend.

    Computes log(S) = x - x^2/2 + x^3/3 - ... for x = S - 1 directly in the
    truncated tensor algebra; the series is exact (not merely truncated)
    because x has no level-0 part, so x^k vanishes below level k. The result
    is a genuine Lie element -- it lands in the free Lie algebra, which the
    primitivity property test in tests/test_backend.py checks directly -- but
    expressed in the same redundant `n_features(dim, depth)` tensor-word
    coordinates as `signature()`, not the minimal Hall/Lyndon basis
    `log_signature()` returns. The two are therefore NOT comparable
    elementwise; this function exists only to make the algebra executable
    for tests (docs/notes.html s4.4).
    """
    dim = path.shape[1]
    log_levels = tensor_log(_signature_levels_numpy(path, depth), dim, depth)
    return np.concatenate(log_levels[1:])


def _log_signature_expanded(path: np.ndarray, depth: int, backend: Backend) -> np.ndarray:
    """A backend's log-signature re-expressed in full tensor-word coordinates.

    A correctness oracle, not a feature builder. `log_signature()` returns
    coordinates in whichever basis of the free Lie algebra its backend
    prefers, and those bases genuinely differ (see that function's
    docstring). Mapping both back into the redundant but canonical
    `n_features(dim, depth)` word coordinates gives a basis-independent
    place to compare them -- to each other, and to
    `_log_signature_full_numpy`. Used only by tests.
    """
    dim = path.shape[1]
    if backend == "iisignature":
        import iisignature

        # iisignature computes the expanded form directly when asked for
        # method "x", so nothing here has to know the Lyndon convention.
        return np.asarray(iisignature.logsig(path, _iisignature_prepared(dim, depth), "x"), dtype=float)
    if backend == "roughpy":
        import roughpy as rp

        ctx, sig = _roughpy_signature(path, depth)
        tensor = ctx.lie_to_tensor(ctx.to_logsignature(sig))
        return np.array(
            [tensor[rp.TensorKey(list(word), width=dim, depth=depth)].to_float() for word in _words(dim, depth)],
            dtype=float,
        )
    raise ValueError(f"cannot expand log-signature for backend {backend!r}")


_BACKENDS = {
    "roughpy": _signature_roughpy,
    "iisignature": _signature_iisignature,
    "numpy": _signature_numpy,
}

_LOG_BACKENDS = {
    "roughpy": _log_signature_roughpy,
    "iisignature": _log_signature_iisignature,
}

_LOG_LABELS = {
    "roughpy": _log_labels_roughpy,
    "iisignature": _log_labels_iisignature,
}


def signature(path: np.ndarray, depth: int, backend: Backend = "roughpy") -> np.ndarray:
    """Truncated signature of a `(n_points, dim)` path.

    Returns levels 1..depth flattened in lexicographic word order (no
    level-0 constant term). A path with fewer than 2 points has no
    increments, so its signature is all zeros.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a two-dimensional array of shape (n_points, n_channels)")
    if depth < 1:
        raise ValueError("depth must be at least 1")
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {tuple(_BACKENDS)}")
    dim = path.shape[1]
    if path.shape[0] < 2:
        return np.zeros(n_features(dim, depth))
    return _BACKENDS[backend](path, depth)


def log_signature(path: np.ndarray, depth: int, backend: Backend = "roughpy") -> np.ndarray:
    """Truncated log-signature of a `(n_points, dim)` path.

    Returns coordinates of log(signature) in a basis of the free Lie algebra
    -- the minimal representation, `n_log_features(dim, depth)` numbers
    instead of `signature()`'s `n_features(dim, depth)`; see
    docs/notes.html ch.4.

    IMPORTANT -- unlike `signature()`, the two backends here are *not*
    numerically interchangeable. Both return genuine log-signatures, but in
    different bases: they agree through level 2 and diverge from level 3, in
    both the ordering and the sign convention of the nested brackets
    (docs/notes-orvp.html s6.2). Either is a valid feature set and the two
    span the same space, but a model fitted on one backend's coordinates
    must be scored on the same backend's coordinates. Use
    `_log_signature_expanded` to compare them on basis-independent ground;
    `get_feature_names_out` reports the backend's own bracket labels.

    `iisignature` is roughly two orders of magnitude faster than `roughpy`
    here and is what the ORVP benchmark uses; `roughpy` remains the default
    for consistency with `signature()`. `_log_signature_full_numpy` is a
    numpy correctness oracle only, not a backend -- see its docstring.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a two-dimensional array of shape (n_points, n_channels)")
    if depth < 1:
        raise ValueError("depth must be at least 1")
    if backend not in _LOG_BACKENDS:
        raise ValueError(f"unknown log-signature backend {backend!r}; expected one of {tuple(_LOG_BACKENDS)}")
    dim = path.shape[1]
    if path.shape[0] < 2:
        return np.zeros(n_log_features(dim, depth))
    return _LOG_BACKENDS[backend](path, depth)
