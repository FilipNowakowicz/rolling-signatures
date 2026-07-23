"""Narrow interface over signature-computation backends.

Everything else in sigtrade calls only `signature()` and `n_features()`.
Swapping or adding a backend means touching this file alone.
"""

from __future__ import annotations

import itertools
from typing import Literal

import numpy as np

Backend = Literal["roughpy", "iisignature"]


def _words(dim: int, depth: int):
    for level in range(1, depth + 1):
        yield from itertools.product(range(1, dim + 1), repeat=level)


def n_features(dim: int, depth: int) -> int:
    """Length of a depth-`depth` signature over a `dim`-dimensional path."""
    return sum(dim**level for level in range(1, depth + 1))


def _signature_roughpy(path: np.ndarray, depth: int) -> np.ndarray:
    import roughpy as rp

    dim = path.shape[1]
    increments = np.diff(path, axis=0)
    ctx = rp.get_context(width=dim, depth=depth, coeffs=rp.DPReal)
    stream = rp.LieIncrementStream.from_increments(increments, ctx=ctx)
    sig = stream.signature(rp.RealInterval(0, len(increments)))
    # NB: TensorKey built with ctx= silently mis-indexes repeated-letter words
    # (e.g. (1,1)); width=/depth= indexes correctly and doesn't itself warn.
    return np.array(
        [
            sig[rp.TensorKey(list(word), width=dim, depth=depth)].to_float()
            for word in _words(dim, depth)
        ],
        dtype=float,
    )


def _signature_iisignature(path: np.ndarray, depth: int) -> np.ndarray:
    import iisignature

    return np.asarray(iisignature.sig(path, depth), dtype=float)


_BACKENDS = {
    "roughpy": _signature_roughpy,
    "iisignature": _signature_iisignature,
}


def signature(path: np.ndarray, depth: int, backend: Backend = "roughpy") -> np.ndarray:
    """Truncated signature of a `(n_points, dim)` path.

    Returns levels 1..depth flattened in lexicographic word order (no
    level-0 constant term). A path with fewer than 2 points has no
    increments, so its signature is all zeros.
    """
    path = np.asarray(path, dtype=float)
    dim = path.shape[1]
    if path.shape[0] < 2:
        return np.zeros(n_features(dim, depth))
    return _BACKENDS[backend](path, depth)
