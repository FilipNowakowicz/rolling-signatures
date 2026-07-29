"""The truncated tensor algebra, as an executable object.

`_backend.py` computes signatures; this module is the *ring they live in*.
A signature is a grouplike element of the tensor algebra
T((V)) = sum_n V^{tensor n} truncated at `depth`, and the two facts that make
`rollsig.streaming` possible are both stated here:

* **Chen's identity** (docs/notes.html s1.6) -- concatenating paths multiplies
  their signatures, so `tensor_multiply` extends a window at its right end.
* **The group inverse** -- every element with scalar part 1 is invertible, so
  `tensor_inverse` *removes* a segment from the left end. There is no series
  to truncate and no approximation: in the truncated algebra the inverse is a
  finite recursion, exact in exact arithmetic.

Elements are represented as **level lists**: `levels[k]` is a flat array of
length `dim**k` holding the level-k tensor in lexicographic word order, and
`levels[0]` is the scalar part. `signature()` returns levels 1..depth
concatenated with the scalar dropped (it is always 1 for a signature);
`to_levels` and `from_levels` convert between the two conventions.

Everything here is pure numpy on small arrays. That is deliberate -- the
signature *backends* are the place for compiled code, but a sliding-window
update touches only `n_features(dim, depth)` numbers regardless of how long
the window is, and the interesting cost is the algebra, not the path.
"""

from __future__ import annotations

import itertools

import numpy as np

Levels = list[np.ndarray]


def _tensor_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Tensor product of two flattened levels.

    Equal to `np.kron(a, b)` for one-dimensional inputs, which is what every
    level here is, and roughly twenty times faster: `np.kron` pays for
    `expand_dims`/`normalize_axis_tuple` machinery it needs only in the general
    n-dimensional case. At these array sizes that Python overhead, not the
    arithmetic, is the whole cost -- and a streaming update is dozens of these
    per tick, so it is the difference between the update being fast and merely
    being asymptotically fast.
    """
    return np.multiply.outer(a, b).ravel()


def _words(dim: int, depth: int):
    """Tensor-algebra words of length 1..depth, in lexicographic order."""
    for level in range(1, depth + 1):
        yield from itertools.product(range(1, dim + 1), repeat=level)


def n_features(dim: int, depth: int) -> int:
    """Length of a depth-`depth` signature over a `dim`-dimensional path."""
    return sum(dim**level for level in range(1, depth + 1))


def tensor_identity(dim: int, depth: int) -> Levels:
    """The multiplicative unit: scalar 1, every higher level zero.

    This is the signature of a constant path -- a window that has not moved.
    """
    return [np.ones(1)] + [np.zeros(dim**level) for level in range(1, depth + 1)]


def to_levels(flat: np.ndarray, dim: int, depth: int, scalar: float = 1.0) -> Levels:
    """Split a flat signature vector into level arrays, restoring the scalar.

    `signature()` omits the level-0 term because it is 1 for every signature;
    the algebra needs it back to multiply.
    """
    flat = np.asarray(flat, dtype=float)
    expected = n_features(dim, depth)
    if flat.shape != (expected,):
        raise ValueError(f"expected a flat signature of length {expected}, got shape {flat.shape}")
    levels, offset = [np.full(1, float(scalar))], 0
    for level in range(1, depth + 1):
        size = dim**level
        levels.append(flat[offset : offset + size].copy())
        offset += size
    return levels


def from_levels(levels: Levels) -> np.ndarray:
    """Flatten level arrays back to `signature()`'s layout, dropping the scalar."""
    return np.concatenate(levels[1:])


def tensor_multiply(a: Levels, b: Levels, dim: int, depth: int) -> Levels:
    """Concatenation product in the truncated tensor algebra.

    Level k of the product is sum_{i+j=k} a_i (x) b_j; on flattened
    lexicographic coordinates the tensor product of two levels is exactly a
    flattened outer product. This is Chen's identity as an algorithm.
    """
    out = []
    for level in range(depth + 1):
        acc = np.zeros(dim**level)
        for left in range(level + 1):
            acc += _tensor_product(a[left], b[level - left])
        out.append(acc)
    return out


def tensor_exp(vector: np.ndarray, depth: int) -> Levels:
    """Truncated exponential exp(v) = sum_k v^{(x)k} / k!.

    The signature of a single straight-line increment `v`, and the building
    block every other construction here is made of.
    """
    vector = np.asarray(vector, dtype=float)
    levels = [np.ones(1)]
    for level in range(1, depth + 1):
        levels.append(_tensor_product(levels[-1], vector) / level)
    return levels


def tensor_inverse(a: Levels, dim: int, depth: int) -> Levels:
    """Group inverse of an element with scalar part 1.

    Solving (a * inv)_k = 0 for k >= 1 level by level gives the recursion
    inv_k = -sum_{j=1..k} a_j (x) inv_{k-j}, with inv_0 = 1. Because a_0 = 1
    the recursion is triangular, so it terminates at level `depth` -- this is a
    finite computation, not a truncated series. The same element is a two-sided
    inverse (`tests/test_algebra.py` checks both sides).

    For a signature this is the *antipode* of the Hopf algebra, and it has a
    path-level meaning: the inverse of S(x) is S(x run backwards). Reversing a
    path undoes it, and the algebra records that exactly.
    """
    if not np.isclose(a[0][0], 1.0):
        raise ValueError(f"only elements with scalar part 1 are invertible here, got {a[0][0]!r}")
    inv = [np.ones(1)]
    for level in range(1, depth + 1):
        acc = np.zeros(dim**level)
        for j in range(1, level + 1):
            acc -= _tensor_product(a[j], inv[level - j])
        inv.append(acc)
    return inv


def tensor_log(a: Levels, dim: int, depth: int) -> Levels:
    """Truncated logarithm of an element with scalar part 1.

    log(1 + x) = x - x^2/2 + x^3/3 - ... for x = a - 1. The series is exact
    rather than merely truncated: x has no level-0 part, so x^k vanishes below
    level k and only `depth` terms survive (docs/notes.html s4.3).
    """
    if not np.isclose(a[0][0], 1.0):
        raise ValueError(f"only elements with scalar part 1 have a logarithm here, got {a[0][0]!r}")
    x = [np.zeros(1)] + [a[level].copy() for level in range(1, depth + 1)]

    out = [np.zeros(dim**level) for level in range(depth + 1)]
    term, sign = x, 1.0
    for k in range(1, depth + 1):
        for level in range(depth + 1):
            out[level] = out[level] + sign * term[level] / k
        if k < depth:
            term = tensor_multiply(term, x, dim, depth)
        sign = -sign
    return out


def dilate(a: Levels, scales: np.ndarray, dim: int, depth: int) -> Levels:
    """Rescale channel `i` of the underlying path by `scales[i]`.

    Scaling a path channel is a linear map on the path, and the signature turns
    it into a diagonal map: the coefficient of a word `w` picks up
    `prod_i scales[i]` over the letters of `w`. Level `k`'s multiplier is
    therefore `scales` kron'd with itself `k` times, which lines up with the
    lexicographic layout by construction.

    `nested_suffix_signatures` needs this because `time_augment` normalises its
    time channel to [0, 1] *per window*, so a short window's time channel runs
    at a different rate than the same stretch seen inside a longer one. That is
    a one-line dilation, not a reason to recompute.
    """
    scales = np.asarray(scales, dtype=float)
    if scales.shape != (dim,):
        raise ValueError(f"expected {dim} scale factors, got shape {scales.shape}")
    out, factor = [a[0].copy()], np.ones(1)
    for level in range(1, depth + 1):
        factor = _tensor_product(factor, scales)
        out.append(a[level] * factor)
    return out


def tensor_norm(a: Levels) -> float:
    """Max absolute coordinate over levels 1..depth.

    The scale streaming drift is measured against; the scalar part is excluded
    because it is identically 1 and would flatten every ratio towards it.
    """
    return max((float(np.max(np.abs(level))) for level in a[1:]), default=0.0)
