"""O(1)-per-tick sliding-window signatures.

Recomputing a rolling signature from scratch at every tick costs O(window),
and almost all of that work is re-traversing path the previous tick already
traversed. The tensor algebra makes the redundancy removable, because a
signature is a grouplike element and both ends of a window are group
operations (`sigtrade.algebra`):

    S(new window) = S(departing increment)^-1 * S(old window) * S(arriving increment)

Left-multiplying by an inverse deletes the departing segment; right-multiplying
adds the new one. Neither touches the interior of the window, so the cost per
tick depends on `depth` and the channel count and *not at all* on `window`.

Three practical points, all of which the benchmark in `benchmarks/streaming/`
measures rather than assumes:

1. The inverse of a straight-line increment's signature is available in closed
   form -- `exp(v)^-1 = exp(-v)` -- so the hot path never runs the general
   `tensor_inverse` recursion. The general inverse is still what makes the
   identity true, and `suffix_signature` below is the case where it is the
   only route.
2. Preprocessing survives the move. Every option in `sigtrade.preprocessing`
   except `basepoint` turns one raw increment into a fixed-size *block* of
   preprocessed increments, so sliding stays a constant-size operation.
   `basepoint` does not: it pins the path to the origin, so every slide
   rewrites the leading increment, and streaming is refused for it.
3. Repeated group operations accumulate floating-point error where a
   recomputation would not. `refresh_every` re-anchors on a real
   recomputation; doing so once per `window` ticks is still O(1) amortized,
   which is why that is the default.
"""

from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np

from sigtrade._backend import _signature_levels_numpy
from sigtrade.algebra import (
    Levels,
    dilate,
    from_levels,
    n_features,
    tensor_exp,
    tensor_identity,
    tensor_inverse,
    tensor_multiply,
    to_levels,
)
from sigtrade.preprocessing import preprocess_path

Refresh = Literal["auto"] | int | None


class _PathEncoding:
    """How one raw increment becomes a block of preprocessed increments.

    This is the whole reason preprocessing is compatible with streaming. The
    preprocessed path is not some opaque transform of the window: it is a
    concatenation of per-increment blocks, verified against
    `preprocessing.preprocess_path` in `tests/test_streaming.py`.

        plain        d          -> [d]
        time-aug     d          -> [(d, dt)]
        lead-lag     d          -> [(d, 0), (0, d)]
        both         d          -> [(d, dt, 0, 0), (0, 0, d, dt)]

    `dt` is constant once the window is full -- `time_augment` spreads [0, 1]
    over a fixed number of points -- which is exactly what lets the time
    channel slide along with everything else.
    """

    def __init__(self, dim: int, time_augmentation: bool, lead_lag_transform: bool):
        self.dim = dim
        self.time_augmentation = time_augmentation
        self.lead_lag_transform = lead_lag_transform

    @property
    def channels(self) -> int:
        """Dimension of the preprocessed path."""
        dim = self.dim + int(self.time_augmentation)
        return 2 * dim if self.lead_lag_transform else dim

    @property
    def time_channels(self) -> tuple[int, ...]:
        """Indices of the augmented time channel(s) in the preprocessed path."""
        if not self.time_augmentation:
            return ()
        augmented = self.dim + 1
        if self.lead_lag_transform:
            return (self.dim, augmented + self.dim)
        return (self.dim,)

    def block(self, delta: np.ndarray, dt: float) -> list[np.ndarray]:
        vector = np.concatenate([delta, [dt]]) if self.time_augmentation else np.asarray(delta, dtype=float)
        if not self.lead_lag_transform:
            return [vector]
        zeros = np.zeros_like(vector)
        return [np.concatenate([vector, zeros]), np.concatenate([zeros, vector])]


def _block_signature(block: list[np.ndarray], dim: int, depth: int) -> Levels:
    """Signature of the straight-line segments in one increment block."""
    levels = tensor_identity(dim, depth)
    for vector in block:
        levels = tensor_multiply(levels, tensor_exp(vector, depth), dim, depth)
    return levels


def _block_inverse(block: list[np.ndarray], dim: int, depth: int) -> Levels:
    """Group inverse of `_block_signature`, in closed form.

    (exp(a) exp(b))^-1 = exp(-b) exp(-a): reverse the segments and negate them.
    Running the segments backwards undoes them, which is the path-level reading
    of the antipode. `tests/test_streaming.py` pins this against the general
    `tensor_inverse` recursion -- they agree to machine precision, and this one
    is both cheaper and better conditioned.
    """
    levels = tensor_identity(dim, depth)
    for vector in reversed(block):
        levels = tensor_multiply(levels, tensor_exp(-vector, depth), dim, depth)
    return levels


class StreamingSignature:
    """Rolling depth-`depth` signature of the last `window` points, O(1) per tick.

    Push points in with `update`; each call returns the signature of the window
    ending at that point, in the same layout and with the same causal guarantee
    as `SignatureTransformer` -- the value returned at tick `t` is a function of
    the points pushed at times <= `t`, which is structural here rather than
    merely tested, since no future point has been supplied yet.

    Before the window fills, the output is the signature of however many points
    have arrived, matching `SignatureTransformer`'s handling of the start of a
    series.

    Parameters
    ----------
    window, depth
        As in `SignatureTransformer`.
    dim
        Number of channels of the *raw* input points.
    time_augmentation, lead_lag_transform
        As in `SignatureTransformer`. `basepoint` has no streaming form and is
        not offered -- see the module docstring.
    refresh_every
        How often to re-anchor on a from-scratch recomputation, to stop
        floating-point drift accumulating. `"auto"` (the default) is every
        `window` ticks, which bounds the drift while staying O(1) amortized;
        an integer sets the period explicitly; `None` never re-anchors, which
        is what `benchmarks/streaming/` uses to measure drift in the raw.
    """

    def __init__(
        self,
        window: int = 20,
        depth: int = 2,
        dim: int = 1,
        *,
        time_augmentation: bool = False,
        lead_lag_transform: bool = False,
        refresh_every: Refresh = "auto",
    ):
        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            raise ValueError("window must be a positive integer")
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
            raise ValueError("depth must be a positive integer")
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise ValueError("dim must be a positive integer")
        if refresh_every == "auto":
            refresh_every = window
        elif refresh_every is not None and (
            not isinstance(refresh_every, int) or isinstance(refresh_every, bool) or refresh_every < 1
        ):
            raise ValueError("refresh_every must be 'auto', a positive integer, or None")

        self.window = window
        self.depth = depth
        self.dim = dim
        self.time_augmentation = time_augmentation
        self.lead_lag_transform = lead_lag_transform
        self.refresh_every = refresh_every

        self._encoding = _PathEncoding(dim, time_augmentation, lead_lag_transform)
        # Spacing of the augmented time channel once the window is full. Under
        # a partial window it differs, which is what `_recompute` is for.
        self._dt = 1.0 / (window - 1) if window > 1 else 0.0
        self._buffer: deque[np.ndarray] = deque(maxlen=window)
        self.reset()

    @property
    def n_channels(self) -> int:
        """Dimension of the preprocessed path the signature is taken over."""
        return self._encoding.channels

    @property
    def n_output_features(self) -> int:
        return n_features(self.n_channels, self.depth)

    @property
    def signature(self) -> np.ndarray:
        """Current window's signature, levels 1..depth flattened."""
        return from_levels(self._levels)

    def reset(self) -> None:
        """Forget the stream. Counters go back to zero."""
        self._buffer.clear()
        self._levels = tensor_identity(self.n_channels, self.depth)
        self._since_recompute = 0
        self.n_updates_ = 0
        self.n_recomputes_ = 0

    def update(self, point) -> np.ndarray:
        """Push one point and return the new window's signature."""
        point = np.asarray(point, dtype=float).reshape(-1)
        if point.shape != (self.dim,):
            raise ValueError(f"expected a point with {self.dim} channels, got shape {point.shape}")

        if self.window == 1:
            # A one-point window has no increments, so its signature is the
            # identity forever and there is nothing to slide.
            self._buffer.append(point)
            self.n_updates_ += 1
            return self.signature

        was_full = len(self._buffer) == self.window
        departing = self._buffer[1] - self._buffer[0] if was_full else None
        previous = self._buffer[-1] if self._buffer else None
        self._buffer.append(point)  # a maxlen deque evicts the oldest point for us
        self.n_updates_ += 1

        if previous is None:
            self._levels = tensor_identity(self.n_channels, self.depth)
            self._since_recompute = 0
            return self.signature

        due = self.refresh_every is not None and self._since_recompute >= self.refresh_every
        # A window that is not yet full renormalises the time channel on every
        # tick, so the sliding identity does not apply to it: dt changes, and
        # with it every increment already in the accumulated signature.
        must_recompute = self.time_augmentation and not was_full
        if due or must_recompute:
            self._recompute()
        else:
            self._slide(departing, point - previous)
        return self.signature

    def extend(self, points) -> np.ndarray:
        """Push several points; return one signature row per point."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2:
            raise ValueError("points must be a two-dimensional array of shape (n_points, dim)")
        return np.array([self.update(point) for point in points])

    def _slide(self, departing: np.ndarray | None, arriving: np.ndarray) -> None:
        """The O(1) update: drop the left end by an inverse, add the right end."""
        dim, depth = self.n_channels, self.depth
        if departing is not None:
            block = self._encoding.block(departing, self._dt)
            self._levels = tensor_multiply(_block_inverse(block, dim, depth), self._levels, dim, depth)
        for vector in self._encoding.block(arriving, self._dt):
            self._levels = tensor_multiply(self._levels, tensor_exp(vector, depth), dim, depth)
        self._since_recompute += 1

    def _recompute(self) -> None:
        """Re-anchor on a from-scratch signature of the buffered window."""
        path = preprocess_path(
            np.asarray(self._buffer, dtype=float),
            time_augmentation=self.time_augmentation,
            lead_lag_transform=self.lead_lag_transform,
        )
        self._levels = _signature_levels_numpy(path, self.depth)
        self._since_recompute = 0
        self.n_recomputes_ += 1


def rolling_signature(
    X,
    window: int = 20,
    depth: int = 2,
    *,
    time_augmentation: bool = False,
    lead_lag_transform: bool = False,
    refresh_every: Refresh = "auto",
) -> np.ndarray:
    """Causal rolling signatures of `X`, one row per observation.

    The functional form of `StreamingSignature`, and a drop-in match for
    `SignatureTransformer(...).fit_transform(X)` with the same options and
    `basepoint=False`.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array of shape (n_samples, n_channels)")
    engine = StreamingSignature(
        window=window,
        depth=depth,
        dim=X.shape[1],
        time_augmentation=time_augmentation,
        lead_lag_transform=lead_lag_transform,
        refresh_every=refresh_every,
    )
    return engine.extend(X)


def suffix_signature(window_signature, prefix_signature, dim: int, depth: int) -> np.ndarray:
    """Signature of a path suffix, from the whole path's and the prefix's.

    Chen's identity says S(prefix . suffix) = S(prefix) * S(suffix); the group
    inverse solves it for the unknown factor:

        S(suffix) = S(prefix)^-1 * S(whole)

    Use it when the long window's signature is already in hand and the prefix's
    is cheap — dropping the front of a cached window without re-reading its
    tail. When you are computing everything from raw path anyway,
    `nested_suffix_signatures` is strictly less work; see its docstring.

    Both arguments and the return value are flat `n_features(dim, depth)`
    vectors, the layout `signature()` uses.
    """
    whole = to_levels(window_signature, dim, depth)
    prefix = to_levels(prefix_signature, dim, depth)
    return from_levels(tensor_multiply(tensor_inverse(prefix, dim, depth), whole, dim, depth))


def nested_suffix_signatures(
    path,
    lengths,
    depth: int = 2,
    *,
    time_augmentation: bool = False,
    lead_lag_transform: bool = False,
) -> dict[int, np.ndarray]:
    """Signatures over several nested suffix windows, sharing the shared part.

    `benchmarks/orvp` takes each segment's last 600, 300 and 150 seconds and
    computes a signature over each. The 150-point path is a suffix of the
    300-point path, which is a suffix of the 600-point one, so the naive route
    walks 1050 points of path where 600 exist. Splitting the longest window
    into *disjoint* chunks at the shorter windows' boundaries and combining
    them right-to-left with Chen's identity walks each point exactly once.

    Note which half of the algebra that uses. `docs/notes-orvp.html` s9.1
    proposed getting the suffixes by left-multiplying the full window's
    signature by the inverse of the departing prefix, and that is correct but
    slower: it needs the full window *and* every prefix, which is the same 1050
    points again. Group inverses are what a sliding window needs, because there
    the shared part cannot be re-decomposed. A nested family can be, so it
    should be. (`suffix_signature` implements the inverse route for the case
    where the long window's signature is already cached.)

    Returns a dict keyed by the requested lengths. Each entry equals
    `signature(preprocess_path(path[-length:], ...), depth)`.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a two-dimensional array of shape (n_points, n_channels)")
    lengths = sorted({int(length) for length in lengths})
    if not lengths:
        raise ValueError("lengths must not be empty")
    if lengths[0] < 1:
        raise ValueError("lengths must be positive")
    if lengths[-1] > len(path):
        raise ValueError(f"longest window {lengths[-1]} exceeds the {len(path)} points available")

    longest = lengths[-1]
    encoding = _PathEncoding(path.shape[1], time_augmentation, lead_lag_transform)
    channels = encoding.channels
    increments = np.diff(path[-longest:], axis=0)
    # Time spacing of the *longest* window; shorter ones are corrected below.
    dt = 1.0 / (longest - 1) if longest > 1 else 0.0

    results: dict[int, np.ndarray] = {}
    accumulated = tensor_identity(channels, depth)
    covered = 0
    for length in lengths:  # ascending: grow the shared suffix leftwards
        needed = length - 1
        chunk = increments[len(increments) - needed : len(increments) - covered]
        if len(chunk):
            chunk_levels = tensor_identity(channels, depth)
            for increment in chunk:
                for vector in encoding.block(increment, dt):
                    chunk_levels = tensor_multiply(chunk_levels, tensor_exp(vector, depth), channels, depth)
            accumulated = tensor_multiply(chunk_levels, accumulated, channels, depth)
        covered = needed
        results[length] = from_levels(_rescale_time(accumulated, encoding, longest, length, depth))
    return results


def _rescale_time(levels: Levels, encoding: _PathEncoding, longest: int, length: int, depth: int) -> Levels:
    """Put a sub-window's time channel back on its own [0, 1] scale.

    Inside the longest window the time channel advances by 1/(longest-1) per
    step; computed standalone, a length-`length` window advances by
    1/(length-1). The two paths differ by a dilation of one channel, so the
    correction is `algebra.dilate` rather than a recomputation.
    """
    channels = encoding.time_channels
    if not channels or length < 2:
        return levels
    scales = np.ones(encoding.channels)
    scales[list(channels)] = (longest - 1) / (length - 1)
    return dilate(levels, scales, encoding.channels, depth)
