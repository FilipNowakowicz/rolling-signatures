"""Compiled (Numba) kernel for the streaming update.

This is the same algorithm as `rollsig.algebra` and `rollsig.streaming`,
written as loops over flat arrays so that Numba can compile it. The readable
numpy version stays the reference; `tests/test_compiled.py` holds this one to it.

Layout. An element of the truncated tensor algebra is one flat float64 array,
levels 0..depth back to back, with `offs[k]` the start of level k and
`offs[depth + 1]` the total length. Level k of a product is
sum_{i+j=k} a_i (x) b_j, and on lexicographic word order the word (x, y) of
levels (i, j) sits at x * C**j + y, so the tensor product is an outer product
written straight into the output.

What the kernel does per tick is exactly what `StreamingSignature._slide`
does: delete the departing increment by left-multiplying the exp(-v) of its
segments, append the arriving one by right-multiplying exp(v), and re-anchor
with a from-scratch pass over the buffered window when due. The re-anchor is
compiled too; leaving it to numpy would make an O(window) interpreted pass the
dominant cost once the slide is fast.

Importing this module imports Numba; `rollsig.streaming` does so lazily and
falls back to pure numpy when it is not installed.
"""

from __future__ import annotations

import numpy as np
from numba import njit


def level_offsets(channels: int, depth: int) -> np.ndarray:
    sizes = [channels**k for k in range(depth + 1)]
    return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)


@njit(cache=True)
def _block(delta, dt, dim, time_aug, lead_lag, blk):
    """Fill `blk` with the segments one raw increment becomes; return how many."""
    width = dim + 1 if time_aug else dim
    blk[:, :] = 0.0
    for c in range(dim):
        blk[0, c] = delta[c]
    if time_aug:
        blk[0, dim] = dt
    if not lead_lag:
        return 1
    for c in range(width):
        blk[1, width + c] = blk[0, c]
    return 2


@njit(cache=True)
def _right(state, v, sign, work, offs, depth, channels):
    """state <- state * exp(sign * v), in place.

    Level k of the product is sum_m S_{k-m} (x) v^m / m!, evaluated by Horner's
    rule in m so that no exponential is formed and no full product is taken:
    h = S_0; h = S_m + h (x) v / (k - m + 1) for m = 1..k. Levels are updated
    from the top down, so every level read is still the old one.
    """
    h = work[0]
    g = work[1]
    for k in range(depth, 0, -1):
        h[0] = state[0]
        hlen = 1
        for m in range(1, k + 1):
            base = offs[m]
            scale = 1.0 / (k - m + 1)
            if m == k:
                for x in range(hlen):
                    hx = h[x] * scale
                    for y in range(channels):
                        state[base + x * channels + y] += hx * (sign * v[y])
            else:
                for x in range(hlen):
                    hx = h[x] * scale
                    for y in range(channels):
                        g[x * channels + y] = state[base + x * channels + y] + hx * (sign * v[y])
                h, g = g, h
            hlen *= channels
        h = work[0]
        g = work[1]


@njit(cache=True)
def _left(state, v, sign, work, offs, depth, channels):
    """state <- exp(sign * v) * state, in place; the mirror image of `_right`."""
    h = work[0]
    g = work[1]
    for k in range(depth, 0, -1):
        h[0] = state[0]
        hlen = 1
        for m in range(1, k + 1):
            base = offs[m]
            scale = 1.0 / (k - m + 1)
            if m == k:
                for c in range(channels):
                    vc = sign * v[c] * scale
                    for z in range(hlen):
                        state[base + c * hlen + z] += vc * h[z]
            else:
                for c in range(channels):
                    vc = sign * v[c] * scale
                    for z in range(hlen):
                        g[c * hlen + z] = state[base + c * hlen + z] + vc * h[z]
                h, g = g, h
            hlen *= channels
        h = work[0]
        g = work[1]


@njit(cache=True)
def _identity(state):
    state[:] = 0.0
    state[0] = 1.0


@njit(cache=True)
def _recompute(state, buf, ctr, work, blk, offs, depth, channels, dim, window, time_aug, lead_lag):
    count = ctr[0]
    head = ctr[1]
    dt = 1.0 / (count - 1) if count > 1 else 0.0
    _identity(state)
    delta = np.empty(dim)
    for n in range(1, count):
        a = buf[(head + n - 1) % window]
        b = buf[(head + n) % window]
        for c in range(dim):
            delta[c] = b[c] - a[c]
        nseg = _block(delta, dt, dim, time_aug, lead_lag, blk)
        for s in range(nseg):
            _right(state, blk[s], 1.0, work, offs, depth, channels)
    ctr[2] = 0
    ctr[3] += 1


@njit(cache=True)
def _step(state, buf, ctr, point, work, blk, offs, depth, channels, dim, window, time_aug, lead_lag, refresh):
    """One tick. `ctr` = [points held, index of oldest, ticks since re-anchor, re-anchors]."""
    count = ctr[0]
    head = ctr[1]
    was_full = count == window
    has_previous = count > 0
    delta_in = np.empty(dim)
    delta_out = np.empty(dim)
    if has_previous:
        last = buf[(head + count - 1) % window]
        for c in range(dim):
            delta_in[c] = point[c] - last[c]
    if was_full:
        old = buf[head]
        nxt = buf[(head + 1) % window]
        for c in range(dim):
            delta_out[c] = nxt[c] - old[c]
        buf[head, :] = point
        ctr[1] = (head + 1) % window
    else:
        buf[(head + count) % window, :] = point
        ctr[0] = count + 1

    if not has_previous:
        _identity(state)
        ctr[2] = 0
        return

    due = refresh > 0 and ctr[2] >= refresh
    must = time_aug and not was_full
    if due or must:
        _recompute(state, buf, ctr, work, blk, offs, depth, channels, dim, window, time_aug, lead_lag)
        return

    dt = 1.0 / (window - 1)
    if was_full:
        nseg = _block(delta_out, dt, dim, time_aug, lead_lag, blk)
        # inv = exp(-v_n)...exp(-v_1), so inv * S applies exp(-v_1) first.
        for s in range(nseg):
            _left(state, blk[s], -1.0, work, offs, depth, channels)
    nseg = _block(delta_in, dt, dim, time_aug, lead_lag, blk)
    for s in range(nseg):
        _right(state, blk[s], 1.0, work, offs, depth, channels)
    ctr[2] += 1


@njit(cache=True)
def _extend(out, points, state, buf, ctr, work, blk, offs, depth, channels, dim, window, time_aug, lead_lag, refresh):
    for n in range(points.shape[0]):
        _step(state, buf, ctr, points[n], work, blk, offs, depth, channels, dim, window, time_aug, lead_lag, refresh)
        out[n, :] = state[1:]
