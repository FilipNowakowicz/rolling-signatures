"""Causal path preprocessing for financial signature features."""

from __future__ import annotations

import numpy as np


def add_basepoint(path: np.ndarray) -> np.ndarray:
    """Prepend the origin so the signature records the initial level.

    Path signatures otherwise depend only on increments and are translation
    invariant.  Adding a basepoint is useful when the initial value itself is
    meaningful to the downstream model.
    """
    path = _as_path(path)
    return np.vstack((np.zeros((1, path.shape[1])), path))


def time_augment(path: np.ndarray) -> np.ndarray:
    """Append a local, uniformly spaced time channel in the interval [0, 1]."""
    path = _as_path(path)
    time = np.linspace(0.0, 1.0, num=len(path), dtype=float)[:, None]
    return np.hstack((path, time))


def lead_lag(path: np.ndarray) -> np.ndarray:
    """Return the lead-lag embedding of a discrete path.

    The rows are ``(x0, x0), (x1, x0), (x1, x1), ...``.  Its second-level
    signature captures the path's discrete quadratic variation.
    """
    path = _as_path(path)
    repeated = np.repeat(path, 2, axis=0)
    return np.hstack((repeated[1:], repeated[:-1]))


def preprocess_path(
    path: np.ndarray,
    *,
    basepoint: bool = False,
    time_augmentation: bool = False,
    lead_lag_transform: bool = False,
) -> np.ndarray:
    """Apply requested causal transforms in a fixed, documented order."""
    path = _as_path(path)
    if basepoint:
        path = add_basepoint(path)
    if time_augmentation:
        path = time_augment(path)
    if lead_lag_transform:
        path = lead_lag(path)
    return path


def _as_path(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a two-dimensional array of shape (n_points, n_channels)")
    if path.shape[0] == 0:
        raise ValueError("path must contain at least one point")
    return path
