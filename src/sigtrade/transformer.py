from __future__ import annotations

from math import factorial
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from sigtrade._backend import (
    _LOG_LABELS,
    Backend,
    _words,
    log_signature,
    n_features,
    n_log_features,
    signature,
)
from sigtrade.preprocessing import preprocess_path

Output = Literal["signature", "log_signature"]


class SignatureTransformer(BaseEstimator, TransformerMixin):
    """Causal, rolling-window path-signature features.

    For each time step `t`, the output row is the depth-`depth` signature
    of the most recent `window` points ending at and including `t` (fewer
    points at the start of the series, when a full window isn't yet
    available). The row at `t` is a function of `X[:t+1]` only.
    """

    def __init__(
        self,
        window: int = 20,
        depth: int = 2,
        backend: Backend = "roughpy",
        basepoint: bool = False,
        time_augmentation: bool = False,
        lead_lag_transform: bool = False,
        rescale: bool = False,
        output: Output = "signature",
    ):
        self.window = window
        self.depth = depth
        self.backend = backend
        self.basepoint = basepoint
        self.time_augmentation = time_augmentation
        self.lead_lag_transform = lead_lag_transform
        self.rescale = rescale
        self.output = output

    def fit(self, X, y=None):
        X = check_array(X)
        self._validate_parameters()
        self.n_features_in_ = X.shape[1]
        transformed_dim = self.n_features_in_ + int(self.time_augmentation)
        if self.lead_lag_transform:
            transformed_dim *= 2
        if self.output == "log_signature":
            self.n_output_features_ = n_log_features(transformed_dim, self.depth)
        else:
            self.n_output_features_ = n_features(transformed_dim, self.depth)
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}")

        n_samples = X.shape[0]
        out = np.empty((n_samples, self.n_output_features_))
        compute = log_signature if self.output == "log_signature" else signature
        for t in range(n_samples):
            start = max(0, t + 1 - self.window)
            path = preprocess_path(
                X[start : t + 1],
                basepoint=self.basepoint,
                time_augmentation=self.time_augmentation,
                lead_lag_transform=self.lead_lag_transform,
            )
            out[t] = compute(path, self.depth, backend=self.backend)
        if self.rescale:
            out *= np.concatenate(
                [np.full(self._transformed_dim**level, factorial(level)) for level in range(1, self.depth + 1)]
            )
        return out

    def get_feature_names_out(self, input_features=None):
        if self.output == "log_signature":
            # Backend-specific: the two log-signature backends use different
            # bases of the free Lie algebra (see `log_signature`), so the
            # names must come from whichever one produced the columns.
            labels = _LOG_LABELS[self.backend](self._transformed_dim, self.depth)
            return np.array(["logsig_" + label for label in labels])
        words = _words(self._transformed_dim, self.depth)
        return np.array(["sig_" + "".join(map(str, w)) for w in words])

    @property
    def _transformed_dim(self):
        dim = self.n_features_in_ + int(self.time_augmentation)
        return 2 * dim if self.lead_lag_transform else dim

    def _validate_parameters(self):
        if not isinstance(self.window, int) or isinstance(self.window, bool) or self.window < 1:
            raise ValueError("window must be a positive integer")
        if not isinstance(self.depth, int) or isinstance(self.depth, bool) or self.depth < 1:
            raise ValueError("depth must be a positive integer")
        if self.output not in ("signature", "log_signature"):
            raise ValueError("output must be 'signature' or 'log_signature'")
        if self.output == "log_signature" and self.rescale:
            raise ValueError("rescale is not yet implemented for output='log_signature'")
