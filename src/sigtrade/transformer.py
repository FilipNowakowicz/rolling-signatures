from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from sigtrade._backend import Backend, _words, n_features, signature


class SignatureTransformer(BaseEstimator, TransformerMixin):
    """Causal, rolling-window path-signature features.

    For each time step `t`, the output row is the depth-`depth` signature
    of the most recent `window` points ending at and including `t` (fewer
    points at the start of the series, when a full window isn't yet
    available). The row at `t` is a function of `X[:t+1]` only.
    """

    def __init__(self, window: int = 20, depth: int = 2, backend: Backend = "roughpy"):
        self.window = window
        self.depth = depth
        self.backend = backend

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        self.n_output_features_ = n_features(self.n_features_in_, self.depth)
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}")

        n_samples = X.shape[0]
        out = np.empty((n_samples, self.n_output_features_))
        for t in range(n_samples):
            start = max(0, t + 1 - self.window)
            out[t] = signature(X[start : t + 1], self.depth, backend=self.backend)
        return out

    def get_feature_names_out(self, input_features=None):
        words = _words(self.n_features_in_, self.depth)
        return np.array(["sig_" + "".join(map(str, w)) for w in words])
