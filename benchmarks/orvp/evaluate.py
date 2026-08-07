"""Metric, cross-validation, and the shared model class.

Every arm is scored the same way, on the same folds, with the same learner.
The only thing that varies between arms is which columns go in -- otherwise
a difference in RMSPE would say nothing about the features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

RANDOM_STATE = 0


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared *percentage* error -- the competition's metric.

    Dividing by the truth means a miss on a quiet segment counts as much as
    the same relative miss on a violent one, which is why every arm is
    trained with weights that mirror the metric rather than on raw squared
    error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))


def make_model() -> HistGradientBoostingRegressor:
    """The one learner every arm shares.

    scikit-learn's histogram gradient booster, not LightGBM: it is the same
    algorithm family, it is already a hard dependency of `rollsig`, and
    keeping the learner fixed and unremarkable is the point -- this
    benchmark is a claim about features, not about tuning.
    """
    return HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )


def folds(groups: np.ndarray, n_splits: int = 5):
    """Cross-validation splits that keep whole time_ids together.

    ORVP's `time_id`s are deliberately shuffled by the organisers, so their
    numeric order is not chronological and a genuine walk-forward split is
    not reconstructable from the shipped data. What *is* both reconstructable
    and essential is grouping: a `time_id` is a single instant of market
    time observed across all 112 stocks, and realized volatility is strongly
    correlated across stocks at the same instant. Letting one stock's row
    from a given time_id train a model that is then scored on another
    stock's row from the *same* time_id leaks the answer almost directly.
    Grouping by time_id closes that, and it is the leak that actually
    matters here.

    Reported as a limitation rather than papered over: this is grouped CV,
    not walk-forward, and the write-up says so.
    """
    return GroupKFold(n_splits=n_splits)


@dataclass
class ArmResult:
    """Out-of-fold predictions and scores for one arm."""

    arm: str
    n_features: int
    oof_rmspe: float
    fold_rmspe: list[float]
    predictions: np.ndarray

    def summary(self) -> dict:
        return {
            "arm": self.arm,
            "n_features": self.n_features,
            "rmspe": self.oof_rmspe,
            "fold_std": float(np.std(self.fold_rmspe)),
        }


def run_arm(
    arm: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> ArmResult:
    """Fit and score one feature set out-of-fold."""
    predictions = np.zeros(len(y))
    fold_scores = []
    for train_idx, test_idx in folds(groups, n_splits).split(X, y, groups):
        model = make_model()
        # Weighting by 1/y^2 under squared error is algebraically the same
        # objective as RMSPE, so the learner optimises the metric it is
        # scored on instead of a proxy.
        model.fit(X.iloc[train_idx], y[train_idx], sample_weight=1.0 / np.square(y[train_idx]))
        predictions[test_idx] = model.predict(X.iloc[test_idx])
        fold_scores.append(rmspe(y[test_idx], predictions[test_idx]))
    return ArmResult(
        arm=arm,
        n_features=X.shape[1],
        oof_rmspe=rmspe(y, predictions),
        fold_rmspe=fold_scores,
        predictions=predictions,
    )


def paired_bootstrap(
    y: np.ndarray,
    baseline: np.ndarray,
    challenger: np.ndarray,
    groups: np.ndarray,
    n_resamples: int = 2000,
    seed: int = RANDOM_STATE,
) -> dict:
    """Confidence interval on the RMSPE improvement of one arm over another.

    Resampling whole `time_id` groups, not individual rows: rows sharing a
    time_id are the same market instant seen across stocks and are nowhere
    near independent, so a row-level bootstrap would report an interval far
    tighter than the evidence supports.
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}

    observed = rmspe(y, baseline) - rmspe(y, challenger)
    deltas = np.empty(n_resamples)
    relative_deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        drawn = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([index_by_group[g] for g in drawn])
        baseline_score = rmspe(y[idx], baseline[idx])
        deltas[i] = baseline_score - rmspe(y[idx], challenger[idx])
        relative_deltas[i] = 100.0 * deltas[i] / baseline_score

    low, high = np.percentile(deltas, [2.5, 97.5])
    low_pct, high_pct = np.percentile(relative_deltas, [2.5, 97.5])
    return {
        "improvement": observed,
        "improvement_pct": 100.0 * observed / rmspe(y, baseline),
        "ci_low": float(low),
        "ci_high": float(high),
        "ci_low_pct": float(low_pct),
        "ci_high_pct": float(high_pct),
        # A one-sided read of how often resampling reverses the sign.
        "p_no_improvement": float(np.mean(deltas <= 0.0)),
    }
