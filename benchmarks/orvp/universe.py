"""Run the pre-specified full-universe ORVP confirmation study.

See ``UNIVERSE_STUDY.md`` for the protocol fixed before scores were computed.

    python -m benchmarks.orvp.universe --jobs 8
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from datetime import UTC, datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from benchmarks.orvp import data, evaluate, features
from benchmarks.orvp.run import RESULTS_DIR, build_table

ARMS = ("book+har", "multisig+book+har")
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
OUTER_SPLITS = 5
INNER_SPLITS = 3


@dataclass
class RidgeResult:
    predictions: np.ndarray
    fold_rmspe: list[float]
    selected_alphas: list[float]


def arm_frame(table: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Select one arm and append the common stock fixed-effect column."""
    feature_columns = table.drop(columns=["stock_id", "time_id", "target"])
    columns = [c for c in feature_columns if c.startswith(features.ARM_PREFIXES[arm])]
    return feature_columns[columns].assign(stock_id=table["stock_id"].to_numpy())


def ridge_transformer(numeric_columns: list[str]) -> ColumnTransformer:
    """Fold-local imputation, scaling, and stock-effect encoding."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, numeric_columns),
            ("stock", OneHotEncoder(handle_unknown="ignore"), ["stock_id"]),
        ],
        sparse_threshold=1.0,
    )
def rmspe_weights(y: np.ndarray) -> np.ndarray:
    """RMSPE-equivalent weights normalised to keep ridge alpha interpretable."""
    weights = 1.0 / np.square(y)
    return weights / weights.mean()


def run_nested_ridge(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    outer_splits: int = OUTER_SPLITS,
    inner_splits: int = INNER_SPLITS,
    alphas: tuple[float, ...] = RIDGE_ALPHAS,
) -> RidgeResult:
    """Generate outer-fold OOF predictions with group-nested alpha choice."""
    predictions = np.zeros(len(y))
    fold_scores: list[float] = []
    selected_alphas: list[float] = []
    numeric_columns = [c for c in X.columns if c != "stock_id"]

    outer = evaluate.folds(groups, outer_splits)
    for fold_number, (train_idx, test_idx) in enumerate(outer.split(X, y, groups), start=1):
        inner_groups = groups[train_idx]
        alpha_predictions = {alpha: np.zeros(len(train_idx)) for alpha in alphas}
        inner = evaluate.folds(inner_groups, inner_splits)
        for inner_train, inner_test in inner.split(X.iloc[train_idx], y[train_idx], inner_groups):
            fit_rows = train_idx[inner_train]
            validation_rows = train_idx[inner_test]
            transform = ridge_transformer(numeric_columns)
            transformed_train = transform.fit_transform(X.iloc[fit_rows])
            transformed_validation = transform.transform(X.iloc[validation_rows])
            for alpha in alphas:
                model = Ridge(alpha=alpha, solver="lsqr", tol=1e-4)
                model.fit(
                    transformed_train,
                    y[fit_rows],
                    sample_weight=rmspe_weights(y[fit_rows]),
                )
                alpha_predictions[alpha][inner_test] = model.predict(transformed_validation)

        alpha_scores = [
            (evaluate.rmspe(y[train_idx], alpha_predictions[alpha]), alpha) for alpha in alphas
        ]

        # The numeric alpha is the deterministic tie-breaker.
        _, selected = min(alpha_scores)
        transform = ridge_transformer(numeric_columns)
        transformed_train = transform.fit_transform(X.iloc[train_idx])
        transformed_test = transform.transform(X.iloc[test_idx])
        model = Ridge(alpha=selected, solver="lsqr", tol=1e-4)
        model.fit(
            transformed_train,
            y[train_idx],
            sample_weight=rmspe_weights(y[train_idx]),
        )
        predictions[test_idx] = model.predict(transformed_test)
        fold_score = evaluate.rmspe(y[test_idx], predictions[test_idx])
        fold_scores.append(fold_score)
        selected_alphas.append(selected)
        print(f"    ridge fold {fold_number}: alpha={selected:g}, RMSPE={fold_score:.5f}")

    return RidgeResult(predictions, fold_scores, selected_alphas)


def per_stock_metrics(
    table: pd.DataFrame, baseline: np.ndarray, challenger: np.ndarray
) -> pd.DataFrame:
    """Descriptive heterogeneity table; no subgroup is promoted post hoc."""
    rows = []
    y = table["target"].to_numpy()
    stock_ids = table["stock_id"].to_numpy()
    for stock_id in sorted(np.unique(stock_ids)):
        mask = stock_ids == stock_id
        base_score = evaluate.rmspe(y[mask], baseline[mask])
        challenger_score = evaluate.rmspe(y[mask], challenger[mask])
        rows.append(
            {
                "stock_id": int(stock_id),
                "n_segments": int(mask.sum()),
                "baseline_rmspe": base_score,
                "challenger_rmspe": challenger_score,
                "improvement_pct": 100.0 * (base_score - challenger_score) / base_score,
            }
        )
    return pd.DataFrame(rows)


def learner_summary(
    table: pd.DataFrame,
    baseline: np.ndarray,
    challenger: np.ndarray,
    *,
    feature_counts: dict[str, int],
    fold_rmspe: dict[str, list[float]],
    fit_seconds: float,
    selected_alphas: dict[str, list[float]] | None = None,
) -> tuple[dict, pd.DataFrame]:
    y = table["target"].to_numpy()
    groups = table["time_id"].to_numpy()
    stocks = per_stock_metrics(table, baseline, challenger)
    comparison = evaluate.paired_bootstrap(y, baseline, challenger, groups)
    result = {
        "arms": {
            ARMS[0]: {
                "n_features": feature_counts[ARMS[0]],
                "rmspe": evaluate.rmspe(y, baseline),
                "fold_rmspe": fold_rmspe[ARMS[0]],
            },
            ARMS[1]: {
                "n_features": feature_counts[ARMS[1]],
                "rmspe": evaluate.rmspe(y, challenger),
                "fold_rmspe": fold_rmspe[ARMS[1]],
            },
        },
        "comparison": comparison,
        "stock_heterogeneity": {
            "n_stocks_improved": int((stocks["improvement_pct"] > 0).sum()),
            "n_stocks": int(len(stocks)),
            "median_improvement_pct": float(stocks["improvement_pct"].median()),
            "q10_improvement_pct": float(stocks["improvement_pct"].quantile(0.1)),
            "q90_improvement_pct": float(stocks["improvement_pct"].quantile(0.9)),
        },
        "fit_seconds": fit_seconds,
    }
    if selected_alphas is not None:
        result["selected_alphas"] = selected_alphas
    return result, stocks


def markdown(payload: dict) -> str:
    lines = [
        "| Learner | Baseline RMSPE | Challenger RMSPE | Improvement | 95% grouped CI | Stocks improved |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in payload["learners"].items():
        comparison = result["comparison"]
        heterogeneity = result["stock_heterogeneity"]
        lines.append(
            f"| {name} | {result['arms'][ARMS[0]]['rmspe']:.5f} | "
            f"{result['arms'][ARMS[1]]['rmspe']:.5f} | {comparison['improvement_pct']:+.2f}% | "
            f"[{comparison['ci_low_pct']:+.2f}%, {comparison['ci_high_pct']:+.2f}%] | "
            f"{heterogeneity['n_stocks_improved']}/{heterogeneity['n_stocks']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1, help="parallel feature-building workers")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "universe.json")
    args = parser.parse_args()

    root = data.data_dir()
    if not data.is_available(root):
        raise SystemExit(f"no ORVP data at {root}; see benchmarks/orvp/README.md")
    targets = data.load_targets(root)
    stocks = sorted(int(stock) for stock in targets["stock_id"].unique())
    missing = [
        stock
        for stock in stocks
        if not (root / "book_train.parquet" / f"stock_id={stock}").is_dir()
        or not (root / "trade_train.parquet" / f"stock_id={stock}").is_dir()
    ]
    if missing:
        raise SystemExit(f"full-universe study is missing inputs for stocks: {missing}")

    # The frozen comparison does not use the price-only signature arm.
    spec = features.SignatureSpec(windows=())
    multi_spec = features.MultiSignatureSpec()
    started = time.perf_counter()
    table = build_table(stocks, targets, spec, root, jobs=args.jobs, multi_spec=multi_spec)
    build_seconds = time.perf_counter() - started
    print(f"feature table: {len(table)} segments, {len(stocks)} stocks, {build_seconds:.1f}s")

    matrices = {arm: arm_frame(table, arm) for arm in ARMS}
    feature_counts = {arm: int(matrix.shape[1]) for arm, matrix in matrices.items()}
    y = table["target"].to_numpy()
    groups = table["time_id"].to_numpy()
    predictions: dict[str, np.ndarray] = {}

    print("\nfixed HistGradientBoostingRegressor")
    hgb_started = time.perf_counter()
    hgb_results = {
        arm: evaluate.run_arm(arm, matrices[arm], y, groups, n_splits=OUTER_SPLITS)
        for arm in ARMS
    }
    for arm, result in hgb_results.items():
        predictions[f"hgb_{arm}"] = result.predictions
        print(f"  {arm}: {result.oof_rmspe:.5f}")
    hgb_summary, hgb_stocks = learner_summary(
        table,
        hgb_results[ARMS[0]].predictions,
        hgb_results[ARMS[1]].predictions,
        feature_counts=feature_counts,
        fold_rmspe={arm: hgb_results[arm].fold_rmspe for arm in ARMS},
        fit_seconds=time.perf_counter() - hgb_started,
    )

    print("\nweighted ridge with nested grouped alpha selection")
    ridge_started = time.perf_counter()
    ridge_results = {arm: run_nested_ridge(matrices[arm], y, groups) for arm in ARMS}
    for arm, result in ridge_results.items():
        predictions[f"ridge_{arm}"] = result.predictions
    ridge_summary, ridge_stocks = learner_summary(
        table,
        ridge_results[ARMS[0]].predictions,
        ridge_results[ARMS[1]].predictions,
        feature_counts=feature_counts,
        fold_rmspe={arm: ridge_results[arm].fold_rmspe for arm in ARMS},
        fit_seconds=time.perf_counter() - ridge_started,
        selected_alphas={arm: ridge_results[arm].selected_alphas for arm in ARMS},
    )

    payload = {
        "config": {
            "command": f"uv run python -m benchmarks.orvp.universe --jobs {args.jobs}",
            "run_at_utc": datetime.now(UTC).isoformat(),
            "software": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": importlib.metadata.version("scikit-learn"),
                "iisignature": importlib.metadata.version("iisignature"),
            },
            "n_stocks": len(stocks),
            "stocks": stocks,
            "n_segments": int(len(table)),
            "outer_splits": OUTER_SPLITS,
            "ridge_inner_splits": INNER_SPLITS,
            "ridge_alphas": list(RIDGE_ALPHAS),
            "multi_depth": multi_spec.depth,
            "multi_windows": list(multi_spec.windows),
            "multi_channels": list(multi_spec.channels),
            "build_seconds": build_seconds,
        },
        "learners": {"fixed_hgb": hgb_summary, "nested_ridge": ridge_summary},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    table_path = args.out.with_name("universe_table.md")
    table_path.write_text(markdown(payload) + "\n")
    pd.concat(
        {"fixed_hgb": hgb_stocks, "nested_ridge": ridge_stocks}, names=["learner"]
    ).reset_index(level=0).to_csv(args.out.with_name("universe_per_stock.csv"), index=False)
    np.savez_compressed(
        args.out.with_name("universe_oof_predictions.npz"),
        y=y,
        stock_id=table["stock_id"].to_numpy(),
        time_id=groups,
        **predictions,
    )
    print("\n" + markdown(payload))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
