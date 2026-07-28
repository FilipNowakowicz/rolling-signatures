"""Build the feature table and score every arm.

    python -m benchmarks.orvp.run --stocks 20 --jobs 8

Writes `benchmarks/orvp/results/benchmark.json` and a markdown table ready
to paste into the README.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.orvp import data, evaluate, features

RESULTS_DIR = Path(__file__).resolve().parent / "results"

ARMS = ["har", "book", "book+har", "sig", "sig+har", "sig+book", "sig+book+har"]


def cache_path(stock_id: int, spec: features.SignatureSpec, root: Path) -> Path:
    windows = "-".join(str(w) for w in spec.windows)
    tag = f"d{spec.depth}_w{windows}_s{spec.subsample}_ta{int(spec.time_augmentation)}_ll{int(spec.lead_lag_transform)}"
    return root / "cache" / tag / f"stock_{stock_id}.parquet"


def _build_one(args) -> pd.DataFrame:
    """Worker: features for one stock, cached to parquet.

    Returns a single frame (index columns + target + features) because
    process pools have to pickle whatever comes back, and one frame is
    cheaper to ship than a dataclass of three.
    """
    stock_id, spec, root, targets = args
    path = cache_path(stock_id, spec, root)
    if path.exists():
        return pd.read_parquet(path)

    block = features.build_stock_features(stock_id, targets, root=root, spec=spec)
    frame = pd.concat([block.index, block.target.rename("target"), block.features], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def build_table(
    stocks: list[int],
    targets: pd.DataFrame,
    spec: features.SignatureSpec,
    root: Path,
    jobs: int = 1,
) -> pd.DataFrame:
    """Feature table for all selected stocks, one row per (stock_id, time_id)."""
    work = [(stock_id, spec, root, targets[targets["stock_id"] == stock_id]) for stock_id in stocks]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            frames = list(pool.map(_build_one, work))
    else:
        frames = [_build_one(item) for item in work]
    return pd.concat(frames, ignore_index=True)


def score_arms(table: pd.DataFrame, arms: list[str], n_splits: int = 5) -> dict:
    """Run every arm plus the naive predictor on identical folds."""
    y = table["target"].to_numpy()
    groups = table["time_id"].to_numpy()
    feature_columns = table.drop(columns=["stock_id", "time_id", "target"])
    # stock_id is a legitimate feature -- it carries each stock's baseline
    # volatility level -- and it goes to every arm equally so it can't
    # advantage one.
    feature_columns = feature_columns.assign(stock_id=table["stock_id"].to_numpy())

    naive = features.naive_prediction(table)
    results = {
        "naive": {
            "arm": "naive (RV of observed window)",
            "n_features": 1,
            "rmspe": evaluate.rmspe(y, naive),
            "fold_std": 0.0,
        }
    }
    predictions = {"naive": naive}

    for arm in arms:
        columns = [c for c in feature_columns.columns if c.startswith(features.ARM_PREFIXES[arm])]
        X = feature_columns[columns + ["stock_id"]]
        started = time.perf_counter()
        result = evaluate.run_arm(arm, X, y, groups, n_splits=n_splits)
        summary = result.summary()
        summary["fit_seconds"] = round(time.perf_counter() - started, 1)
        results[arm] = summary
        predictions[arm] = result.predictions
        print(f"  {arm:>14s}  {result.n_features:4d} features  RMSPE {result.oof_rmspe:.5f}")

    comparisons = {}
    for baseline, challenger in [
        ("naive", "book+har"),
        ("har", "sig"),
        ("book", "sig+book"),
        ("book+har", "sig+book+har"),
    ]:
        if baseline in predictions and challenger in predictions:
            comparisons[f"{challenger} vs {baseline}"] = evaluate.paired_bootstrap(
                y, predictions[baseline], predictions[challenger], groups
            )

    return {"arms": results, "comparisons": comparisons}


def markdown_table(results: dict) -> str:
    rows = ["| Arm | Features | RMSPE | Fold std |", "| --- | ---: | ---: | ---: |"]
    for entry in results["arms"].values():
        rows.append(
            f"| {entry['arm']} | {entry['n_features']} | {entry['rmspe']:.5f} | {entry['fold_std']:.5f} |"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stocks", type=int, default=20, help="number of stocks in the subset")
    parser.add_argument("--seed", type=int, default=0, help="seed for the stock subset")
    parser.add_argument("--depth", type=int, default=3, help="signature truncation depth")
    parser.add_argument("--subsample", type=int, default=1, help="take every Nth second of the grid")
    parser.add_argument("--splits", type=int, default=5, help="number of grouped CV folds")
    parser.add_argument("--jobs", type=int, default=1, help="parallel feature-building workers")
    parser.add_argument(
        "--arms",
        nargs="+",
        default=ARMS,
        choices=ARMS,
        help="subset of arms to score (features are cached, so reruns are cheap)",
    )
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "benchmark.json")
    args = parser.parse_args()

    root = data.data_dir()
    if not data.is_available(root):
        raise SystemExit(
            f"no ORVP data at {root}. Run `python -m benchmarks.orvp.download`, "
            f"or point {data._ENV_VAR} at an existing copy."
        )

    targets = data.load_targets(root)
    stocks = data.select_stocks(targets, args.stocks, seed=args.seed)
    spec = features.SignatureSpec(depth=args.depth, subsample=args.subsample)
    print(f"stocks: {stocks}")

    started = time.perf_counter()
    table = build_table(stocks, targets, spec, root, jobs=args.jobs)
    build_seconds = time.perf_counter() - started
    print(f"feature table: {table.shape[0]} segments x {table.shape[1] - 3} features "
          f"in {build_seconds:.0f}s")

    results = score_arms(table, args.arms, n_splits=args.splits)
    results["config"] = {
        "stocks": stocks,
        "n_segments": int(table.shape[0]),
        "depth": args.depth,
        "subsample": args.subsample,
        "signature_windows": list(spec.windows),
        "n_splits": args.splits,
        "build_seconds": round(build_seconds, 1),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    (args.out.parent / "benchmark_table.md").write_text(markdown_table(results) + "\n")
    print(f"\nwrote {args.out}\n")
    print(markdown_table(results))
    for name, comparison in results["comparisons"].items():
        print(
            f"{name}: {comparison['improvement_pct']:+.2f}% RMSPE "
            f"(95% CI on absolute delta [{comparison['ci_low']:+.5f}, {comparison['ci_high']:+.5f}], "
            f"p(no improvement)={comparison['p_no_improvement']:.3f})"
        )


if __name__ == "__main__":
    main()
