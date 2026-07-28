"""Truncation-depth x window-length study.

    python -m benchmarks.orvp.study --stocks 20 --jobs 8

Signature feature count grows fast in the truncation depth: the free Lie
algebra on `d` generators has roughly `d^n / n` independent brackets at
level `n`, so a depth-4 signature of a lead-lag, time-augmented univariate
path already carries 90 coordinates per window against depth 3's 30. More
coordinates is not more information about the *target* -- past some depth
the extra terms describe finer path detail than a 10-minute volatility
forecast can use, and the model starts fitting them as noise.

This finds where that turnover happens on the real task rather than in the
abstract, and does it on the same folds and the same learner as the main
benchmark so the numbers are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from benchmarks.orvp import data, evaluate, features
from benchmarks.orvp.run import RESULTS_DIR, build_table

DEPTHS = (2, 3, 4)
WINDOW_SETS = {
    "600": (600,),
    "600-300-150": (600, 300, 150),
    "600-300-150-60": (600, 300, 150, 60),
}


def run_study(
    stocks: list[int],
    targets: pd.DataFrame,
    root: Path,
    depths: tuple[int, ...] = DEPTHS,
    window_sets: dict[str, tuple[int, ...]] = WINDOW_SETS,
    subsample: int = 1,
    n_splits: int = 5,
    jobs: int = 1,
) -> list[dict]:
    rows = []
    for depth in depths:
        for name, windows in window_sets.items():
            spec = features.SignatureSpec(depth=depth, windows=windows, subsample=subsample)
            started = time.perf_counter()
            table = build_table(stocks, targets, spec, root, jobs=jobs)
            build_seconds = time.perf_counter() - started

            y = table["target"].to_numpy()
            groups = table["time_id"].to_numpy()
            columns = [c for c in table.columns if c.startswith("sig_")]
            X = table[columns].assign(stock_id=table["stock_id"].to_numpy())

            result = evaluate.run_arm("sig", X, y, groups, n_splits=n_splits)
            rows.append(
                {
                    "depth": depth,
                    "windows": name,
                    "n_features": result.n_features,
                    "rmspe": result.oof_rmspe,
                    "fold_std": float(pd.Series(result.fold_rmspe).std()),
                    "build_seconds": round(build_seconds, 1),
                }
            )
            print(
                f"  depth={depth} windows={name:>15s} "
                f"{result.n_features:4d} features  RMSPE {result.oof_rmspe:.5f} "
                f"({build_seconds:.0f}s to build)"
            )
    return rows


def markdown_table(rows: list[dict]) -> str:
    out = [
        "| Depth | Windows (s) | Features | RMSPE | Fold std | Build (s) |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        out.append(
            f"| {row['depth']} | {row['windows']} | {row['n_features']} | "
            f"{row['rmspe']:.5f} | {row['fold_std']:.5f} | {row['build_seconds']:.0f} |"
        )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stocks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "depth_window_study.json")
    args = parser.parse_args()

    root = data.data_dir()
    if not data.is_available(root):
        raise SystemExit(f"no ORVP data at {root}. Run `python -m benchmarks.orvp.download` first.")

    targets = data.load_targets(root)
    stocks = data.select_stocks(targets, args.stocks, seed=args.seed)
    rows = run_study(
        stocks, targets, root, subsample=args.subsample, n_splits=args.splits, jobs=args.jobs
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"config": {"stocks": stocks}, "rows": rows}, indent=2, default=float))
    (args.out.parent / "depth_window_table.md").write_text(markdown_table(rows) + "\n")
    print(f"\nwrote {args.out}\n")
    print(markdown_table(rows))


if __name__ == "__main__":
    main()
