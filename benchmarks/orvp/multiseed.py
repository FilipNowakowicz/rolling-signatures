"""Repeat the whole benchmark on several 20-stock subsets.

    python -m benchmarks.orvp.multiseed --seeds 0 1 2 --stocks 20 --jobs 8

The single-seed run in `run.py` scores every arm on one random subset of
stocks. A difference of a fraction of a percent in RMSPE between two arms is
small enough that it could plausibly be a property of *those twenty stocks*
rather than of the feature sets, and the grouped bootstrap cannot detect
that: it resamples time_ids, so it prices in market-instant variation but
holds the stock universe fixed.

Repeating the run on disjointly-drawn subsets is what prices in the stock
universe. The seeds are fixed in `SEEDS` and chosen before any result is
seen, so this is a pre-registered replication and not a search over subsets
for a favourable one.

Writes `results/multiseed.json` and a markdown summary.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from benchmarks.orvp import data, features
from benchmarks.orvp.run import ARMS, RESULTS_DIR, build_table, score_arms

SEEDS = (0, 1, 2)
"""The pre-registered stock subsets, fixed before any arm was scored."""

HEADLINE = "multisig+book+har vs book+har"
"""The comparison v0.2.1 exists to settle.

`book+har` is the best reproduced baseline. If multichannel signatures do
not improve on it, adding channels has not rescued the negative v0.2 result
and the ORVP search should stop.
"""

CONSISTENCY_ALPHA = 0.05
"""A seed counts as a win only if the grouped bootstrap clears this."""


def run_seeds(
    seeds: tuple[int, ...],
    n_stocks: int,
    root: Path,
    spec: features.SignatureSpec,
    multi_spec: features.MultiSignatureSpec,
    arms: list[str],
    n_splits: int = 5,
    jobs: int = 1,
    predictions_dir: Path | None = None,
) -> dict:
    """Score every arm once per seed, keeping the learner and folds fixed."""
    targets = data.load_targets(root)
    per_seed = {}
    for seed in seeds:
        stocks = data.select_stocks(targets, n_stocks, seed=seed)
        missing = [s for s in stocks if not (root / "book_train.parquet" / f"stock_id={s}").is_dir()]
        if missing:
            raise SystemExit(
                f"seed {seed} needs stocks {missing}, which are not downloaded. Run\n"
                f"  python -m benchmarks.orvp.download --stocks {n_stocks} --seed {seed}"
            )

        print(f"\n=== seed {seed}: {len(stocks)} stocks ===")
        started = time.perf_counter()
        table = build_table(stocks, targets, spec, root, jobs=jobs, multi_spec=multi_spec)
        print(
            f"feature table: {table.shape[0]} segments x {table.shape[1] - 3} features "
            f"in {time.perf_counter() - started:.0f}s"
        )
        predictions_out = None
        if predictions_dir is not None:
            predictions_out = predictions_dir / f"oof_predictions_seed{seed}.npz"
        results = score_arms(table, arms, n_splits=n_splits, predictions_out=predictions_out)
        results["stocks"] = stocks
        results["n_segments"] = int(table.shape[0])
        per_seed[str(seed)] = results
    return per_seed


def summarise(per_seed: dict, arms: list[str]) -> dict:
    """Collapse the per-seed runs into per-arm and per-comparison summaries.

    Reported as mean and *range* across seeds rather than a standard error:
    three seeds is far too few for a standard error to mean anything, and
    the range is the honest statement of how much the number moves.
    """
    seeds = list(per_seed)
    arm_rows = []
    for arm in ["naive"] + arms:
        scores = [per_seed[s]["arms"][arm]["rmspe"] for s in seeds if arm in per_seed[s]["arms"]]
        if not scores:
            continue
        arm_rows.append(
            {
                "arm": arm,
                "n_features": per_seed[seeds[0]]["arms"][arm]["n_features"],
                "per_seed_rmspe": scores,
                "mean_rmspe": float(np.mean(scores)),
                "min_rmspe": float(np.min(scores)),
                "max_rmspe": float(np.max(scores)),
            }
        )

    comparison_names = list(per_seed[seeds[0]]["comparisons"])
    comparison_rows = []
    for name in comparison_names:
        entries = [per_seed[s]["comparisons"][name] for s in seeds if name in per_seed[s]["comparisons"]]
        if not entries:
            # `all([])` is True, so an unscored comparison would otherwise
            # report itself as consistently winning on zero evidence.
            continue
        improvements = [e["improvement_pct"] for e in entries]
        wins = [e["improvement"] > 0 and e["p_no_improvement"] < CONSISTENCY_ALPHA for e in entries]
        comparison_rows.append(
            {
                "comparison": name,
                "per_seed_improvement_pct": improvements,
                "mean_improvement_pct": float(np.mean(improvements)),
                "seeds_improved": int(sum(i > 0 for i in improvements)),
                "seeds_significant": int(sum(wins)),
                "n_seeds": len(entries),
                # "Consistent" means every seed, not most of them. A feature
                # family that helps on two subsets out of three has not
                # earned a place in a pipeline.
                "consistent": bool(all(wins)),
            }
        )

    headline = next((row for row in comparison_rows if row["comparison"] == HEADLINE), None)
    return {
        "arms": arm_rows,
        "comparisons": comparison_rows,
        "headline": headline,
        "verdict": _verdict(headline),
    }


def _verdict(headline: dict | None) -> str:
    """The stop/continue call, stated in the artefact rather than inferred."""
    if headline is None:
        return "inconclusive: the headline comparison was not scored"
    if headline["consistent"]:
        return (
            f"multichannel signatures improve {HEADLINE.split(' vs ')[1]} on all "
            f"{headline['n_seeds']} seeds -- continue the ORVP search"
        )
    return (
        f"multichannel signatures do not consistently improve "
        f"{HEADLINE.split(' vs ')[1]} ({headline['seeds_significant']}/{headline['n_seeds']} "
        f"seeds significant, mean {headline['mean_improvement_pct']:+.2f}%) -- stop the ORVP search"
    )


def markdown_table(summary: dict, seeds: tuple[int, ...]) -> str:
    seed_columns = " | ".join(f"seed {s}" for s in seeds)
    lines = [
        f"| Arm | Features | {seed_columns} | Mean |",
        "| --- | ---: |" + " ---: |" * (len(seeds) + 1),
    ]
    for row in summary["arms"]:
        scores = " | ".join(f"{v:.5f}" for v in row["per_seed_rmspe"])
        lines.append(f"| {row['arm']} | {row['n_features']} | {scores} | {row['mean_rmspe']:.5f} |")

    lines += [
        "",
        f"| Comparison | {seed_columns} | Mean | Seeds significant |",
        "| --- |" + " ---: |" * (len(seeds) + 2),
    ]
    for row in summary["comparisons"]:
        deltas = " | ".join(f"{v:+.2f}%" for v in row["per_seed_improvement_pct"])
        lines.append(
            f"| {row['comparison']} | {deltas} | {row['mean_improvement_pct']:+.2f}% | "
            f"{row['seeds_significant']}/{row['n_seeds']} |"
        )
    lines += ["", f"**Verdict:** {summary['verdict']}."]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--stocks", type=int, default=20)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--multi-depth", type=int, default=2)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "multiseed.json")
    args = parser.parse_args()

    root = data.data_dir()
    if not data.is_available(root):
        raise SystemExit(f"no ORVP data at {root}. Run `python -m benchmarks.orvp.download` first.")

    spec = features.SignatureSpec(depth=args.depth, subsample=args.subsample)
    multi_spec = features.MultiSignatureSpec(depth=args.multi_depth, subsample=args.subsample)
    seeds = tuple(args.seeds)

    per_seed = run_seeds(
        seeds,
        args.stocks,
        root,
        spec,
        multi_spec,
        args.arms,
        n_splits=args.splits,
        jobs=args.jobs,
        predictions_dir=args.out.parent,
    )
    summary = summarise(per_seed, args.arms)

    payload = {
        "config": {
            "seeds": list(seeds),
            "n_stocks": args.stocks,
            "depth": args.depth,
            "signature_windows": list(spec.windows),
            "multi_depth": multi_spec.depth,
            "multi_windows": list(multi_spec.windows),
            "multi_channels": list(multi_spec.channels),
            "n_splits": args.splits,
            "consistency_alpha": CONSISTENCY_ALPHA,
        },
        "summary": summary,
        "per_seed": per_seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=float))
    table = markdown_table(summary, seeds)
    (args.out.parent / "multiseed_table.md").write_text(table + "\n")
    print(f"\nwrote {args.out}\n")
    print(table)


if __name__ == "__main__":
    main()
