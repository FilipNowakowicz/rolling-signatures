"""Measure the v0.3 streaming engine: cost, crossover, and drift.

    python -m benchmarks.streaming.bench

Three questions, each of which can come back against the engine:

1. **Is the per-tick cost really independent of the window?** That is the
   whole claim -- O(1) amortized instead of O(window) -- and it is the one
   thing here that is a theorem, so a measured slope in `window` would mean a
   bug rather than a disappointment.
2. **Does that beat recomputing?** Against the pure-numpy reference, trivially.
   Against `iisignature`'s compiled recompute it is a genuine race, because the
   streaming update is interpreted Python doing dozens of small numpy calls
   while the batch route is one C call over a longer path. There is a crossover
   window and this finds it, rather than quoting the asymptotics and stopping.
3. **What does it cost in accuracy?** Repeated group operations accumulate
   floating-point error that a recomputation never incurs
   (`docs/notes-orvp.html` s9.1 flagged this before the code existed). Measured
   against from-scratch signatures, by depth, by level, and with re-anchoring
   on and off.

Everything runs on synthetic paths -- a random walk with a configurable
increment scale -- because all three questions are about the arithmetic, not
about any dataset. The one place real data would matter, the nested-window
pattern `benchmarks/orvp` actually uses, is study 4, and it is measured on the
same 600/300/150-second geometry at the same channel count.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from sigtrade import signature
from sigtrade.preprocessing import preprocess_path
from sigtrade.streaming import StreamingSignature, nested_suffix_signatures

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# One channel, time-augmented and lead-lagged: the shape `benchmarks/orvp`
# feeds its signature arm, and the smallest configuration that exercises both
# preprocessing paths the streaming encoding has to reproduce.
DIM = 1
PREPROCESS = {"time_augmentation": True, "lead_lag_transform": True}

WINDOWS = (10, 30, 100, 300, 600, 1200)
DEPTHS = (2, 3)
DRIFT_DEPTHS = (2, 3, 4, 5)
DRIFT_WINDOW = 250
DRIFT_TICKS = 20_000
ORVP_WINDOWS = (600, 300, 150)


def random_walk(n: int, scale: float = 1e-3, seed: int = 0) -> np.ndarray:
    """A log-price-like path: `scale` is the per-tick return standard deviation."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, DIM)) * scale).cumsum(axis=0)


def _time_ticks(step, max_ticks: int, budget: float) -> tuple[float, int]:
    """Microseconds per tick, running until `budget` seconds or `max_ticks`.

    Batch recomputation at window 1200 is four orders of magnitude slower per
    tick than a streaming update, so a fixed tick count would either take an
    hour or measure nothing. The budget lets every cell cost the same wall
    clock and report the count it managed.
    """
    started = time.perf_counter()
    ticks = 0
    while ticks < max_ticks:
        step(ticks)
        ticks += 1
        if time.perf_counter() - started > budget:
            break
    return (time.perf_counter() - started) / ticks * 1e6, ticks


def _batch_step(X: np.ndarray, window: int, depth: int, backend: str, offset: int):
    """One tick of the honest batch alternative: preprocess, then recompute."""

    def step(tick: int) -> None:
        end = offset + tick + 1
        path = preprocess_path(X[max(0, end - window) : end], **PREPROCESS)
        signature(path, depth, backend=backend)

    return step


def cost_by_window(backends: tuple[str, ...], budget: float, seed: int = 0) -> list[dict]:
    """Study 1 + 2: per-tick cost against window length, streaming vs recompute."""
    rows = []
    for depth in DEPTHS:
        for window in WINDOWS:
            max_ticks = 3000
            X = random_walk(window + max_ticks + 2, seed=seed)
            row = {"depth": depth, "window": window}

            for label, refresh in (("streaming", None), ("streaming_refresh", "auto")):
                engine = StreamingSignature(window, depth, DIM, refresh_every=refresh, **PREPROCESS)
                for point in X[: window + 1]:  # warm past the fill: measure the steady state
                    engine.update(point)
                offset = window + 1
                micros, ticks = _time_ticks(
                    lambda tick: engine.update(X[offset + tick]), max_ticks, budget
                )
                row[label] = round(micros, 2)
                row[f"{label}_ticks"] = ticks

            for backend in backends:
                micros, ticks = _time_ticks(
                    _batch_step(X, window, depth, backend, window + 1), max_ticks, budget
                )
                row[f"batch_{backend}"] = round(micros, 2)
                row[f"batch_{backend}_ticks"] = ticks

            rows.append(row)
            print(f"  depth={depth} window={window:5d}  " + "  ".join(
                f"{k}={row[k]:.1f}us" for k in row if not k.endswith("_ticks") and k not in ("depth", "window")
            ))
    return rows


def drift(
    depth: int,
    window: int = DRIFT_WINDOW,
    ticks: int = DRIFT_TICKS,
    scale: float = 1e-3,
    refresh_every="auto",
    checkpoint: int = 250,
    seed: int = 0,
) -> dict:
    """Study 3: worst relative disagreement with a from-scratch signature.

    Reported as an infinity-norm ratio, overall and per level, because the
    absolute error is meaningless on its own: a path with large total variation
    has a large signature, and both grow together.
    """
    X = random_walk(ticks, scale=scale, seed=seed)
    engine = StreamingSignature(window, depth, DIM, refresh_every=refresh_every, **PREPROCESS)
    channels = engine.n_channels
    sizes = [channels**level for level in range(1, depth + 1)]

    worst_overall = 0.0
    worst_by_level = [0.0] * depth
    for tick, point in enumerate(X):
        streamed = engine.update(point)
        if tick < window or tick % checkpoint:
            continue
        exact = signature(preprocess_path(X[tick + 1 - window : tick + 1], **PREPROCESS), depth, backend="numpy")
        worst_overall = max(worst_overall, _relative(streamed, exact))
        offset = 0
        for level, size in enumerate(sizes):
            piece = slice(offset, offset + size)
            worst_by_level[level] = max(worst_by_level[level], _relative(streamed[piece], exact[piece]))
            offset += size

    return {
        "depth": depth,
        "window": window,
        "ticks": ticks,
        "scale": scale,
        "refresh_every": refresh_every if refresh_every != "auto" else window,
        "relative_error": worst_overall,
        "relative_error_by_level": worst_by_level,
        # Time augmentation renormalises its channel on every partial window,
        # so the fill costs one recompute per tick whatever `refresh_every`
        # says. Only the excess is re-anchoring.
        "warmup_recomputes": window - 1,
        "steady_recomputes": engine.n_recomputes_ - (window - 1),
    }


def _relative(streamed: np.ndarray, exact: np.ndarray) -> float:
    scale = float(np.max(np.abs(exact)))
    return float(np.max(np.abs(streamed - exact)) / scale) if scale else 0.0


def drift_study(seed: int = 0) -> dict:
    """Drift by depth, by refresh setting, by tick count, and by path scale."""
    by_depth = []
    for depth in DRIFT_DEPTHS:
        for refresh in (None, "auto"):
            print(f"  drift depth={depth} refresh={refresh}")
            by_depth.append(drift(depth, refresh_every=refresh, seed=seed))

    by_ticks = []
    for ticks in (2_500, 5_000, 10_000, 20_000, 40_000):
        print(f"  drift growth ticks={ticks}")
        by_ticks.append(drift(4, ticks=ticks, refresh_every=None, seed=seed))

    by_scale = []
    for scale in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        print(f"  drift scale={scale}")
        by_scale.append(drift(4, scale=scale, refresh_every=None, seed=seed))

    return {"by_depth": by_depth, "by_ticks": by_ticks, "by_scale": by_scale}


def nested_windows(depth: int = 3, segments: int = 200, seed: int = 0) -> dict:
    """Study 4: the nested-suffix pattern `benchmarks/orvp` pays for three times.

    Each ORVP segment needs signatures over its last 600, 300 and 150 seconds.
    The naive route walks 1050 points of path; splitting the longest window at
    the shorter ones' boundaries and combining with Chen's identity walks 600.
    """
    rng = np.random.default_rng(seed)
    paths = [(rng.standard_normal((ORVP_WINDOWS[0], DIM)) * 1e-3).cumsum(axis=0) for _ in range(segments)]

    def naive(path, backend):
        return {
            length: signature(preprocess_path(path[-length:], **PREPROCESS), depth, backend=backend)
            for length in ORVP_WINDOWS
        }

    results = {"depth": depth, "segments": segments, "windows": list(ORVP_WINDOWS)}
    started = time.perf_counter()
    shared = [nested_suffix_signatures(path, ORVP_WINDOWS, depth, **PREPROCESS) for path in paths]
    results["chen_ms"] = (time.perf_counter() - started) / segments * 1e3

    for backend in ("numpy", "iisignature"):
        try:
            started = time.perf_counter()
            reference = [naive(path, backend) for path in paths]
            results[f"naive_{backend}_ms"] = (time.perf_counter() - started) / segments * 1e3
        except ImportError:
            continue
        results[f"max_abs_difference_vs_{backend}"] = max(
            float(np.max(np.abs(a[length] - b[length])))
            for a, b in zip(shared, reference)
            for length in ORVP_WINDOWS
        )
    results["speedup_vs_numpy"] = results["naive_numpy_ms"] / results["chen_ms"]
    return results


def crossover(rows: list[dict], reference: str) -> dict[int, int | None]:
    """Smallest window at which streaming beats a given batch arm, per depth."""
    out: dict[int, int | None] = {}
    for depth in DEPTHS:
        beat = [
            row["window"]
            for row in rows
            if row["depth"] == depth
            and f"batch_{reference}" in row
            and row["streaming"] < row[f"batch_{reference}"]
        ]
        out[depth] = min(beat) if beat else None
    return out


def markdown_table(results: dict) -> str:
    rows = results["cost_by_window"]
    backends = [key[len("batch_") :] for key in rows[0] if key.startswith("batch_") and not key.endswith("_ticks")]

    lines = ["### Per-tick cost (microseconds), 1 channel + time augmentation + lead-lag", ""]
    header = ["depth", "window", "streaming", "streaming (auto refresh)"]
    header += [f"batch `{backend}`" for backend in backends]
    header += ["x vs numpy"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |")
    for row in rows:
        cells = [
            str(row["depth"]),
            str(row["window"]),
            f"{row['streaming']:.1f}",
            f"{row['streaming_refresh']:.1f}",
        ]
        cells += [f"{row[f'batch_{backend}']:.1f}" for backend in backends]
        cells += [f"{row['batch_numpy'] / row['streaming']:.0f}x"]
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "### Drift vs a from-scratch signature (relative, infinity norm)", ""]
    lines.append("| depth | refresh every | relative error | worst level | re-anchors |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in results["drift"]["by_depth"]:
        worst_level = int(np.argmax(row["relative_error_by_level"])) + 1
        lines.append(
            f"| {row['depth']} | {row['refresh_every'] or 'never'} | {row['relative_error']:.2e} "
            f"| {max(row['relative_error_by_level']):.2e} (level {worst_level}) | {row['steady_recomputes']} |"
        )

    nested = results["nested_windows"]
    lines += ["", "### Nested 600/300/150 windows, per ORVP segment (milliseconds)", ""]
    lines.append("| route | ms/segment |")
    lines.append("| --- | ---: |")
    lines.append(f"| naive: three independent signatures (numpy) | {nested['naive_numpy_ms']:.2f} |")
    if "naive_iisignature_ms" in nested:
        lines.append(
            f"| naive: three independent signatures (iisignature) | {nested['naive_iisignature_ms']:.2f} |"
        )
    lines.append(f"| Chen: disjoint chunks combined | {nested['chen_ms']:.2f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--budget", type=float, default=1.0, help="seconds per timed cell")
    parser.add_argument("--segments", type=int, default=200, help="synthetic segments for the nested-window study")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-roughpy", action="store_true", help="omit the (very slow) default backend")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "streaming.json")
    args = parser.parse_args()

    backends = ["numpy"]
    for name in ("iisignature", "roughpy"):
        if name == "roughpy" and args.skip_roughpy:
            continue
        try:
            __import__(name)
        except ImportError:
            print(f"skipping backend {name}: not installed")
            continue
        backends.append(name)

    print("study 1+2: per-tick cost vs window")
    rows = cost_by_window(tuple(backends), budget=args.budget, seed=args.seed)
    print("study 3: numerical drift")
    drift_results = drift_study(seed=args.seed)
    print("study 4: nested suffix windows")
    nested = nested_windows(segments=args.segments, seed=args.seed)

    results = {
        "config": {
            "dim": DIM,
            "preprocessing": PREPROCESS,
            "windows": list(WINDOWS),
            "depths": list(DEPTHS),
            "backends": backends,
            "budget_seconds": args.budget,
            "seed": args.seed,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "cost_by_window": rows,
        "crossover_window": {
            backend: crossover(rows, backend) for backend in backends if backend != "numpy"
        },
        "drift": drift_results,
        "nested_windows": nested,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    (args.out.parent / "streaming_table.md").write_text(markdown_table(results) + "\n")
    print(f"\nwrote {args.out}\n")
    print(markdown_table(results))


if __name__ == "__main__":
    main()
