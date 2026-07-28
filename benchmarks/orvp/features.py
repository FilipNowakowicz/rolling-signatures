"""Feature sets for every arm of the benchmark.

One pass over a stock's book and trade files produces the columns for all
arms at once, prefixed by arm (`har_`, `book_`, `sig_`). Reading the parquet
dominates the cost, so computing every arm together -- rather than once per
arm -- keeps the arms genuinely comparable (identical rows, identical row
order) and the run cheap.

Every feature is a function of the observed 10-minute segment only. The
target is the realized volatility of the *following* 10 minutes and is never
touched here, so no feature can see it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.orvp import data
from sigtrade import log_signature, preprocess_path
from sigtrade._backend import _log_labels_iisignature

SUFFIX_WINDOWS = (600, 300, 150, 60, 30)
"""Nested suffix windows, in seconds, for the HAR-style arm.

HAR-RV's premise is that volatility is driven by components acting over
different horizons. Here every window ends at the segment's close, so each
is available causally at prediction time.
"""

BOOK_WINDOWS = (600, 300, 150)
"""Suffix windows for the book/trade aggregate arm."""

SIGNATURE_WINDOWS = (600, 300, 150)
"""Suffix windows over which log-signatures are taken."""


@dataclass(frozen=True)
class SignatureSpec:
    """How the signature arm turns a segment into features."""

    depth: int = 3
    windows: tuple[int, ...] = SIGNATURE_WINDOWS
    time_augmentation: bool = True
    lead_lag_transform: bool = True
    backend: str = "iisignature"
    subsample: int = 1
    """Take every `subsample`-th second of the grid before computing.

    Signature cost is linear in the number of points, and reparametrisation
    invariance means dropping points only loses the price moves that
    happened between them -- never distorts the ones that remain. Used by
    the depth study to keep deep truncations affordable.
    """


@dataclass
class StockFeatures:
    """Feature block for one stock: index columns, features, and the target."""

    index: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def har_features(log_returns: np.ndarray, windows: tuple[int, ...] = SUFFIX_WINDOWS) -> pd.DataFrame:
    """Multi-horizon realized volatility, the HAR-RV analogue for one segment.

    Also carries the count of seconds that actually moved: a segment whose
    volatility came from three big jumps behaves differently from one that
    drifted continuously, and RV alone cannot tell them apart.
    """
    out = {}
    for window in windows:
        suffix = log_returns[:, -window:]
        rv = data.realized_volatility(suffix)
        out[f"har_rv_{window}"] = rv
        out[f"har_log_rv_{window}"] = np.log(np.maximum(rv, 1e-12))
        out[f"har_absmean_{window}"] = np.abs(suffix).mean(axis=1)
        out[f"har_maxabs_{window}"] = np.abs(suffix).max(axis=1)
        out[f"har_active_{window}"] = (suffix != 0).sum(axis=1) / window
    # Realized quarticity: the fourth-moment analogue, which governs how
    # noisy RV itself is as an estimator.
    out["har_quarticity"] = np.power(np.square(np.square(log_returns)).sum(axis=1), 0.25)
    return pd.DataFrame(out)


def book_features(
    book: pd.DataFrame,
    trade: pd.DataFrame,
    time_ids: np.ndarray,
    windows: tuple[int, ...] = BOOK_WINDOWS,
) -> pd.DataFrame:
    """Reproduced top-solution-style order-book and trade aggregates.

    This is the public-kernel feature family the competition's leading
    solutions were built on: WAP realized volatility at both book levels,
    relative spread, depth imbalance, and trade intensity, each aggregated
    over nested suffix windows. It is the arm the signature arm has to beat
    to mean anything.
    """
    book = book.copy()
    book["wap1"] = data.wap(book, level=1)
    book["wap2"] = data.wap(book, level=2)
    book["spread"] = (book["ask_price1"] - book["bid_price1"]) / (
        (book["ask_price1"] + book["bid_price1"]) / 2
    )
    book["depth"] = book["bid_size1"] + book["ask_size1"] + book["bid_size2"] + book["ask_size2"]
    book["imbalance"] = (book["bid_size1"] + book["bid_size2"] - book["ask_size1"] - book["ask_size2"]) / book[
        "depth"
    ]
    for level in (1, 2):
        grouped = book.groupby("time_id")[f"wap{level}"]
        book[f"logret{level}"] = np.log(book[f"wap{level}"] / grouped.shift(1)).fillna(0.0)

    frames = []
    for window in windows:
        start = data.SECONDS_PER_SEGMENT - window
        chunk = book[book["seconds_in_bucket"] >= start]
        grouped = chunk.groupby("time_id")
        agg = pd.DataFrame(
            {
                f"book_rv1_{window}": np.sqrt(grouped["logret1"].apply(lambda r: np.square(r).sum())),
                f"book_rv2_{window}": np.sqrt(grouped["logret2"].apply(lambda r: np.square(r).sum())),
                f"book_spread_mean_{window}": grouped["spread"].mean(),
                f"book_spread_max_{window}": grouped["spread"].max(),
                f"book_depth_mean_{window}": grouped["depth"].mean(),
                f"book_imbalance_mean_{window}": grouped["imbalance"].mean(),
                f"book_imbalance_std_{window}": grouped["imbalance"].std(),
                f"book_updates_{window}": grouped.size(),
            }
        )
        frames.append(agg)

        trade_chunk = trade[trade["seconds_in_bucket"] >= start]
        trade_grouped = trade_chunk.groupby("time_id")
        trade_agg = pd.DataFrame(
            {
                f"book_trade_size_{window}": trade_grouped["size"].sum(),
                f"book_trade_count_{window}": trade_grouped.size(),
                f"book_trade_orders_{window}": trade_grouped["order_count"].sum(),
                f"book_trade_price_std_{window}": trade_grouped["price"].std(),
            }
        )
        frames.append(trade_agg)

    out = pd.concat(frames, axis=1).reindex(time_ids)
    # A quiet window is not a missing one. No trades means zero traded
    # volume, and no book updates means the price did not move, so realized
    # volatility over that window is genuinely zero -- filling these with 0
    # is the truth, not an imputation.
    zero_filled = [c for c in out.columns if "_trade_" in c or "_rv" in c or "_updates_" in c]
    out[zero_filled] = out[zero_filled].fillna(0.0)
    # Spread, depth and imbalance are left NaN when the window has no
    # updates: the standing quote is unknown from this window alone, and
    # the gradient booster handles missing values natively rather than
    # being fed a made-up number.
    return out.reset_index(drop=True)


def signature_feature_names(spec: SignatureSpec, n_channels: int = 1) -> list[str]:
    """Column names for `signature_features`, one per Lie-basis coordinate.

    Names carry the backend's own bracket labels, because the two backends
    use different bases of the free Lie algebra (see `sigtrade.log_signature`).
    """
    dim = n_channels + int(spec.time_augmentation)
    if spec.lead_lag_transform:
        dim *= 2
    labels = _log_labels_iisignature(dim, spec.depth)
    return [f"sig_w{window}_{label}" for window in spec.windows for label in labels]


def signature_features(grid: pd.DataFrame, spec: SignatureSpec = SignatureSpec()) -> pd.DataFrame:
    """Log-signatures of the WAP path over nested causal suffix windows.

    The path is the log-WAP recentred on each window's own first point --
    signatures see increments only, so recentring changes nothing
    mathematically, but it keeps the numbers small and comparable across
    stocks trading at different price levels.

    Taking a *set* of nested suffix windows rather than one gives the
    signature arm the same multi-horizon structure the HAR arm gets, so the
    comparison tests the feature family and not the horizon choice.

    Each window is computed from scratch here. Nested suffixes share almost
    all of their path, and Chen's identity would let the shorter windows be
    peeled off the longer ones instead -- that is exactly the redundancy the
    v0.3 streaming engine removes, and this benchmark is the thing that
    quantifies what it is worth.
    """
    log_prices = np.log(grid.to_numpy())
    blocks = []
    for window in spec.windows:
        suffix = log_prices[:, -window:][:, :: spec.subsample]
        rows = [
            log_signature(
                preprocess_path(
                    (segment - segment[0])[:, None],
                    time_augmentation=spec.time_augmentation,
                    lead_lag_transform=spec.lead_lag_transform,
                ),
                spec.depth,
                backend=spec.backend,
            )
            for segment in suffix
        ]
        blocks.append(np.asarray(rows))
    columns = signature_feature_names(spec)
    return pd.DataFrame(np.hstack(blocks), columns=columns)


def build_stock_features(
    stock_id: int,
    targets: pd.DataFrame,
    root: Path | None = None,
    spec: SignatureSpec = SignatureSpec(),
) -> StockFeatures:
    """All arms' features for one stock, aligned to `train.csv`'s rows."""
    book = data.load_book(stock_id, root)
    trade = data.load_trade(stock_id, root)
    grid = data.wap_grid(book)

    stock_targets = targets[targets["stock_id"] == stock_id].set_index("time_id")
    # Keep only segments that have both a target and book data, in a fixed
    # order, so every arm sees exactly the same rows.
    time_ids = np.array([t for t in grid.index if t in stock_targets.index])
    grid = grid.loc[time_ids]

    log_returns = data.log_return_grid(grid)
    features = pd.concat(
        [
            har_features(log_returns),
            book_features(book, trade, time_ids),
            signature_features(grid, spec),
        ],
        axis=1,
    )
    index = pd.DataFrame({"stock_id": stock_id, "time_id": time_ids})
    target = stock_targets.loc[time_ids, "target"].reset_index(drop=True)
    return StockFeatures(index=index, features=features, target=target)


def naive_prediction(features: pd.DataFrame) -> np.ndarray:
    """The zero-model benchmark: tomorrow's volatility equals today's.

    Realized volatility is strongly persistent, so this is a genuinely hard
    baseline -- it is what any feature set has to improve on before it has
    earned anything.
    """
    return features["har_rv_600"].to_numpy()


ARM_PREFIXES = {
    "har": ("har_",),
    "book": ("book_",),
    "book+har": ("book_", "har_"),
    "sig": ("sig_",),
    "sig+har": ("sig_", "har_"),
    "sig+book": ("sig_", "book_"),
    "sig+book+har": ("sig_", "book_", "har_"),
}


def select_arm(features: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Columns belonging to one arm."""
    prefixes = ARM_PREFIXES[arm]
    columns = [c for c in features.columns if c.startswith(prefixes)]
    return features[columns]
