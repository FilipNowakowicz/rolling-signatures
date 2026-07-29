"""Loading and reshaping the Optiver Realized Volatility Prediction data.

The competition ships one 10-minute order-book segment per
`(stock_id, time_id)` pair. Within a segment, `seconds_in_bucket` runs over
[0, 600) but rows appear only when the book *changed*, so the raw frame is
an irregularly sampled event stream.

Everything downstream wants a regular one-second grid instead, which
`second_grid` builds by forward-filling -- `wap_grid` for the price alone,
`channel_grids` for the several channels the multichannel signature arm
reads as coordinates of one path. That is not a modelling compromise:
forward-filling inserts no new geometry, it only repeats the current price
until the next update. Since the signature is invariant under
reparametrisation (docs/notes.html s5.1), the *price* path it sees is
literally unchanged; the only thing forward-filling alters is the time
channel added by `time_augment`, and there it makes the channel measure
real elapsed seconds rather than event count -- which is what we want.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

SECONDS_PER_SEGMENT = 600
"""Length of an ORVP segment. The competition's buckets are 10 minutes."""

_ENV_VAR = "ROLLSIG_ORVP_DIR"
_DEFAULT_DIR = Path("data/orvp")


def data_dir() -> Path:
    """Directory holding the extracted competition files.

    Override with the `ROLLSIG_ORVP_DIR` environment variable. The expected
    layout is the competition zip's own:

        <dir>/train.csv
        <dir>/book_train.parquet/stock_id=<n>/*.parquet
        <dir>/trade_train.parquet/stock_id=<n>/*.parquet
    """
    return Path(os.environ.get(_ENV_VAR, _DEFAULT_DIR))


def is_available(root: Path | None = None) -> bool:
    """Whether the real competition data is present and readable."""
    root = root or data_dir()
    return (root / "train.csv").exists() and (root / "book_train.parquet").is_dir()


def load_targets(root: Path | None = None) -> pd.DataFrame:
    """`train.csv`: one row per (stock_id, time_id) with the target.

    The target is the realized volatility of the *next* 10-minute window --
    it is never computable from the segment's own book data, which is what
    makes this a forecasting task rather than a measurement one.
    """
    root = root or data_dir()
    return pd.read_csv(root / "train.csv")


def select_stocks(targets: pd.DataFrame, n_stocks: int, seed: int = 0) -> list[int]:
    """A fixed random subset of stock ids, sorted.

    The benchmark runs on a subset because building rolling signatures for
    all 112 stocks is hours of compute. Drawing the subset from a seeded
    generator (rather than taking the first N ids) avoids any accidental
    ordering effect, and keeps runs comparable across arms.
    """
    available = np.sort(targets["stock_id"].unique())
    if n_stocks >= len(available):
        return [int(s) for s in available]
    rng = np.random.default_rng(seed)
    return sorted(int(s) for s in rng.choice(available, size=n_stocks, replace=False))


def load_book(stock_id: int, root: Path | None = None) -> pd.DataFrame:
    """Order-book snapshots for one stock, all time_ids."""
    root = root or data_dir()
    return pd.read_parquet(root / "book_train.parquet" / f"stock_id={stock_id}")


def load_trade(stock_id: int, root: Path | None = None) -> pd.DataFrame:
    """Executed trades for one stock, all time_ids."""
    root = root or data_dir()
    return pd.read_parquet(root / "trade_train.parquet" / f"stock_id={stock_id}")


def wap(book: pd.DataFrame, level: int = 1) -> pd.Series:
    """Weighted average price at one book level.

    The size-weighting is crossed -- the bid price is weighted by the *ask*
    size and vice versa -- so that a large resting bid pushes the WAP up
    toward the ask. This is the competition's own definition of the price
    whose realized volatility is the target.
    """
    bid_price = book[f"bid_price{level}"]
    ask_price = book[f"ask_price{level}"]
    bid_size = book[f"bid_size{level}"]
    ask_size = book[f"ask_size{level}"]
    return (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)


def relative_spread(book: pd.DataFrame) -> pd.Series:
    """Top-of-book spread as a fraction of the mid.

    Relative, not absolute, so the number is comparable across stocks
    trading at different price levels -- the same reason the target is a
    volatility of log returns rather than of prices.
    """
    bid = book["bid_price1"]
    ask = book["ask_price1"]
    return (ask - bid) / ((ask + bid) / 2)


def depth_imbalance(book: pd.DataFrame) -> pd.Series:
    """Signed share of resting size on the bid, over both book levels.

    Runs in [-1, 1]: +1 is an all-bid book, -1 an all-ask one. This is the
    standard order-flow-pressure proxy, and the channel most likely to carry
    information the price path alone does not.
    """
    bid = book["bid_size1"] + book["bid_size2"]
    ask = book["ask_size1"] + book["ask_size2"]
    # A book with no resting size on either side has no imbalance to speak
    # of; guarding the denominator keeps that case at 0 rather than NaN.
    return (bid - ask) / np.maximum(bid + ask, 1.0)


def second_grid(
    book: pd.DataFrame, values: pd.Series, n_seconds: int = SECONDS_PER_SEGMENT
) -> pd.DataFrame:
    """Forward-fill any per-snapshot series onto a regular one-second grid.

    Returns a frame indexed by `time_id` with columns `0 .. n_seconds-1`.
    Where a segment has several book updates inside the same second, the
    last one wins -- taking the last observation in a second is causal at
    that second's close, which is when the feature would be read.
    """
    frame = pd.DataFrame(
        {
            "time_id": book["time_id"].to_numpy(),
            "second": book["seconds_in_bucket"].to_numpy(),
            "value": np.asarray(values, dtype=float),
        }
    )
    grid = frame.pivot_table(index="time_id", columns="second", values="value", aggfunc="last")
    grid = grid.reindex(columns=range(n_seconds))
    # ffill carries the standing quote forward; bfill only fires for the rare
    # segment whose first update lands after second 0, and looks forward only
    # within the observed window, never into the target window.
    return grid.ffill(axis=1).bfill(axis=1)


def wap_grid(book: pd.DataFrame, level: int = 1, n_seconds: int = SECONDS_PER_SEGMENT) -> pd.DataFrame:
    """Forward-filled WAP on a regular one-second grid."""
    return second_grid(book, wap(book, level), n_seconds)


CHANNELS = ("log_wap", "spread", "imbalance")
"""Book-derived channels the multichannel signature arm reads.

Log-WAP is the price path the single-channel arm already sees. Spread and
imbalance are the two order-book state variables the reproduced
top-solution arm found most useful, and they are what a *path* signature
can say something new about: the cross terms at level 2 are the quadratic
covariations between price and book pressure, which no per-window average
of either channel can express.
"""


def channel_grids(
    book: pd.DataFrame, channels: tuple[str, ...] = CHANNELS, n_seconds: int = SECONDS_PER_SEGMENT
) -> dict[str, pd.DataFrame]:
    """One forward-filled one-second grid per named channel, sharing an index.

    Built in a single pass and returned together because every channel has
    to be sampled on the *same* grid: a multichannel signature reads the
    channels as coordinates of one path, so a row of one grid must describe
    the same second as the same row of another.
    """
    series = {
        "log_wap": lambda: np.log(wap(book, level=1)),
        "wap": lambda: wap(book, level=1),
        "spread": lambda: relative_spread(book),
        "imbalance": lambda: depth_imbalance(book),
    }
    unknown = [c for c in channels if c not in series]
    if unknown:
        raise ValueError(f"unknown channel(s) {unknown}; expected from {tuple(series)}")
    grids = {name: second_grid(book, series[name](), n_seconds) for name in channels}

    first = grids[channels[0]]
    for name, grid in grids.items():
        if not grid.index.equals(first.index) or not grid.columns.equals(first.columns):
            raise AssertionError(f"channel {name!r} is not on the same grid as {channels[0]!r}")
    return grids


def log_return_grid(grid: pd.DataFrame) -> np.ndarray:
    """Second-by-second log returns of a WAP grid, shape (n_segments, n-1)."""
    return np.diff(np.log(grid.to_numpy()), axis=1)


def realized_volatility(log_returns: np.ndarray) -> np.ndarray:
    """sqrt of summed squared log returns, along the last axis.

    This is the competition's definition, and the quantity being forecast.
    """
    return np.sqrt(np.sum(np.square(log_returns), axis=-1))


def suffix_realized_volatility(log_returns: np.ndarray, window: int) -> np.ndarray:
    """Realized volatility over the last `window` seconds of each segment.

    Suffix windows -- not arbitrary slices -- because the feature is read at
    the segment's close, so only a window ending there is available.
    """
    return realized_volatility(log_returns[:, -window:])
