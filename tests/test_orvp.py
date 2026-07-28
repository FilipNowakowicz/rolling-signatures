"""Tests for the ORVP benchmark harness.

The competition data is 3.5 GB and needs credentials, so everything here
runs against a synthetic dataset written into a tmp dir in the competition's
exact on-disk layout. That keeps CI honest about the loading path -- the
parquet partitioning and column names are the real ones -- while testing the
things that would silently corrupt a benchmark: alignment, leakage, and
causality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from benchmarks.orvp import data, evaluate, features  # noqa: E402

N_TIME_IDS = 24
STOCKS = (0, 1)


def _synthetic_stock(stock_id: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One stock's book and trade frames, with irregular update times."""
    books, trades = [], []
    for time_id in range(N_TIME_IDS):
        n_updates = int(rng.integers(40, 120))
        seconds = np.unique(np.concatenate([[0], rng.integers(1, 600, size=n_updates)]))
        mid = 1.0 + np.cumsum(rng.normal(0, 2e-4, size=len(seconds)))
        half_spread = rng.uniform(1e-4, 5e-4, size=len(seconds))
        books.append(
            pd.DataFrame(
                {
                    "time_id": time_id,
                    "seconds_in_bucket": seconds,
                    "bid_price1": mid - half_spread,
                    "ask_price1": mid + half_spread,
                    "bid_price2": mid - 2 * half_spread,
                    "ask_price2": mid + 2 * half_spread,
                    "bid_size1": rng.integers(1, 500, size=len(seconds)),
                    "ask_size1": rng.integers(1, 500, size=len(seconds)),
                    "bid_size2": rng.integers(1, 500, size=len(seconds)),
                    "ask_size2": rng.integers(1, 500, size=len(seconds)),
                }
            )
        )
        n_trades = int(rng.integers(1, 20))
        trade_seconds = np.unique(rng.integers(0, 600, size=n_trades))
        trades.append(
            pd.DataFrame(
                {
                    "time_id": time_id,
                    "seconds_in_bucket": trade_seconds,
                    "price": 1.0 + rng.normal(0, 1e-3, size=len(trade_seconds)),
                    "size": rng.integers(1, 200, size=len(trade_seconds)),
                    "order_count": rng.integers(1, 10, size=len(trade_seconds)),
                }
            )
        )
    return pd.concat(books, ignore_index=True), pd.concat(trades, ignore_index=True)


@pytest.fixture(scope="module")
def orvp_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("orvp")
    rng = np.random.default_rng(0)
    rows = []
    for stock_id in STOCKS:
        book, trade = _synthetic_stock(stock_id, rng)
        book_dir = root / "book_train.parquet" / f"stock_id={stock_id}"
        trade_dir = root / "trade_train.parquet" / f"stock_id={stock_id}"
        book_dir.mkdir(parents=True)
        trade_dir.mkdir(parents=True)
        book.to_parquet(book_dir / "part-0.parquet", index=False)
        trade.to_parquet(trade_dir / "part-0.parquet", index=False)
        for time_id in range(N_TIME_IDS):
            rows.append({"stock_id": stock_id, "time_id": time_id, "target": rng.uniform(1e-4, 5e-3)})
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)
    return root


def test_is_available_detects_the_layout(orvp_root, tmp_path):
    assert data.is_available(orvp_root)
    assert not data.is_available(tmp_path)


def test_wap_grid_is_regular_and_forward_filled(orvp_root):
    book = data.load_book(0, orvp_root)
    grid = data.wap_grid(book)
    assert list(grid.columns) == list(range(data.SECONDS_PER_SEGMENT))
    assert grid.notna().all().all()

    # A second with no book update must repeat the previous second's WAP,
    # never interpolate toward the next one.
    first = book[book["time_id"] == 0]
    observed = set(first["seconds_in_bucket"])
    gaps = [s for s in range(1, 600) if s not in observed]
    row = grid.loc[0]
    for second in gaps[:20]:
        assert row[second] == row[second - 1]


def test_wap_matches_the_competition_definition(orvp_root):
    book = data.load_book(0, orvp_root).head(5)
    expected = (book["bid_price1"] * book["ask_size1"] + book["ask_price1"] * book["bid_size1"]) / (
        book["bid_size1"] + book["ask_size1"]
    )
    assert data.wap(book, level=1).to_numpy() == pytest.approx(expected.to_numpy())


def test_realized_volatility_matches_the_definition():
    returns = np.array([[0.01, -0.02, 0.005]])
    assert data.realized_volatility(returns)[0] == pytest.approx(np.sqrt(0.0001 + 0.0004 + 0.000025))


def test_suffix_realized_volatility_uses_only_the_tail():
    returns = np.zeros((1, 600))
    returns[0, :300] = 1.0  # entirely in the first half
    assert data.suffix_realized_volatility(returns, 300)[0] == pytest.approx(0.0)
    assert data.suffix_realized_volatility(returns, 600)[0] > 0


def test_features_depend_only_on_the_segment_they_describe(orvp_root, tmp_path):
    """The causality guarantee, tested rather than assumed.

    Rebuild one stock's features from a dataset in which every *later*
    time_id's book has been overwritten with garbage. If any feature for the
    earlier segments moved, something is reaching across segments -- which
    on real data would mean reaching into the future.
    """
    targets = data.load_targets(orvp_root)
    spec = features.SignatureSpec(depth=2, windows=(600, 300))
    original = features.build_stock_features(0, targets, root=orvp_root, spec=spec)

    corrupted_root = tmp_path / "corrupted"
    book = data.load_book(0, orvp_root)
    trade = data.load_trade(0, orvp_root)
    late = book["time_id"] >= N_TIME_IDS // 2
    for column in ("bid_price1", "ask_price1", "bid_price2", "ask_price2"):
        book.loc[late, column] = 999.0
    book_dir = corrupted_root / "book_train.parquet" / "stock_id=0"
    trade_dir = corrupted_root / "trade_train.parquet" / "stock_id=0"
    book_dir.mkdir(parents=True)
    trade_dir.mkdir(parents=True)
    book.to_parquet(book_dir / "part-0.parquet", index=False)
    trade.to_parquet(trade_dir / "part-0.parquet", index=False)
    (corrupted_root / "train.csv").write_text((orvp_root / "train.csv").read_text())

    corrupted = features.build_stock_features(0, targets, root=corrupted_root, spec=spec)
    early = slice(0, N_TIME_IDS // 2)
    pd.testing.assert_frame_equal(
        original.features.iloc[early], corrupted.features.iloc[early], check_exact=False
    )


def test_signature_features_have_the_advertised_shape(orvp_root):
    from sigtrade import n_log_features

    book = data.load_book(0, orvp_root)
    grid = data.wap_grid(book)
    spec = features.SignatureSpec(depth=3, windows=(600, 300))
    out = features.signature_features(grid, spec)
    # log WAP -> +time -> lead-lag doubles it: dimension 4.
    per_window = n_log_features(4, 3)
    assert out.shape == (len(grid), per_window * len(spec.windows))
    assert list(out.columns) == features.signature_feature_names(spec)
    assert out.notna().all().all()


def test_signature_windows_differ_from_each_other(orvp_root):
    """A shorter suffix window must actually see a different path.

    Guards against an off-by-one in the slicing that would make every window
    the full segment and quietly turn the multi-horizon arm into duplicates.
    """
    grid = data.wap_grid(data.load_book(0, orvp_root))
    spec = features.SignatureSpec(depth=2, windows=(600, 150))
    out = features.signature_features(grid, spec)
    long_columns = [c for c in out.columns if c.startswith("sig_w600_")]
    short_columns = [c for c in out.columns if c.startswith("sig_w150_")]
    assert not np.allclose(out[long_columns].to_numpy(), out[short_columns].to_numpy())


def test_arm_prefixes_select_disjoint_nonempty_column_sets(orvp_root):
    targets = data.load_targets(orvp_root)
    spec = features.SignatureSpec(depth=2, windows=(600,))
    block = features.build_stock_features(0, targets, root=orvp_root, spec=spec)
    har = features.select_arm(block.features, "har")
    book = features.select_arm(block.features, "book")
    sig = features.select_arm(block.features, "sig")
    assert len(har.columns) and len(book.columns) and len(sig.columns)
    assert set(har.columns).isdisjoint(book.columns)
    assert set(har.columns).isdisjoint(sig.columns)
    assert set(book.columns).isdisjoint(sig.columns)
    combined = features.select_arm(block.features, "sig+book+har")
    assert len(combined.columns) == len(har.columns) + len(book.columns) + len(sig.columns)


def test_features_align_with_targets(orvp_root):
    targets = data.load_targets(orvp_root)
    spec = features.SignatureSpec(depth=2, windows=(600,))
    block = features.build_stock_features(0, targets, root=orvp_root, spec=spec)
    assert len(block.features) == len(block.target) == len(block.index)
    expected = targets[targets["stock_id"] == 0].set_index("time_id")["target"]
    for i, time_id in enumerate(block.index["time_id"]):
        assert block.target.iloc[i] == pytest.approx(expected.loc[time_id])


def test_rmspe_is_relative_not_absolute():
    y = np.array([1.0, 100.0])
    # Same absolute error on both points, but the metric must weight the
    # small one far more heavily.
    assert evaluate.rmspe(y, y + 0.1) == pytest.approx(np.sqrt((0.01 + 1e-6) / 2))
    assert evaluate.rmspe(y, y * 1.1) == pytest.approx(0.1)


def test_folds_never_split_a_time_id_across_train_and_test():
    """The leak this benchmark most needs to not have.

    A time_id is one instant of market time shared by every stock. If one
    stock's row from that instant trains a model scored on another stock's
    row from the same instant, cross-sectional volatility correlation hands
    over most of the answer.
    """
    groups = np.repeat(np.arange(50), 4)
    X = pd.DataFrame({"a": np.arange(len(groups), dtype=float)})
    y = np.ones(len(groups))
    for train_idx, test_idx in evaluate.folds(groups, n_splits=5).split(X, y, groups):
        assert set(groups[train_idx]).isdisjoint(groups[test_idx])
    assert len(np.unique(groups)) == 50


def test_run_arm_produces_one_prediction_per_row(orvp_root):
    targets = data.load_targets(orvp_root)
    spec = features.SignatureSpec(depth=2, windows=(600,))
    blocks = [features.build_stock_features(s, targets, root=orvp_root, spec=spec) for s in STOCKS]
    X = pd.concat([b.features for b in blocks], ignore_index=True)
    y = pd.concat([b.target for b in blocks], ignore_index=True).to_numpy()
    groups = pd.concat([b.index for b in blocks], ignore_index=True)["time_id"].to_numpy()

    result = evaluate.run_arm("sig", features.select_arm(X, "sig"), y, groups, n_splits=3)
    assert result.predictions.shape == y.shape
    assert np.isfinite(result.predictions).all()
    assert len(result.fold_rmspe) == 3
    assert result.n_features == len(features.select_arm(X, "sig").columns)


def test_paired_bootstrap_detects_a_strictly_better_arm():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 3)
    y = rng.uniform(1e-3, 5e-3, size=len(groups))
    worse = y * rng.normal(1.0, 0.30, size=len(y))
    better = y * rng.normal(1.0, 0.05, size=len(y))
    stats = evaluate.paired_bootstrap(y, worse, better, groups, n_resamples=200)
    assert stats["improvement"] > 0
    assert stats["p_no_improvement"] < 0.05


def test_paired_bootstrap_reports_no_improvement_for_identical_arms():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(40), 3)
    y = rng.uniform(1e-3, 5e-3, size=len(groups))
    predictions = y * rng.normal(1.0, 0.2, size=len(y))
    stats = evaluate.paired_bootstrap(y, predictions, predictions, groups, n_resamples=200)
    assert stats["improvement"] == pytest.approx(0.0)
    assert stats["ci_low"] == pytest.approx(0.0)
    assert stats["ci_high"] == pytest.approx(0.0)


def test_naive_prediction_is_the_observed_window_rv(orvp_root):
    targets = data.load_targets(orvp_root)
    spec = features.SignatureSpec(depth=2, windows=(600,))
    block = features.build_stock_features(0, targets, root=orvp_root, spec=spec)
    grid = data.wap_grid(data.load_book(0, orvp_root)).loc[block.index["time_id"]]
    expected = data.realized_volatility(data.log_return_grid(grid))
    assert features.naive_prediction(block.features) == pytest.approx(expected)
