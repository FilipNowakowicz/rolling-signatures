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
    from rollsig import n_log_features

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


# --- multichannel signature arm (v0.2.1) ------------------------------------


def test_channel_grids_share_one_index_and_one_time_axis(orvp_root):
    """Alignment: the channels have to be coordinates of a single path.

    If one channel's grid were indexed differently, row k of the stacked
    array would mix second k of one channel with second k of another
    segment, and every cross term would be meaningless.
    """
    book = data.load_book(0, orvp_root)
    grids = data.channel_grids(book)
    assert set(grids) == set(data.CHANNELS)
    reference = data.wap_grid(book)
    for grid in grids.values():
        assert grid.index.equals(reference.index)
        assert list(grid.columns) == list(range(data.SECONDS_PER_SEGMENT))
        assert grid.notna().all().all()

    # The price channel must be the log of the grid the single-channel arm
    # uses, not an independently resampled series.
    assert grids["log_wap"].to_numpy() == pytest.approx(np.log(reference.to_numpy()))


def test_channel_grids_agree_with_the_book_arm_definitions(orvp_root):
    """One definition of spread and imbalance, shared by both arms.

    The comparison only means something if `multisig` and `book` are reading
    the same two quantities; this pins them to the same functions.
    """
    book = data.load_book(0, orvp_root)
    assert data.relative_spread(book).to_numpy() == pytest.approx(
        ((book["ask_price1"] - book["bid_price1"]) / ((book["ask_price1"] + book["bid_price1"]) / 2)).to_numpy()
    )
    imbalance = (book["bid_size1"] + book["bid_size2"] - book["ask_size1"] - book["ask_size2"]) / (
        book["bid_size1"] + book["ask_size1"] + book["bid_size2"] + book["ask_size2"]
    )
    assert data.depth_imbalance(book).to_numpy() == pytest.approx(imbalance.to_numpy())
    assert data.depth_imbalance(book).abs().max() <= 1.0


def test_channel_grids_reject_an_unknown_channel(orvp_root):
    book = data.load_book(0, orvp_root)
    with pytest.raises(ValueError, match="unknown channel"):
        data.channel_grids(book, ("log_wap", "not_a_channel"))


def test_multisignature_features_have_the_advertised_shape(orvp_root):
    """Feature count is Witt's formula on the *joint* path's dimension."""
    from rollsig import n_log_features

    grids = data.channel_grids(data.load_book(0, orvp_root))
    spec = features.MultiSignatureSpec()
    out = features.multisignature_features(grids, spec)

    # 3 channels + time = 4, doubled by lead-lag = 8 generators.
    assert spec.path_dimension == 8
    per_window = n_log_features(8, 2)
    assert per_window == 36
    assert out.shape == (len(grids["log_wap"]), per_window * len(spec.windows))
    assert list(out.columns) == features.multisignature_feature_names(spec)


def test_multisignature_features_are_finite_and_nan_free(orvp_root):
    """NaN-free by construction, not by imputation.

    Every channel is forward-filled onto a full one-second grid before the
    signature is taken, so there is no missing value for a Lie coordinate to
    inherit -- and unlike the book aggregates, a signature has no defensible
    "leave it missing" value.
    """
    grids = data.channel_grids(data.load_book(0, orvp_root))
    out = features.multisignature_features(grids, features.MultiSignatureSpec())
    assert out.notna().all().all()
    assert np.isfinite(out.to_numpy()).all()


def test_multisignature_windows_read_only_their_own_suffix(orvp_root):
    """Causality within the segment: a 150s window sees 150 seconds.

    Overwriting everything before the last 150 seconds must leave the w150
    block untouched and must move the w600 block. This is the off-by-one
    that would quietly turn every window into the full segment.
    """
    grids = data.channel_grids(data.load_book(0, orvp_root))
    spec = features.MultiSignatureSpec(windows=(600, 150))
    original = features.multisignature_features(grids, spec)

    corrupted = {name: grid.copy() for name, grid in grids.items()}
    for name, grid in corrupted.items():
        grid.iloc[:, : data.SECONDS_PER_SEGMENT - 150] = grid.iloc[:, 0].to_numpy()[:, None]
    perturbed = features.multisignature_features(corrupted, spec)

    short = [c for c in original.columns if c.startswith("multisig_w150_")]
    long = [c for c in original.columns if c.startswith("multisig_w600_")]
    assert perturbed[short].to_numpy() == pytest.approx(original[short].to_numpy())
    assert not np.allclose(perturbed[long].to_numpy(), original[long].to_numpy())


def test_multisignature_features_do_not_reach_across_segments(orvp_root, tmp_path):
    """The no-lookahead guarantee, for the multichannel arm specifically.

    Same construction as the single-channel causality test: corrupt every
    later time_id's book and require the earlier segments' features to be
    bit-for-bit unmoved.
    """
    targets = data.load_targets(orvp_root)
    multi_spec = features.MultiSignatureSpec(windows=(600, 300))
    spec = features.SignatureSpec(depth=2, windows=(600,))
    original = features.build_stock_features(0, targets, root=orvp_root, spec=spec, multi_spec=multi_spec)
    assert any(c.startswith("multisig_") for c in original.features.columns)

    corrupted_root = tmp_path / "corrupted_multi"
    book = data.load_book(0, orvp_root)
    trade = data.load_trade(0, orvp_root)
    late = book["time_id"] >= N_TIME_IDS // 2
    for column in ("bid_price1", "ask_price1", "bid_price2", "ask_price2"):
        book.loc[late, column] = 999.0
    for column in ("bid_size1", "ask_size1", "bid_size2", "ask_size2"):
        book.loc[late, column] = 7
    book_dir = corrupted_root / "book_train.parquet" / "stock_id=0"
    trade_dir = corrupted_root / "trade_train.parquet" / "stock_id=0"
    book_dir.mkdir(parents=True)
    trade_dir.mkdir(parents=True)
    book.to_parquet(book_dir / "part-0.parquet", index=False)
    trade.to_parquet(trade_dir / "part-0.parquet", index=False)
    (corrupted_root / "train.csv").write_text((orvp_root / "train.csv").read_text())

    corrupted = features.build_stock_features(
        0, targets, root=corrupted_root, spec=spec, multi_spec=multi_spec
    )
    early = slice(0, N_TIME_IDS // 2)
    multisig = [c for c in original.features.columns if c.startswith("multisig_")]
    pd.testing.assert_frame_equal(
        original.features.iloc[early][multisig],
        corrupted.features.iloc[early][multisig],
        check_exact=False,
    )


def test_multisignature_features_align_with_the_other_arms(orvp_root):
    """Row k of every arm describes the same (stock_id, time_id) segment."""
    targets = data.load_targets(orvp_root)
    block = features.build_stock_features(
        0, targets, root=orvp_root, spec=features.SignatureSpec(depth=2, windows=(600,))
    )
    assert len(block.features) == len(block.target) == len(block.index)

    # Rebuild the multichannel block on its own, from grids restricted to the
    # same time_ids, and require it to reproduce the block inside the table.
    grids = data.channel_grids(data.load_book(0, orvp_root))
    grids = {name: grid.loc[block.index["time_id"].to_numpy()] for name, grid in grids.items()}
    standalone = features.multisignature_features(grids, features.MultiSignatureSpec())
    columns = features.multisignature_feature_names(features.MultiSignatureSpec())
    assert block.features[columns].to_numpy() == pytest.approx(standalone.to_numpy())


def test_multisig_columns_never_leak_into_the_single_channel_arm(orvp_root):
    """`multisig_` must not be selected by the `sig_` prefix, or vice versa."""
    targets = data.load_targets(orvp_root)
    block = features.build_stock_features(
        0, targets, root=orvp_root, spec=features.SignatureSpec(depth=2, windows=(600,))
    )
    sig = features.select_arm(block.features, "sig")
    multisig = features.select_arm(block.features, "multisig")
    har = features.select_arm(block.features, "har")
    book = features.select_arm(block.features, "book")
    assert len(sig.columns) and len(multisig.columns)
    assert set(sig.columns).isdisjoint(multisig.columns)
    assert set(multisig.columns).isdisjoint(har.columns)
    assert set(multisig.columns).isdisjoint(book.columns)

    combined = features.select_arm(block.features, "multisig+book+har")
    assert len(combined.columns) == len(multisig.columns) + len(book.columns) + len(har.columns)
    assert not any(c.startswith("sig_") for c in combined.columns)


def test_multisignature_channel_scaling_is_a_per_feature_rescaling(orvp_root):
    """The reason no channel normalisation is applied.

    Each Lie-basis coordinate is homogeneous in the channels it involves, so
    multiplying a channel by a constant multiplies that coordinate by a
    fixed power of it -- the same factor for every row. A per-feature
    positive rescaling is invisible to the gradient-boosted trees
    downstream, which is what makes "leave the channels in natural units" a
    choice with no consequences rather than an oversight.
    """
    grids = data.channel_grids(data.load_book(0, orvp_root))
    spec = features.MultiSignatureSpec(windows=(600,))
    base = features.multisignature_features(grids, spec).to_numpy()

    scaled_grids = dict(grids)
    scaled_grids["imbalance"] = grids["imbalance"] * 10.0
    scaled = features.multisignature_features(scaled_grids, spec).to_numpy()

    for column in range(base.shape[1]):
        nonzero = np.abs(base[:, column]) > 1e-15
        if nonzero.sum() < 2:
            continue
        ratios = scaled[nonzero, column] / base[nonzero, column]
        assert np.ptp(ratios) / np.abs(ratios).mean() < 1e-8


def test_empty_windows_produce_an_empty_multisig_block(orvp_root):
    """The depth study's opt-out: no windows, no columns, rows preserved."""
    grids = data.channel_grids(data.load_book(0, orvp_root))
    out = features.multisignature_features(grids, features.MultiSignatureSpec(windows=()))
    assert out.shape == (len(grids["log_wap"]), 0)


def test_cache_paths_separate_different_multichannel_specs(tmp_path):
    """A cached frame is reused verbatim, so its path must name every spec.

    Two runs that differ only in the multichannel depth, windows or channels
    must not share a cache entry -- otherwise the second run silently scores
    the first run's features.
    """
    from benchmarks.orvp import run

    spec = features.SignatureSpec(depth=3)
    default = features.MultiSignatureSpec()
    variants = [
        features.MultiSignatureSpec(depth=3),
        features.MultiSignatureSpec(windows=(600,)),
        features.MultiSignatureSpec(channels=("log_wap", "spread")),
        features.MultiSignatureSpec(windows=()),
        features.MultiSignatureSpec(lead_lag_transform=False),
        features.MultiSignatureSpec(time_augmentation=False),
        features.MultiSignatureSpec(subsample=2),
    ]
    paths = {run.cache_path(0, spec, tmp_path, variant) for variant in variants}
    baseline = run.cache_path(0, spec, tmp_path, default)
    assert baseline not in paths
    assert len(paths) == len(variants)

    # Same spec, same path -- otherwise the cache would never hit.
    assert run.cache_path(0, spec, tmp_path, features.MultiSignatureSpec()) == baseline
    # And the single-channel spec still separates entries.
    assert run.cache_path(0, features.SignatureSpec(depth=2), tmp_path, default) != baseline


def test_every_arm_selects_a_nonempty_column_set(orvp_root):
    """No arm in the registry silently resolves to zero features."""
    from benchmarks.orvp import run

    targets = data.load_targets(orvp_root)
    block = features.build_stock_features(
        0, targets, root=orvp_root, spec=features.SignatureSpec(depth=2, windows=(600,))
    )
    for arm in run.ARMS:
        assert len(features.select_arm(block.features, arm).columns) > 0, arm


def _fake_seed_result(improvement: float, p_no_improvement: float) -> dict:
    return {
        "arms": {
            "naive": {"arm": "naive", "n_features": 1, "rmspe": 0.33},
            "book+har": {"arm": "book+har", "n_features": 63, "rmspe": 0.231},
            "multisig+book+har": {"arm": "multisig+book+har", "n_features": 170, "rmspe": 0.230},
        },
        "comparisons": {
            "multisig+book+har vs book+har": {
                "improvement": improvement,
                "improvement_pct": 100.0 * improvement / 0.231,
                "ci_low": improvement - 0.001,
                "ci_high": improvement + 0.001,
                "p_no_improvement": p_no_improvement,
            }
        },
    }


def test_multiseed_calls_a_win_only_when_every_seed_agrees():
    """The stop rule, tested rather than applied by eye.

    Two seeds out of three is not consistency, and the verdict has to say
    "stop" in that case -- otherwise the pre-registration is decorative.
    """
    from benchmarks.orvp import multiseed

    arms = ["book+har", "multisig+book+har"]
    all_win = {
        "0": _fake_seed_result(0.002, 0.001),
        "1": _fake_seed_result(0.003, 0.002),
        "2": _fake_seed_result(0.001, 0.010),
    }
    summary = multiseed.summarise(all_win, arms)
    assert summary["headline"]["consistent"]
    assert summary["headline"]["seeds_below_no_improvement_threshold"] == 3
    assert "continue" in summary["verdict"]

    mixed = dict(all_win, **{"2": _fake_seed_result(0.0005, 0.400)})
    summary = multiseed.summarise(mixed, arms)
    assert not summary["headline"]["consistent"]
    assert summary["headline"]["seeds_below_no_improvement_threshold"] == 2
    assert "stop" in summary["verdict"]

    # A seed that improves on average but with the sign unstable under the
    # bootstrap is not a win either.
    regressed = dict(all_win, **{"1": _fake_seed_result(-0.002, 0.990)})
    summary = multiseed.summarise(regressed, arms)
    assert not summary["headline"]["consistent"]
    assert summary["headline"]["seeds_improved"] == 2
    assert "stop" in summary["verdict"]


def test_multiseed_summary_keeps_every_seed_visible():
    """No averaging away of a seed: the per-seed numbers stay in the artefact."""
    from benchmarks.orvp import multiseed

    per_seed = {
        "0": _fake_seed_result(0.002, 0.001),
        "1": _fake_seed_result(-0.001, 0.900),
    }
    summary = multiseed.summarise(per_seed, ["book+har", "multisig+book+har"])
    headline = summary["headline"]
    assert len(headline["per_seed_improvement_pct"]) == 2
    assert headline["per_seed_improvement_pct"][0] > 0 > headline["per_seed_improvement_pct"][1]
    for row in summary["arms"]:
        assert len(row["per_seed_rmspe"]) == 2
        assert row["min_rmspe"] <= row["mean_rmspe"] <= row["max_rmspe"]

    table = multiseed.markdown_table(summary, (0, 1))
    assert "multisig+book+har vs book+har" in table
    assert "Verdict" in table


def test_naive_prediction_is_the_observed_window_rv(orvp_root):
    targets = data.load_targets(orvp_root)
    spec = features.SignatureSpec(depth=2, windows=(600,))
    block = features.build_stock_features(0, targets, root=orvp_root, spec=spec)
    grid = data.wap_grid(data.load_book(0, orvp_root)).loc[block.index["time_id"]]
    expected = data.realized_volatility(data.log_return_grid(grid))
    assert features.naive_prediction(block.features) == pytest.approx(expected)
