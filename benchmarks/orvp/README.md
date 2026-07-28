# ORVP benchmark

The v0.2 headline benchmark: do path-signature features improve realized
volatility forecasts on the Kaggle [Optiver Realized Volatility
Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction)
dataset, against reproduced baselines, under the competition's own metric?

This directory is research code. It is deliberately *not* part of the
installable `sigtrade` package — it consumes `sigtrade`'s public API exactly
as an outside user would.

## Why this task

Signatures have their strongest theoretical claim here. The lead-lag
transform of a price path has a second-level signature term equal to that
path's discrete quadratic variation (`docs/notes.html` §1.5) — which *is*
realized volatility. So the feature family contains the classical estimator
by construction, and the question becomes whether the rest of the signature
adds anything beyond it. That is a sharp, falsifiable question, and it is
the one this benchmark answers.

## Getting the data

Needs a Kaggle API token at `~/.kaggle/kaggle.json` and acceptance of the
competition rules on the competition page.

```bash
python -m benchmarks.orvp.download          # ~3.5 GB into data/orvp
SIGTRADE_ORVP_DIR=/path/to/orvp python -m benchmarks.orvp.download
```

`SIGTRADE_ORVP_DIR` points every module at an existing copy. The expected
layout is the competition zip's own:

```
<dir>/train.csv
<dir>/book_train.parquet/stock_id=<n>/*.parquet
<dir>/trade_train.parquet/stock_id=<n>/*.parquet
```

## Reproducing a run

```bash
python -m benchmarks.orvp.run   --stocks 20 --jobs 8    # all arms
python -m benchmarks.orvp.study --stocks 20 --jobs 8    # depth x window
```

Results land in `results/` as JSON plus a markdown table. Per-stock features
are cached under `<data dir>/cache/`, keyed by the signature spec, so
rerunning a scoring pass costs only the model fits.

Runtime is dominated by the gradient-boosting fits, not by the signatures —
building every feature for a stock takes well under a second, while one
arm's five folds take minutes.

## The arms

Every arm goes into the *same* learner
(`HistGradientBoostingRegressor`, fixed hyperparameters) on the *same*
folds. Only the input columns change, so a difference in RMSPE is a
statement about features and nothing else. `stock_id` is given to every arm
equally, since each stock has its own baseline volatility level.

| Arm | What it is |
| --- | --- |
| `naive` | Predict the observed window's realized volatility. No model. Volatility is strongly persistent, so this is a hard floor. |
| `har` | HAR-RV-style: realized volatility over nested suffix windows (600/300/150/60/30 s), plus activity and quarticity terms. |
| `book` | Reproduced top-solution-style order-book and trade aggregates: WAP realized volatility at both levels, relative spread, depth imbalance, trade intensity, over nested windows. |
| `sig` | `sigtrade` log-signatures of the lead-lag, time-augmented log-WAP path over the same nested windows. |
| `sig+har`, `sig+book`, `sig+book+har` | Marginal value: what signatures add *on top of* a baseline, which is the question that actually matters. |

## Design decisions worth knowing about

**A regular one-second grid.** The raw book is an irregular event stream.
Everything downstream forward-fills onto a one-second grid. This costs
nothing geometrically — the signature is invariant under reparametrisation,
so forward-filling repeats a point without changing the path — and it makes
the time channel measure real elapsed seconds rather than event count.

**Suffix windows only.** Every window ends at the segment's close, because
that is when the feature would be read. No window can see data the
predictor would not have.

**Grouped CV, not walk-forward.** The competition deliberately shuffled
`time_id`s, so their order is not chronological and a genuine walk-forward
split cannot be reconstructed from the shipped data. What *is* essential and
achievable is grouping: a `time_id` is one instant of market time observed
across all stocks, and volatility is strongly correlated across stocks at
the same instant, so splitting a `time_id` between train and test hands over
most of the answer. `GroupKFold` on `time_id` closes that. This is a
limitation of the benchmark and is reported as one, not worked around.

**`iisignature`, not `roughpy`.** Roughly two orders of magnitude faster for
log-signatures at these sizes (0.11 ms vs 12 ms for a depth-3 signature of a
600-point lead-lag path). The two backends return coordinates in *different*
bases of the free Lie algebra — they agree through level 2 and diverge at
level 3 — so an arm must be fitted and scored on the same backend.
`tests/test_backend.py` pins both down against a basis-independent oracle.

## What the results do and do not claim

The private leaderboard was rescored on market data from *after* the
competition closed. No result here can be translated into a leaderboard
position, and none is. The only claim made is the one the setup supports:

> On the competition training data, with the competition metric, under
> grouped cross-validation and a fixed shared learner, signature features
> changed RMSPE by X% relative to the reproduced baselines.

If that number is negative — if signatures do not help — it is reported as
negative. A confidence interval from a bootstrap over whole `time_id`
groups accompanies every headline comparison, because the difference
between two arms is often smaller than the noise in either.
