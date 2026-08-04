# ORVP benchmark

The headline benchmark: do path-signature features improve realized
volatility forecasts on the Kaggle [Optiver Realized Volatility
Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction)
dataset, against reproduced baselines, under the competition's own metric?

This directory is research code. It is deliberately *not* part of the
installable `rollsig` package — it consumes `rollsig`'s public API exactly
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

Two prerequisites, both on kaggle.com:

1. **An API token.** The current Kaggle CLI reads the newer prefixed token
   from `~/.kaggle/access_token` (or `KAGGLE_API_TOKEN`) — *not* the older
   `~/.kaggle/kaggle.json` username/key pair that most documentation still
   describes.
2. **Acceptance of the competition rules**, on the competition page. Worth
   knowing because the failure is misleading: listing files and reading the
   leaderboard work fine without it, and only *downloads* 403 — which looks
   exactly like a broken token and isn't.

```bash
uv run python -m benchmarks.orvp.download               # train.csv + a 20-stock subset
uv run python -m benchmarks.orvp.download --stocks 40   # a larger subset
uv run python -m benchmarks.orvp.download --all         # the entire archive
```

The subset download is the default, since the benchmark evaluates on a stock
subset anyway: 20 stocks is a few hundred megabytes against the full
archive's several gigabytes. `train.csv` is fetched first because the subset
is drawn from the stock ids that actually appear in it.

`ROLLSIG_ORVP_DIR` points every module at an existing copy. The expected
layout is the competition archive's own:

```
<dir>/train.csv
<dir>/book_train.parquet/stock_id=<n>/*.parquet
<dir>/trade_train.parquet/stock_id=<n>/*.parquet
```

## Reproducing a run

```bash
uv run python -m benchmarks.orvp.run       --stocks 20 --jobs 8       # all arms, one subset
uv run python -m benchmarks.orvp.multiseed --seeds 0 1 2 --jobs 8     # all arms, three subsets
uv run python -m benchmarks.orvp.study     --stocks 20 --jobs 8       # depth x window
```

`multiseed` needs each seed's stocks on disk first:

```bash
for seed in 0 1 2; do uv run python -m benchmarks.orvp.download --stocks 20 --seed $seed; done
```

Results land in `results/` as JSON plus a markdown table. Per-stock features
are cached under `<data dir>/cache/`, keyed by *both* signature specs — the
single-channel one and the multichannel one — so a run can never be served a
cached frame that was built under a different configuration.

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
| `sig` | `rollsig` log-signatures of the lead-lag, time-augmented log-WAP path over the same nested windows. Depth 3. |
| `sig+har`, `sig+book`, `sig+book+har` | Marginal value: what signatures add *on top of* a baseline, which is the question that actually matters. |
| `multisig` | Log-signatures of the *joint* (log-WAP, spread, imbalance) path over the same windows. Depth 2. |
| `multisig+har`, `multisig+book`, `multisig+book+har` | The same marginal-value question for the multichannel arm. |

### Why a multichannel arm exists (v0.2.1)

v0.2's negative result had an obvious confound: the `sig` arm saw only the
WAP price path, while the `book` arm it lost to saw order-book *state*. That
made it an input-set comparison, not a feature-family one. `multisig` gives
the signature arm the same two state variables — relative spread and depth
imbalance — as extra channels of one path, computed from the same functions
`book` uses (`data.relative_spread`, `data.depth_imbalance`), so the two arms
are demonstrably reading the same quantities.

**Depth 2, not 3.** Three channels plus time, doubled by lead-lag, is an
8-dimensional path; the free Lie algebra on 8 generators has 36 coordinates
at depth 2 and 336 at depth 3. Depth 2 is also where the theory points: the
level-2 lead-lag coordinates of a channel *pair* are that pair's discrete
quadratic covariation, so what this arm can say that no per-window mean or
standard deviation can is "the spread widened *while* the price moved". If
multichannel signatures earn anything here, that is the term that earns it.

**No channel normalisation, deliberately.** Scaling a channel by a constant
multiplies each log-signature coordinate by a fixed power of it — the same
factor on every row — so it is a positive per-feature rescaling, and the
gradient-boosted trees are exactly invariant to those. Channels stay in
natural units; `tests/test_orvp.py` pins the invariance down rather than
leaving it as an assertion in a docstring.

### Repeating across stock subsets

One 20-stock subset cannot distinguish "this feature family helps" from
"this feature family helps *on these twenty stocks*", and the grouped
bootstrap cannot either — it resamples `time_id`s, so it prices in
market-instant variation with the universe held fixed. `multiseed` reruns
everything on three subsets drawn from fixed seeds (0, 1, 2), chosen before
any arm was scored. Here `p_no_improvement` is the fraction of grouped-
bootstrap resamples whose improvement is non-positive; it is not presented as
a conventional hypothesis-test p-value. A comparison counts as a win only if
its improvement is positive and `p_no_improvement < 0.05` under the grouped
bootstrap on **every** seed; two out of three is not consistency.

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

## Result

Three pre-registered subsets, all eleven arms, one shared learner. Full
tables in `results/multiseed_table.md`; the per-seed bootstrap intervals are
in `results/multiseed.json`.

| Comparison | seed 0 | seed 1 | seed 2 | Mean | Seeds with `p_no_improvement < 0.05` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `book+har` vs `naive` | +31.03% | +32.45% | +31.23% | +31.57% | 3/3 |
| `sig+book+har` vs `book+har` | −0.17% | −1.78% | −1.02% | −0.99% | 0/3 |
| `multisig+book+har` vs `book+har` | −0.08% | −1.32% | −0.15% | −0.52% | 0/3 |
| `multisig` vs `sig` | +0.70% | +0.08% | +1.78% | +0.85% | 1/3 |
| `multisig+book+har` vs `sig+book+har` | +0.09% | +0.45% | +0.87% | +0.47% | 0/3 |

**The multichannel arm beats the price-only arm on every subset and still
loses to the aggregates on every subset.** The v0.2 confound was real —
giving signatures the order-book state does help them — and it was not the
explanation. The headline comparison is negative on all three seeds and has
`p_no_improvement < 0.05` on none, so the pre-registered stop rule fires:
**the ORVP search is closed.**

Two things the replication caught that a single run would not have:

- **Seed 0 was the most favourable subset**, and it is the one v0.2
  reported. `sig+book+har` vs `book+har` is −0.17% there and −1.78% on seed
  1. `multisig+har` vs `har` is *positive* on seed 0 (+0.72%) and negative
  on both others (−1.75%, −2.14%). Single-subset readings of these arms
  would have supported conclusions the other subsets contradict.
- **`book` alone beats `book+har` on two of three subsets**, so the
  strongest baseline is not the same arm everywhere. `book+har` was fixed as
  the comparator in advance precisely so the target could not drift toward
  whichever baseline happened to be weakest.

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

Nor does the negative result generalise beyond this task. It says that on
10-minute order-book segments, with this learner and these baselines,
signatures of the book do not beat aggregates of the book. It says nothing
about signatures on other horizons, other instruments, or other targets, and
it is not evidence about the streaming engine (v0.3), which is a claim about
computational cost rather than predictive value.
