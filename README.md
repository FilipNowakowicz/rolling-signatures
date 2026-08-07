# rolling-signatures

`rollsig` is a `scikit-learn`-compatible library that turns financial time
series into **causal, rolling-window path-signature features**, with
sliding-window updates that cost **O(1) per tick in the steady state** —
derived from the group structure of the tensor algebra (Chen's identity plus
the group inverse).

The project produced a positive computational result and a mixed predictive
result:

- **The streaming engine works, and the asymptotics hold.** Steady-state
  per-tick cost is flat across a 120× change in window length, which makes it
  129× faster than RoughPy at window 1200 without re-anchoring, or **98×**
  faster with the default automatic re-anchoring that bounds numerical drift.
  The numerical price of repeated group operations is measured rather than
  waved away.
  [→ Streaming engine](#streaming-engine-constant-cost-per-tick)
- **On the Optiver Realized Volatility Prediction dataset, incremental value
  was small and learner-dependent.** A final all-112-stock confirmation found
  a statistically supported 0.40% RMSPE improvement under nested ridge, but
  only 0.09% under the fixed gradient booster, whose 95% interval crossed
  zero. The primary success rule therefore did not pass.
  [→ Benchmark](#benchmark-optiver-realized-volatility)

## Install

```bash
uv add "rollsig @ git+https://github.com/FilipNowakowicz/rolling-signatures"
```

Or, for a checkout you intend to work on:

```bash
git clone https://github.com/FilipNowakowicz/rolling-signatures
cd rolling-signatures
uv sync --extra test --extra iisignature
```

Python 3.11+. `RoughPy` is the default backend and installs as a dependency;
`iisignature` is an optional extra (`uv sync --extra test --extra iisignature`)
and is roughly two orders of magnitude faster for log-signatures at the sizes
this library is aimed at.

`rollsig` is installed from this repository rather than from PyPI.

## Use

`SignatureTransformer` follows the scikit-learn estimator interface. Each row
uses only the current and preceding observations; no future value enters the
feature at time `t`, and that is enforced by test rather than assumed.

```python
from rollsig import SignatureTransformer

features = SignatureTransformer(
    window=20,
    depth=3,
    time_augmentation=True,
    lead_lag_transform=True,
).fit_transform(prices)
```

`method="auto"` (the default) is a benchmark-derived heuristic. Against
RoughPy and numpy it chooses streaming because that route won at every measured
window. With `backend="iisignature"`, it uses the measured crossover only for
the ORVP-style configuration of four transformed channels: batch below window
1200 at depth 2 or 600 at depth 3, and streaming from those thresholds. For a
different transformed dimension or a depth outside 2–3, it conservatively
chooses batch. An explicit `method="batch"` or `method="streaming"` overrides
that choice. The two routes agree to floating-point noise, so this is purely a
speed decision; the resolved choice is readable on the fitted estimator as
`transformer.method_`.

`method="auto"` uses crossover thresholds measured on the committed benchmark
environment. For materially different channel counts, hardware, or dependency
versions, benchmark both routes or select the method explicitly.

The engine is also usable directly, on a stream that arrives a tick at a time:

```python
from rollsig import StreamingSignature

engine = StreamingSignature(window=600, depth=3, dim=1, time_augmentation=True)
for price in feed:
    features = engine.update([price])   # constant cost once the window is full
```

For tests and small examples, `backend="numpy"` selects the bundled reference
implementation. The streamed *values* are backend-independent — a sliding
update is arithmetic in the tensor algebra rather than a call into a signature
engine — so `backend` enters the streaming path only through `method="auto"`,
which needs to know what it would otherwise be competing against.

## Positioning and related work

The path signature — an infinite hierarchy of iterated integrals that
characterizes a path up to reparametrization — is well studied (Lyons' rough
path theory, 15+ years of literature), and fast low-level compute libraries
already implement it: `iisignature`, `RoughPy`, `signax`, `esig`. None of them
carry any finance-specific framing.

`rollsig` provides a scikit-learn-compatible, causal rolling interface for
financial time series, together with a stateful tick-by-tick engine,
finance-specific preprocessing, numerical-drift analysis, and reproducible
financial benchmarks. Its streaming engine is the core technical contribution:
an implementation and evaluation of the rolling update, not a claim of a new
mathematical identity.

### Related work

- [`sktime`'s `SignatureTransformer`](https://www.sktime.net/docs/api-reference/sktimetransformationssignature-basedsignaturetransformer/)
  supports signature transforms with global, sliding, expanding, and dyadic
  windowing.
- [`log-signatures-pytorch`](https://pypi.org/project/log-signatures-pytorch/)
  provides PyTorch signature/log-signature computation with streaming outputs;
  its releases also describe windowed signatures derived from a signature
  stream.
- Drobac, Brégère, de Vilmarest, and Wintenberger's
  [*Sliding-Window Signatures for Time Series*](https://arxiv.org/abs/2510.12337)
  studies sliding-window signature features and an efficient sequential
  algorithm.
- Low-level engines including [`RoughPy`](https://roughpy.org/reference/index.html)
  and [`iisignature`](https://github.com/bottler/iisignature) provide efficient
  signature primitives and stream-oriented interfaces.

These projects are not identical to `rollsig`; the distinction here is the
causal sklearn interface, stateful tick-by-tick workflow, finance-oriented
preprocessing, drift analysis, and the committed financial benchmarks.

`rollsig` focuses on:

1. **Causal, rolling features on a stream** — not panel classification. The
   feature at time *t* is a function of data at times ≤ *t* only, enforced and
   tested.
2. **Streaming updates via the algebra, at constant steady-state cost.**
   Signatures are grouplike under concatenation (Chen's identity): sliding a
   window is a left-multiply by the inverse of the departing segment's
   signature and a right-multiply by the new increment, rather than an
   O(window) recomputation.
3. **Finance-specific defaults, evaluated honestly** — lead-lag transformation
   (which recovers quadratic variation), time augmentation, and benchmarks
   that report negative results when signatures don't help.

## Why signatures — the math in one paragraph

The signature of a path lives in the tensor algebra
$T((V)) = \bigoplus_n V^{\otimes n}$, a noncommutative associative ring under
the tensor product; the log-signature lives in the free Lie algebra generated
by $V$. Signatures satisfy **Chen's identity** (path concatenation corresponds
to a tensor product of signatures) and the **shuffle-product identity** (a
Hopf-algebra structure — a commutative ring dual to the tensor algebra). These
aren't decoration: Chen's identity is what makes the constant-cost streaming
update possible, and the shuffle relations are used directly as a correctness
oracle in the tests.

The full expository write-up is the three-part working notes, written chapter
by chapter as the code was built: [`docs/notes.html`](docs/notes.html) (the
algebra, worked from a live example, and the core transformer),
[`docs/notes-orvp.html`](docs/notes-orvp.html) (the benchmark), and
[`docs/notes-streaming.html`](docs/notes-streaming.html) (the streaming
engine). They render properly at
**<https://filipnowakowicz.github.io/rolling-signatures/>**.

## Benchmark: Optiver realized volatility

**Headline: on this task, multichannel signatures contain a small amount of
incremental information, but the gain is learner-dependent rather than a
robust improvement over reproduced order-book baselines.**
The full study is in [`benchmarks/orvp/`](benchmarks/orvp/README.md); the
reasoning behind the design is [`docs/notes-orvp.html`](docs/notes-orvp.html).

Why this task is a sharp test rather than a fishing expedition: the lead-lag
transform makes quadratic variation appear *exactly* at level 2 of the
signature, so realized volatility is not something bolted onto the feature
family — it is a coordinate of it. The question is therefore precise: does
anything **above** level 2 forecast the next ten minutes better than the
classical estimator alone?

Three independent 20-stock subsets (pre-registered seeds 0/1/2), ~76,600
ten-minute segments each, depth-3 log-signatures over 600/300/150-second
causal suffix windows. Every arm goes into the same
`HistGradientBoostingRegressor` on the same `GroupKFold` splits; only the
input columns change. Lower RMSPE is better.

| Arm | Features | seed 0 | seed 1 | seed 2 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| naive (predict the observed window's RV) | 1 | 0.33485 | 0.40147 | 0.39580 | 0.37737 |
| `har` — HAR-RV-style multi-horizon RV | 27 | 0.24075 | 0.27261 | 0.27822 | 0.26386 |
| **`book`** — reproduced top-solution-style book/trade aggregates | 37 | 0.23177 | 0.27035 | 0.26953 | **0.25722** |
| `book+har` | 63 | 0.23093 | 0.27121 | 0.27218 | 0.25811 |
| `sig` — log-signatures of the price path | 91 | 0.24362 | 0.28669 | 0.28993 | 0.27341 |
| `sig+har` | 117 | 0.23980 | 0.28040 | 0.28368 | 0.26796 |
| `sig+book` | 127 | 0.23199 | 0.27928 | 0.27512 | 0.26213 |
| `sig+book+har` | 153 | 0.23133 | 0.27603 | 0.27497 | 0.26078 |
| `multisig` — log-signatures of (price, spread, imbalance) | 109 | 0.24192 | 0.28645 | 0.28476 | 0.27104 |
| `multisig+har` | 135 | 0.23902 | 0.27736 | 0.28419 | 0.26686 |
| `multisig+book` | 145 | 0.23250 | 0.27317 | 0.27218 | 0.25929 |
| `multisig+book+har` | 171 | 0.23112 | 0.27478 | 0.27259 | 0.25950 |

Read honestly, four things happened:

1. **Signatures alone nearly match hand-designed volatility features — on one
   subset.** On seed 0, 91 log-signature coordinates land within 1.2% of a
   27-feature HAR-RV set, without directly supplying realized-volatility,
   quarticity, or activity statistics, although the lead-lag preprocessing and
   suffix horizons were volatility-motivated. On seeds 1 and 2 the same comparison is
   4–5% adrift. The generic construction is competitive on some stock
   universes and clearly not on others.
2. **They add nothing to a strong baseline.** `sig+book+har` is worse than
   `book+har` on all three subsets (−0.17%, −1.78%, −1.02%; mean −0.99%).
3. **Giving signatures the order-book state does not rescue them.** The
   obvious objection to the above was that `sig` saw only the WAP path while
   `book` saw order-book *state*, making it an input-set comparison rather
   than a feature-family one. The multichannel arm closed that: `multisig`
   carries relative spread and depth imbalance as extra channels of one path
   (depth 2, same windows, same functions the `book` arm uses). It does beat
   the price-only arm on every subset (+0.70%, +0.08%, +1.78%) — the confound
   was real — but `multisig+book+har` is still worse than `book+har` on all
   three (−0.08%, −1.32%, −0.15%; mean −0.52%), and
   `p_no_improvement < 0.05` under the grouped bootstrap on **0 of 3**.
4. **Replication mattered more than the point estimates.** Seed 0 was the most
   favourable of the three subsets, and it is the one the first pass reported.
   `multisig+har` beats `har` by 0.72% on seed 0 and loses by 1.75% and 2.14%
   on the other two. Any of these single-subset numbers, read alone, would
   have supported a conclusion the other subsets contradict.

The three-subset stage therefore rejected the claim that the signatures
reliably improved the fixed gradient-boosting pipeline. Because its subsets
overlap and leave some stocks untested, a final confirmation then froze the
headline comparison and evaluated the full universe rather than opening a new
configuration search.

### Full-universe confirmation

The protocol in
[`benchmarks/orvp/UNIVERSE_STUDY.md`](benchmarks/orvp/UNIVERSE_STUDY.md) was
fixed before the new scores were computed: all 112 stocks and 428,932 segments,
`book+har` versus the existing depth-2 `multisig+book+har` arm, with no other
signature configurations scored. The original fixed gradient booster is the
primary learner. A weighted ridge model with fold-local preprocessing and
three-fold grouped inner selection of its regularisation is a pre-specified
robustness learner.

| Learner | Baseline RMSPE | + multichannel signatures | Improvement | 95% grouped CI | Stocks improved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed gradient booster | 0.23399 | 0.23378 | +0.09% | [−0.13%, +0.31%] | 77/112 |
| Nested ridge | 0.23499 | 0.23405 | **+0.40%** | **[+0.31%, +0.49%]** | **92/112** |

The ridge gain appears in all five outer folds and its grouped-bootstrap
interval excludes zero. The primary gradient-booster interval does not, and
the challenger is worse in two of its five folds. The defensible conclusion
is therefore narrower than either “signatures do not help” or “signatures beat
HAR”: the multichannel coordinates expose incremental linear information
after conditioning on book + HAR features, but do not produce a statistically
reliable gain for the fixed nonlinear learner.

The pre-specified success criterion required the primary learner's interval to
exclude zero, so it does not pass and the ORVP configuration search remains
closed. This learner contrast is the empirical result; it will not be followed
by choosing whichever learner or subgroup makes signatures look strongest.
Full aggregate, fold, bootstrap, and per-stock results are committed in
[`benchmarks/orvp/results/`](benchmarks/orvp/results/).

Two smaller findings worth recording. `book` alone beats `book+har` on two of
three subsets, so the best baseline is not the same arm on every subset —
which is why `book+har` was fixed in advance as the comparator rather than
chosen after the fact. And the multichannel arm reaches its results at depth
**2** with 109 features against the price-only arm's depth 3 and 91: the gain
came from channels, not from higher-order terms.

### Truncation depth vs. window count

Sweeping depth against window sets (signature columns only, same folds and
learner) puts the turnover at **depth 3**: going 2 → 3 buys a real gain, while
3 → 4 triples the feature count (91 → 271) and moves RMSPE by 0.00008 — a
thirtieth of the fold standard deviation.

The sharpest line falls out of a coincidence in feature counts. Depth 4 over
one 600 s window and depth 3 over three nested windows both yield exactly
**91 features**, and they do not perform alike:

| 91 features, spent on… | RMSPE |
| --- | ---: |
| depth — level 4 of one window | 0.24627 |
| horizons — level 3 of three windows | 0.24362 |

**At a fixed feature budget, buy horizons rather than depth.** Volatility
forecasting wants to know how the recent past differs from the less-recent
past — a statement about *windows* — more than it wants a finer description of
any one window's geometry. Full table: `benchmarks/orvp/results/`.

The depth/window study was run on seed 0 only, before the replication existed.
Given how much the seed-0 numbers move on other subsets, treat its turnover
point as indicative rather than settled.

### What this does not claim

The competition's private leaderboard was rescored on market data from *after*
it closed, so no result obtainable from the training data can be translated
into a leaderboard position, and none is. Cross-validation here is
`GroupKFold` on `time_id`, not walk-forward — the organisers shuffled
`time_id` and shipped no timestamps, so no chronological order is recoverable
from the data. Grouping still closes the leak that dominates here (a `time_id`
is one instant of market time across all 112 stocks, and volatility is
strongly correlated cross-sectionally), but it does not simulate deployment
across time.

## Streaming engine (constant cost per tick)

A rolling window of *w* points advancing by one shares *w*−1 points with its
predecessor, and Chen's identity says the shared part need not be re-read:

    S(next window) = S(departing increment)⁻¹ · S(window) · S(arriving increment)

Neither factor depends on *w*, so once there is a departing increment to
cancel — that is, once the window is full — the per-tick cost is a function of
`depth` and the channel count alone. `rollsig.algebra` is the truncated tensor
algebra as a callable object (multiply, group inverse, exp, log, dilate, with
the group laws as property-based tests); `rollsig.streaming` turns it into a
rolling-feature API, which `SignatureTransformer` selects wherever it was
measured to be the faster route.

**What is claimed, precisely.** Once the window is full, a tick costs O(1) —
that is the identity above. Re-anchoring (below) spends one O(window)
recomputation every `window` ticks, so the steady state is **O(1) amortized**.
The warm-up is not constant-time: with `time_augmentation=True`, `time_augment`
rescales its channel to [0, 1] over however many points have arrived, so every
tick before the window fills changes the increments already accumulated and is
recomputed instead — O(window²) once at the head of a stream. The measurements
below, like the claim, are steady-state.

**Measured, per tick, in microseconds:**

| depth | window | streaming | batch RoughPy | batch `iisignature` | batch numpy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 10 | 203.0 | 528.0 | 27.1 | 722.5 |
| 3 | 300 | 194.4 | 6673.0 | 172.5 | 22483.7 |
| 3 | 1200 | 201.9 | 25987.6 | 628.0 | 91302.8 |

Two results, in order of how much weight they carry:

1. **Per-tick cost is flat in the window** — 127 → 135 µs at depth 2 and
   203 → 202 µs at depth 3 across a 120× change in window length, while every
   recompute column grows with the window exactly as it must. This is the
   result that matters: it is a statement about the algebra, so a measured
   slope would have meant a bug.
2. **129× faster than RoughPy at window 1200 without re-anchoring**, and
   **98× faster with the default automatic re-anchoring** (264.6 µs versus
   25,987.6 µs). The unrefreshed engine is faster than RoughPy at every window
   measured, from 1.7× at window 10. Because streaming is flat and
   recomputation is not, that ratio keeps growing with the window.

**The qualification belongs in the same breath.** The update is interpreted
Python doing dozens of small numpy calls, so against `iisignature`'s compiled
recompute it wins only past a crossover window — measured at 1200 for depth 2
and 600 for depth 3. Below that, compiled batch is faster despite being
asymptotically worse, and it is what `method="auto"` picks when the backend is
`iisignature`. The pure-numpy comparison (up to 452× at window 1200) is the
weakest of the three and is reported for completeness only: numpy here is a
test oracle, not a serious competitor.

**The numerical price is quantified, not waved away.** Repeated group
operations drift from a from-scratch recomputation by ~3 orders of magnitude
per level of depth, and *quadratically* in unanchored ticks (×4 per doubling).
Re-anchoring on a real recomputation once per `window` ticks keeps the steady
state O(1) amortized, costs 23–37% per tick, and pins the drift at 7e-15 at
every depth — so it is the default (`refresh_every="auto"`).

**And one retraction.** `docs/notes-orvp.html` §9.1 predicted that group
inverses would remove the redundancy in the ORVP benchmark's nested
600/300/150-second windows. Measured, that route saves nothing; the right
decomposition for a *nested* family is disjoint chunks combined forwards, with
no inverse at all — and even that loses by 80× to simply calling `iisignature`
three times. Group inverses earn their keep on sliding windows, where the
shared part cannot be re-decomposed, and not here. Full study:
[`benchmarks/streaming/`](benchmarks/streaming/README.md); reasoning:
[`docs/notes-streaming.html`](docs/notes-streaming.html).

## What is in the repository

| Path | What it holds |
| --- | --- |
| `src/rollsig/` | The library: `transformer.py` (the sklearn estimator), `preprocessing.py` (basepoint, time augmentation, lead-lag, rescaling), `algebra.py` (the truncated tensor algebra), `streaming.py` (the rolling engine), `_backend.py` (the narrow RoughPy / `iisignature` / numpy interface). |
| `tests/` | Tests and linting run in CI on Python 3.11 and 3.12. The correctness oracles are mathematical: Chen's identity, the shuffle identity, primitivity in the free Lie algebra, invariance under time reparametrization, the group laws, and agreement between the backends. |
| `CONTRIBUTING.md` | Local checks, package layout, and benchmark/result-integrity rules. |
| `benchmarks/orvp/` | The realized-volatility study — data pipeline, arms, grouped CV, bootstrap, three-subset replication, and the all-112-stock confirmation. Committed aggregate and per-stock results in `results/`. |
| `benchmarks/streaming/` | The streaming study — per-tick timings, the drift measurement, the nested-window retraction. Committed results in `results/`. |
| `docs/` | The three-part working notes: the mathematics and the build, chapter by chapter. |

Both benchmark directories are research code, deliberately outside the
installable package: they consume `rollsig`'s public API exactly as an outside
user would. Each has a README with the commands to reproduce its numbers from
scratch.

## Scope and limitations

Deliberately not done:

- **Low-level signature computation for production use.** This wraps
  `iisignature` and `RoughPy`; the pure-numpy implementation exists as a test
  oracle. A compiled version of the streaming update would move the
  `iisignature` crossover to a very small window — and would also make
  `rollsig.algebra` unreadable, which is most of its value.
- **Panel/segment classification.** That is `sktime`'s job and it does it
  well.
- **A general-purpose ML library.** This stays scoped to financial time
  series.

Two configurations fall back to an exact batch computation instead of
streaming, so they cost speed and never correctness, and
`SignatureTransformer` selects the fallback itself: `basepoint=True`, which
changes how the window's left boundary is represented, and
`output="log_signature"`, which returns coordinates in a backend-specific
basis of the free Lie algebra while the streaming engine maintains the full
tensor signature. Factorial `rescale` is not defined for log-signature output
— it would need indexing by bracket depth rather than word length — and raises
rather than silently doing the wrong thing.

## License

This project is licensed under the [MIT License](LICENSE).
