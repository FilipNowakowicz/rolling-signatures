# Roadmap

Working name: **sigtrade**. Rename freely — not load-bearing.

One-liner: a `scikit-learn`-compatible transformer that turns raw OHLCV time
series into path-signature features, benchmarked head-to-head against
standard technical indicators on real market data. The gap it fills: the
math (rough path theory / signatures) and the low-level compute libraries
(`iisignature`, `signax`) already exist, but nobody has packaged them as a
drop-in feature extractor for a quant research pipeline.

Full background and rationale: see `CLAUDE.md`.

## v0.1 — core transformer + first honest benchmark

- [ ] Package skeleton: `pyproject.toml`, `src/sigtrade/`, `tests/`
- [ ] `SignatureTransformer` (sklearn `fit`/`transform` API): rolling-window
      multivariate time series in -> truncated signature / log-signature
      features out. Wrap `iisignature` for the actual tensor computation —
      do not reimplement the optimized C code.
- [ ] Path preprocessing: time-augmentation, lead-lag transform (needed to
      let signatures see quadratic variation / realized vol), basepoint
      handling, factorial rescaling of higher-order terms.
- [ ] Pure-numpy reference signature implementation, used only in tests, to
      verify against Chen's identity and the shuffle-product identity.
- [ ] Benchmark harness: signature features vs. RSI / MACD / realized-vol /
      moving-average baselines, same walk-forward split, same cost model,
      causal execution (`position.shift(1) * returns`, no lookahead).
      Reuse the `~/Projects/quant` `multi-asset-trend` data and
      `src/backtest.py` / `src/metrics.py` patterns as the benchmark target
      (as a dependency or vendored harness — not merged into this repo).
- [ ] Write up the benchmark result honestly, including if signatures
      *don't* beat the baselines on some assets/regimes.
- [ ] README: math section (tensor algebra as a ring, log-signature as a
      free Lie algebra element, shuffle-product identity) + quickstart +
      benchmark table.

## v0.2 — depth vs. overfitting

- [ ] Multi-scale windowing (short/medium/long lookback signatures combined)
- [ ] Truncation-depth study: signature terms grow factorially; find where
      more depth stops helping and starts overfitting, on held-out data
- [ ] Regularization / feature selection over signature terms

## v0.3 — regime / change-point detection

- [ ] Signature-kernel (or signature-MMD) based unsupervised regime
      detection on the same multi-asset universe
- [ ] Compare detected regimes against the manually-defined regime windows
      already used in `multi-asset-trend` (2000-07, 2008-09, 2010-19,
      2020-22, 2023-25)

## v0.4 — real-world usage test

- [ ] Enter a live time-series competition (Optiver/Jane Street-style
      Kaggle competition) using `sigtrade` as the feature layer, as a real
      external validation of usefulness, not just a benchmark you wrote
      yourself

## Non-goals

- Not reimplementing fast signature computation from scratch (use
  `iisignature`/`signax`)
- Not a general-purpose ML library — stays scoped to financial time series
- Not merging into the `quant` monorepo — this is a standalone,
  pip-installable public package
