# Roadmap

Working name: **sigtrade**. Rename freely — not load-bearing.

One-liner: a `scikit-learn`-compatible library that turns financial time
series into **causal, rolling-window path-signature features**, with
O(1)-per-tick sliding-window updates derived from the group structure of the
tensor algebra (Chen's identity + group inverse), benchmarked on the Optiver
Realized Volatility Prediction dataset against reproduced top-solution
baselines.

Full background and rationale: see `CLAUDE.md`.

## Honest positioning (read before writing the README)

The claim is **not** "nobody has packaged signatures." `sktime` ships a
`SignatureTransformer` (Generalised Signature Method — Morrill, Fermanian,
Kidger, Lyons 2020). Name it in the README. The differentiation is:

1. **Causal rolling/streaming features, not panel classification.**
   sktime's transformer maps pre-segmented series → one feature vector.
   Quant pipelines need rolling features on a continuous stream with strict
   no-lookahead alignment — a different problem, and an easy one to get
   silently wrong.
2. **Streaming updates via the algebra.** Signatures are grouplike: sliding
   a window = left-multiply by the inverse of the departing segment's
   signature, right-multiply by the new increment (Chen's identity).
   O(window) recompute → O(1) amortized per tick. No existing high-level
   library exposes this as a rolling-feature API.
3. **Finance-specific defaults and honest finance benchmarks.** Lead-lag
   (quadratic variation), time augmentation, cost-aware causal evaluation,
   negative results reported.

"The missing middle layer between signature engines (`iisignature`,
`RoughPy`, `signax`) and quant practice."

## Timeline constraint (dominates everything)

Quant internship applications for summer 2027 open **July–September 2026**
and interviews roll first-come. Hard deadline: **a CV bullet with a
quantified external result by early September 2026** (v0.1 + v0.2).
v0.3+ deepens the interview talk track through autumn; it is not on the
critical path. Check Citadel/HRT/JS/IMC/Optiver portals **now** — some open
in June.

## v0.1 — core transformer, correctness as a feature (≈ weeks 1–2, by ~Aug 8)

- [ ] Week-1 spike: verify `iisignature` installs on current Python
      (it's in maintenance mode at 0.24); evaluate `RoughPy` (active,
      Lyons-lab) as alternative. Put the backend behind a single narrow
      interface either way.
- [ ] Package skeleton: `pyproject.toml`, `src/sigtrade/`, `tests/`, CI
      (GitHub Actions), MIT license, type hints. Publish to PyPI early —
      even 0.1.0; "pip-installable" should be true from the first week.
- [ ] `SignatureTransformer` (sklearn `fit`/`transform`): rolling-window
      multivariate series in → truncated signature / log-signature features
      out, **causally aligned** (feature at time t uses data ≤ t only;
      tested explicitly, not assumed).
- [ ] Path preprocessing: time augmentation, lead-lag transform, basepoint
      handling, factorial rescaling of higher-order terms.
- [ ] Pure-numpy reference implementation, used in tests only. Correctness
      oracles via property-based tests (hypothesis): Chen's identity
      (concatenation ↔ tensor product), shuffle-product identity,
      log-signature lies in the free Lie algebra (Lyndon-basis check),
      invariance under time reparametrization, backend-vs-reference
      agreement on random paths.
- [ ] `docs/math.md`: short expository note — tensor algebra as a ring,
      grouplike elements, Chen, shuffle, why lead-lag recovers quadratic
      variation. This is interview ammunition; write it as you build.

## v0.2 — headline benchmark: Optiver realized volatility (≈ weeks 3–6, by ~Sept 5)

The externally-quantified result. Vol forecasting is the task where
signatures have the strongest *theoretical* reason to help (lead-lag →
quadratic variation), and the Optiver Realized Volatility Prediction
dataset gives real order-book paths, a fixed public metric (RMSPE), and
published top-solution baselines to reproduce and compare against.

- [ ] Data pipeline for the Kaggle ORVP dataset (book/trade parquet).
- [ ] Baselines: naive last-window RV, HAR-RV-style features, and a
      reproduced top-solution-style feature set (e.g. WAP/spread/imbalance
      aggregates + GBM), same model class and CV for all arms.
- [ ] Signature arm: `sigtrade` features (lead-lag + time-augmented WAP
      paths, log-signatures) into the same model class. Report marginal
      value: baseline vs signatures-only vs baseline+signatures.
- [ ] Strict walk-forward / grouped CV (no time leakage across time_ids).
- [ ] Honesty constraint: the official private LB was re-run on future
      market data — never claim "would have ranked Nth." Claim: "on the
      competition dataset, with the competition metric, under walk-forward
      CV, signatures improved/didn't improve RMSPE by X% over reproduced
      baselines." Report the negative if it's negative.
- [ ] Truncation-depth × window-length study on this task (features grow
      exponentially in depth; find where held-out performance turns over).
      This replaces the old standalone v0.2 — it's better as part of a real
      task than in the abstract.
- [ ] README: positioning (incl. sktime comparison), quickstart, math
      summary linking `docs/math.md`, benchmark table with the honest
      numbers. → **CV bullet.**
- [ ] Distribution: publish a clean Kaggle notebook using `sigtrade` on the
      ORVP dataset. That notebook is the discovery channel — it reaches
      exactly the people who'd use the package. Target one genuine external
      user, not a star count.

## v0.3 — streaming engine (Sept–Oct, interview-season depth)

- [ ] Sliding-window signature updates via Chen's identity + group inverse
      of the departing segment. O(1) amortized per tick vs O(window)
      recompute. Benchmark the speedup honestly (there's a numerical-
      stability story here too — repeated group operations at high depth —
      investigate and document it).
- [ ] This is the "what did you actually do, mathematically?" answer: the
      group structure of the tensor algebra doing load-bearing work.

## v0.4 — the honest secondary study: daily multi-asset trend

- [ ] Signature features vs RSI/MACD/realized-vol/MA baselines on the
      `~/Projects/quant` `multi-asset-trend` universe — same feature-set
      comparison discipline as v0.2 (same learner across arms), then a
      cost-aware causal backtest (`position.shift(1) * returns`), reusing
      that repo's harness pattern (dependency or vendored, not merged).
- [ ] Expected honest outcome: signatures likely *won't* beat simple
      baselines on daily bars (low SNR, hundreds of features vs thousands
      of observations). Write that up straight, with the regime windows
      (2000-07, 2008-09, 2010-19, 2020-22, 2023-25). A rigorous negative
      result alongside a positive high-frequency result is a *stronger*
      story than either alone: "signatures earn their keep where path
      geometry matters — microstructure, not daily bars."

## v0.5 — stretch (only if time allows)

- [ ] Signature-kernel / signature-MMD regime detection, compared against
      the manual regime windows. (A full project in itself — do not let it
      onto the critical path.)
- [ ] Enter a *live* market-prediction competition with `sigtrade` as the
      feature layer if one is running at the right time (JS/Optiver/DRW-
      style comps recur, but timing is luck — treat as opportunistic, never
      load-bearing).

## Non-goals

- Not reimplementing low-level signature computation for production use
  (backends: `iisignature`/`RoughPy`; the numpy version is a test oracle)
- Not competing with sktime on panel/segment classification
- Not a general-purpose ML library — scoped to financial time series
- Not merging into the `quant` monorepo — standalone, pip-installable
- Not chasing GitHub stars — usefulness to one real quant-ML practitioner
  is the bar; stars are a side effect or nothing
