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

- [x] Week-1 spike: verify `iisignature` installs on current Python
      (it's in maintenance mode at 0.24); evaluate `RoughPy` (active,
      Lyons-lab) as alternative. Put the backend behind a single narrow
      interface either way. — `RoughPy` is the default backend, `iisignature`
      an optional one, both behind `sigtrade._backend.signature()`.
- [x] Package skeleton: `pyproject.toml`, `src/sigtrade/`, `tests/`, CI
      (GitHub Actions), MIT license, type hints. Publish to PyPI early —
      even 0.1.0; "pip-installable" should be true from the first week. —
      skeleton, tests, CI and license are in place; **PyPI publish is still
      open**, deliberately held back as a separate, explicit decision (a
      public release isn't easily undone).
- [x] `SignatureTransformer` (sklearn `fit`/`transform`): rolling-window
      multivariate series in → truncated signature / log-signature features
      out, **causally aligned** (feature at time t uses data ≤ t only;
      tested explicitly, not assumed).
- [x] Path preprocessing: time augmentation, lead-lag transform, basepoint
      handling, factorial rescaling of higher-order terms.
- [x] Pure-numpy reference implementation, used in tests only. Correctness
      oracles via property-based tests (hypothesis): Chen's identity
      (concatenation ↔ tensor product), shuffle-product identity,
      log-signature lies in the free Lie algebra (Lyndon-basis check),
      invariance under time reparametrization, backend-vs-reference
      agreement on random paths.
- [x] Expository write-up of the math — tensor algebra as a ring, grouplike
      elements, Chen, shuffle, why lead-lag recovers quadratic variation.
      Living in `docs/notes.html` (chapter-by-chapter, built alongside the
      code) rather than a separate `docs/math.md`; the two would only
      duplicate each other.

## v0.2 — headline benchmark: Optiver realized volatility (≈ weeks 3–6, by ~Sept 5)

The externally-quantified result. Vol forecasting is the task where
signatures have the strongest *theoretical* reason to help (lead-lag →
quadratic variation), and the Optiver Realized Volatility Prediction
dataset gives real order-book paths, a fixed public metric (RMSPE), and
published top-solution baselines to reproduce and compare against.

- [x] Data pipeline for the Kaggle ORVP dataset (book/trade parquet).
      `benchmarks/orvp/data.py`; the downloader fetches only the stock
      subset the benchmark evaluates on rather than the full archive.
- [x] Baselines: naive last-window RV, HAR-RV-style features, and a
      reproduced top-solution-style feature set (e.g. WAP/spread/imbalance
      aggregates + GBM), same model class and CV for all arms.
- [x] Signature arm: `sigtrade` features (lead-lag + time-augmented WAP
      paths, log-signatures) into the same model class. Report marginal
      value: baseline vs signatures-only vs baseline+signatures.
- [x] ~~Strict walk-forward~~ / grouped CV (no time leakage across
      time_ids). **Walk-forward is not achievable on this dataset** and
      that is now a documented finding rather than an open task: the
      organisers shuffled `time_id` and shipped no timestamps, so no
      chronological order is recoverable. Grouped CV on `time_id` closes
      the leak that actually dominates here — a `time_id` is one instant of
      market time across all 112 stocks, and volatility is strongly
      correlated cross-sectionally. Reported as a limitation in the README,
      `benchmarks/orvp/README.md` and `docs/notes-orvp.html` §7.3.
- [x] Honesty constraint: the official private LB was re-run on future
      market data — never claim "would have ranked Nth." **The result is
      negative and is reported as such**: on the competition training data,
      with the competition metric, under grouped CV with a fixed shared
      learner, adding depth-3 log-signatures to the best reproduced
      baseline moved RMSPE from 0.23093 to 0.23133 — 0.17% *worse*, and
      consistently so under a group bootstrap. Signatures alone reach
      0.24362 against the HAR-RV set's 0.24075. Caveat stated alongside it:
      the signature arm saw only the WAP path, while the winning baseline
      saw order-book state, so this bounds *price-path* signatures, not
      signatures as such. **Both points were revisited in v0.2.1**: the
      caveat was tested and did not change the conclusion, and these
      figures are seed-0-only — the three-subset mean is −0.99%, not
      −0.17%, so the numbers above understate the negative result.
- [x] Truncation-depth × window-length study on this task (features grow
      exponentially in depth; find where held-out performance turns over).
      This replaces the old standalone v0.2 — it's better as part of a real
      task than in the abstract. `benchmarks/orvp/study.py`.
- [x] README: positioning (incl. sktime comparison), quickstart, math
      summary linking the write-up (`docs/notes.html` +
      `docs/notes-orvp.html`, not a separate `docs/math.md` — see §5.2),
      benchmark table with the honest numbers. → **CV bullet.**
- [ ] Distribution: publish a clean Kaggle notebook using `sigtrade` on the
      ORVP dataset. That notebook is the discovery channel — it reaches
      exactly the people who'd use the package. Target one genuine external
      user, not a star count. **Written and committed**
      (`notebooks/orvp-signature-features.ipynb`); publishing is still open
      — it needs the repo public and, for the `pip install` line to work
      for anyone else, ideally the deferred PyPI release.

## v0.2.1 — closing the input-set confound, and replication (done)

v0.2's negative result had one honest objection outstanding: the signature
arm saw only the WAP path while the baseline it lost to saw order-book
state, so it was an input-set comparison rather than a feature-family one.
This closes that, and adds the replication v0.2 lacked.

- [x] Multichannel arm: depth-2 log-signatures of the joint (log-WAP,
      relative spread, depth imbalance) path over the same 600/300/150 s
      causal suffix windows, through the same pipeline, learner, GroupKFold
      splits and grouped bootstrap. Spread and imbalance come from the same
      functions the `book` arm uses, so the two arms demonstrably read the
      same quantities. Arms: `multisig`, `multisig+har`, `multisig+book`,
      `multisig+book+har` (`benchmarks/orvp/features.py`).
- [x] Replication across three pre-registered 20-stock subsets (seeds 0/1/2,
      fixed before any arm was scored), with a stop rule stated in advance:
      a win requires improvement at p < 0.05 on *every* subset
      (`benchmarks/orvp/multiseed.py`).
- [x] **Result: the confound was real and was not the explanation.**
      `multisig` beats `sig` on all three subsets (+0.70%, +0.08%, +1.78%),
      so the extra channels genuinely help — but `multisig+book+har` is
      still worse than `book+har` on all three (−0.08%, −1.32%, −0.15%;
      mean −0.52%), significant on 0 of 3. The stop rule fires.
- [x] **ORVP is closed.** No further feature engineering on this dataset:
      two independent attempts have now failed to beat aggregates of the
      same data, and a third variation would be searching for a subset
      where the number comes out right rather than testing a hypothesis.
- [x] Replication also revised v0.2's own numbers. Seed 0 — the subset v0.2
      reported — is the most favourable of the three; `sig+book+har` vs
      `book+har` is −0.17% there against a −0.99% mean. The README now
      leads with the three-subset table.

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
