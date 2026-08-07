# Full-universe confirmation study

This protocol was fixed before the full-universe scores were computed. It is
a final confirmation of the existing negative ORVP result, not a new search
over signature configurations.

## Question and population

Does the already specified depth-2 multichannel log-signature add predictive
information to the existing book + HAR feature set across every stock in the
ORVP training data?

All stocks with targets and book/trade inputs are included. Every row is a
10-minute observed segment and its following-10-minute realised-volatility
target. `time_id` is kept intact in five-fold `GroupKFold`; the competition
shuffled these identifiers, so this is grouped out-of-sample evaluation, not
chronological walk-forward validation.

## Frozen comparison

- Baseline: `book+har`.
- Challenger: `multisig+book+har`.
- Signature specification: joint log-WAP, spread, and imbalance paths;
  lead-lag and time augmentation; depth 2; 600/300/150-second suffixes.
- Primary learner: the benchmark's unchanged fixed
  `HistGradientBoostingRegressor`.
- Robustness learner: weighted ridge regression with training-fold median
  imputation, numeric standardisation, and one-hot stock effects. Its alpha is
  chosen separately inside each outer fold by three-fold grouped CV from
  `{0.01, 0.1, 1, 10, 100, 1000}`. Sample weights are proportional to
  `1 / target²`, matching RMSPE.

No other arms, depths, windows, channels, learners, or hyperparameter grids
will be scored as part of this study.

## Outputs and decision rule

For each learner, save out-of-fold predictions and report:

- overall RMSPE for both arms and their absolute and percentage difference;
- a 95% paired bootstrap interval resampling whole `time_id` groups;
- per-stock RMSPE differences, fraction of stocks improved, and the full
  per-stock table without selecting favourable subgroups;
- selected ridge alpha in every outer fold and fit time.

The challenger counts as a reliable improvement only if its primary-learner
RMSPE is lower and the paired 95% interval excludes zero. Agreement from the
ridge robustness learner strengthens interpretation but is not allowed to
replace the primary result. A null or negative result closes this ORVP feature
search; it will be reported rather than followed by configuration tuning on
the same data.
