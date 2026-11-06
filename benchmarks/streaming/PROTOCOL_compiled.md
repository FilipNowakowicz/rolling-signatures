# Protocol: compiled streaming update vs `iisignature`

Written before any compiled-path measurement, so the comparison cannot be
tuned toward the answer.

**Question.** Does a compiled implementation of the same sliding-window update
(Chen's identity plus group inverse, with automatic re-anchoring) beat batch
`iisignature` at small windows, and where is the new crossover?

**What is compared.** Depths 2 and 3; windows 10, 30, 100, 300, 600, 1200; one
channel, time-augmented and lead-lagged (`benchmarks/streaming/bench.py`
defaults). Arms, all measured in the same run:

- interpreted-Python streaming update (the reference, unchanged);
- compiled streaming update;
- batch numpy, batch `iisignature`, batch RoughPy, recomputed per tick.

The streaming arms are reported both with re-anchoring off and with the
default `refresh_every="auto"`. The compiled arm's headline number is the
default configuration, with re-anchoring on, since that is what a user gets.

**Method.** Same as the existing table: steady state after the window has
filled, per-tick cost via `_time_ticks` with the same per-cell time budget
(1.5 s, as in the committed results), same synthetic random-walk paths and
seed. Batch columns are re-measured in the same run; no numbers are carried
over from the committed table. The machine is shared with other jobs, so each
cell is repeated and the median reported; noisy runs are repeated.

**Headline metric.** The comparison against batch `iisignature` is the
headline, whichever way it goes. If the compiled update does not beat
`iisignature` at small windows, that is reported as such along with where it
does win. RoughPy and numpy multiples are secondary.

**Correctness gates (before timing is reported).** The compiled path must agree
with the Python reference and with batch results to floating-point noise, and
its drift with re-anchoring must stay at the level already reported (about
7e-15 relative). Warm-up (JIT compile) cost is reported separately and is not
included in the per-tick figures.
