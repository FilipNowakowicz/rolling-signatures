# Status: compiled streaming branch (temporary note)

Done: Numba kernel (`src/rollsig/_compiled.py`), `engine=` option on
`StreamingSignature` / `rolling_signature` (default "auto": Numba if installed,
else numpy), `compiled` extra, CI runs with and without Numba,
`tests/test_compiled.py` (228 tests pass with Numba; 154 pass, 1 skipped without),
`bench.py` extended (`--compiled`, `--repeats`, warm-up cost, compiled drift).

Not done: the real benchmark run (smoke run only, nothing committed), result
files, README / benchmark README updates, any change to the `method="auto"`
thresholds in `transformer.py` (still calibrated on the interpreted engine).
No compiled-path speed claim is established by this commit.
