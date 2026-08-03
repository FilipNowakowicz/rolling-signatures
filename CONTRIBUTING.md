# Contributing

Run the reproducible local checks before proposing a change:

```bash
pip install -e ".[test]"
ruff check .
pytest -q
```

The installable package is in `src/rollsig/`; `tests/` contains the unit and
property tests. The code in `benchmarks/orvp/` and `benchmarks/streaming/` is
research code that consumes the public package API.

Committed benchmark results are release artefacts. Do not regenerate result
JSON or tables as part of an unrelated code change. Regenerate them only for a
documented benchmark run, record the environment and command, and retain both
positive and negative findings. The ORVP feature search is closed by its
pre-registered stopping rule; do not reopen it casually with another feature
variant or subset.
