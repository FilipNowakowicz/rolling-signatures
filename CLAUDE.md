# Development workflow

Use the repository's locked `uv` environment for all Python work:

```sh
uv sync --extra test --extra iisignature
uv run pytest -q
uv run ruff check .
```

Run scripts and modules with `uv run`, not bare `python`, `pip`, a manually
activated virtual environment, or project-local platform environment files.
Update dependencies through `pyproject.toml` with `uv add`/`uv remove`, and
commit the resulting `uv.lock`. The `.venv` directory is local and must remain
ignored.

Benchmark result files are release artefacts. Do not regenerate them during an
unrelated change; follow the relevant benchmark README when regeneration is
intentional.
