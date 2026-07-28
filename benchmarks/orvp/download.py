"""Fetch the ORVP competition data.

Two prerequisites, both on kaggle.com:

1. An API token. The current Kaggle CLI wants the newer prefixed token in
   `~/.kaggle/access_token` (or the `KAGGLE_API_TOKEN` environment
   variable); the older `~/.kaggle/kaggle.json` username/key pair is not
   what it reads.
2. Acceptance of the competition rules, on the competition page. Listing
   files and reading the leaderboard work without it -- only *downloads*
   403 -- so an unaccepted-rules failure looks confusingly like a broken
   token. It isn't.

Two modes. The subset download is the default because the benchmark runs on
a stock subset anyway, and pulling 20 stocks costs a few hundred megabytes
against the full archive's several gigabytes:

    python -m benchmarks.orvp.download                  # train.csv + 20 stocks
    python -m benchmarks.orvp.download --stocks 40      # a larger subset
    python -m benchmarks.orvp.download --all            # the full archive
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

from benchmarks.orvp.data import data_dir, is_available, load_targets, select_stocks

COMPETITION = "optiver-realized-volatility-prediction"
_PAGE_SIZE = 200


def _kaggle_executable() -> str:
    """The `kaggle` CLI, preferring the one beside the running interpreter.

    `pip install kaggle` into a virtualenv puts the CLI in that venv's bin
    directory, which is not necessarily on PATH when the module is run as
    `/path/to/venv/bin/python -m benchmarks.orvp.download`.
    """
    local = Path(sys.executable).parent / "kaggle"
    return str(local) if local.exists() else "kaggle"


def _kaggle(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([_kaggle_executable(), *args], check=True, capture_output=True, text=True)


def _fetch_file(remote: str, root: Path) -> None:
    """Download one competition file, preserving its path under `root`.

    The CLI writes the file into `-p` as a bare name, or as a zip when it
    decides to compress; both are normalised here so the on-disk layout
    always matches the competition archive's.
    """
    destination = root / remote
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    _kaggle("competitions", "download", "-c", COMPETITION, "-f", remote, "-p", str(destination.parent), "-q")

    landed = destination.parent / Path(remote).name
    zipped = destination.parent / (Path(remote).name + ".zip")
    if zipped.exists():
        with zipfile.ZipFile(zipped) as zf:
            zf.extractall(destination.parent)
        zipped.unlink()
    elif landed != destination and landed.exists():
        landed.rename(destination)


def list_files(root_prefix: str = "") -> list[str]:
    """Every file in the competition, as slash-separated remote paths.

    Paged, because the competition has several hundred files and the API
    caps a page at 200.
    """
    names: list[str] = []
    token: str | None = None
    while True:
        args = ["competitions", "files", COMPETITION, "--page-size", str(_PAGE_SIZE), "-v"]
        if token:
            args += ["--page-token", token]
        result = _kaggle(*args)
        token = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("Next Page Token = "):
                token = line.removeprefix("Next Page Token = ").strip()
            elif line and not line.startswith("name,"):
                name = line.split(",")[0]
                if name.startswith(root_prefix):
                    names.append(name)
        if not token:
            return names


def download_subset(n_stocks: int, root: Path | None = None, seed: int = 0) -> Path:
    """Fetch `train.csv` plus the book and trade files for a stock subset.

    The subset has to be chosen *after* `train.csv` lands, because
    `select_stocks` draws from the stock ids that actually appear there --
    so this is necessarily two round trips, not one.
    """
    root = root or data_dir()
    root.mkdir(parents=True, exist_ok=True)

    print("fetching train.csv")
    _fetch_file("train.csv", root)

    stocks = select_stocks(load_targets(root), n_stocks, seed=seed)
    print(f"stock subset (seed {seed}): {stocks}")

    wanted = {f"book_train.parquet/stock_id={s}/" for s in stocks}
    wanted |= {f"trade_train.parquet/stock_id={s}/" for s in stocks}
    remote_files = [name for name in list_files() if any(name.startswith(prefix) for prefix in wanted)]
    print(f"fetching {len(remote_files)} parquet files")
    for i, remote in enumerate(remote_files, start=1):
        _fetch_file(remote, root)
        print(f"  [{i}/{len(remote_files)}] {remote}")

    if not is_available(root):
        raise RuntimeError(f"download finished but {root} has no train.csv / book_train.parquet")
    print(f"ready: {root}")
    return root


def download_all(root: Path | None = None, force: bool = False) -> Path:
    """Fetch and extract the whole competition archive."""
    root = root or data_dir()
    if is_available(root) and not force:
        print(f"data already present in {root}")
        return root

    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{COMPETITION}.zip"
    if not archive.exists() or force:
        print(f"downloading the full {COMPETITION} archive into {root}")
        subprocess.run([_kaggle_executable(), "competitions", "download", "-c", COMPETITION, "-p", str(root)], check=True)

    print(f"extracting {archive}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)

    if not is_available(root):
        raise RuntimeError(f"extraction finished but {root} has no train.csv / book_train.parquet")
    print(f"ready: {root}")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stocks", type=int, default=20, help="how many stocks to fetch")
    parser.add_argument("--seed", type=int, default=0, help="seed for the stock subset")
    parser.add_argument("--all", action="store_true", help="fetch the entire competition archive instead")
    parser.add_argument("--force", action="store_true", help="re-download even if data is present")
    args = parser.parse_args()

    try:
        if args.all:
            download_all(force=args.force)
        else:
            download_subset(args.stocks, seed=args.seed)
    except subprocess.CalledProcessError as error:
        message = (error.stderr or "") + (error.stdout or "")
        if "403" in message:
            raise SystemExit(
                "Kaggle returned 403 on the download.\n\n"
                "The token is probably fine -- listing files and the leaderboard work without\n"
                "rules acceptance, but downloads do not. Accept the competition rules at\n"
                f"  https://www.kaggle.com/competitions/{COMPETITION}/rules\n"
                "and run this again."
            ) from error
        raise


if __name__ == "__main__":
    sys.exit(main())
