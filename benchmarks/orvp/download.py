"""Fetch and extract the ORVP competition data.

Requires a Kaggle API token at `~/.kaggle/kaggle.json` and acceptance of the
competition rules on the competition page -- the download 403s otherwise.

    python -m benchmarks.orvp.download            # into data/orvp
    SIGTRADE_ORVP_DIR=/mnt/orvp python -m benchmarks.orvp.download
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

from benchmarks.orvp.data import data_dir, is_available

COMPETITION = "optiver-realized-volatility-prediction"


def download(root: Path | None = None, force: bool = False) -> Path:
    root = root or data_dir()
    if is_available(root) and not force:
        print(f"data already present in {root}")
        return root

    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{COMPETITION}.zip"

    if not archive.exists() or force:
        print(f"downloading {COMPETITION} into {root} (~3.5 GB)")
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(root)],
            check=True,
        )

    print(f"extracting {archive}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)

    if not is_available(root):
        raise RuntimeError(f"extraction finished but {root} still has no train.csv / book_train.parquet")
    print(f"ready: {root}")
    return root


if __name__ == "__main__":
    sys.exit(0 if download(force="--force" in sys.argv) else 1)
