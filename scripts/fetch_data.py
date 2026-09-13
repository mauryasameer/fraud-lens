from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "data"
KAGGLE_CREDENTIALS = Path.home() / ".kaggle" / "kaggle.json"


def fetch_creditcard_data() -> Path:
    if not KAGGLE_CREDENTIALS.exists():
        print(
            f"error: Kaggle API credentials not found at {KAGGLE_CREDENTIALS}.\n"
            "Set up your Kaggle API token: https://www.kaggle.com/docs/api#authentication",
            file=sys.stderr,
        )
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "mlg-ulb/creditcardfraud", "-p", str(DATA_DIR)],
        check=True,
    )

    zip_path = DATA_DIR / "creditcardfraud.zip"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink()

    return DATA_DIR


if __name__ == "__main__":
    fetch_creditcard_data()
