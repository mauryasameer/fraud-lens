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
        extract_archive(zf, DATA_DIR)
    zip_path.unlink()

    return DATA_DIR


def extract_archive(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract an archive only when every member remains under destination."""
    resolved_destination = destination.resolve()

    for member in archive.infolist():
        member_path = (destination / member.filename).resolve()
        if not member_path.is_relative_to(resolved_destination):
            raise ValueError(f"unsafe archive member: {member.filename}")

    for member in archive.infolist():
        archive.extract(member, destination)


if __name__ == "__main__":
    fetch_creditcard_data()
