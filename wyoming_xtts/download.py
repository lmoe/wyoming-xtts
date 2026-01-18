import logging
import sys
from pathlib import Path

import requests

_LOGGER = logging.getLogger(__name__)

BASE_URL = "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.3/"

REQUIRED_FILES = [
    "config.json",
    "model.pth",
    "dvae.pth",
    "mel_stats.pth",
    "speakers_xtts.pth",
    "vocab.json",
]


def check_model_exists(model_path: Path) -> bool:
    return all((model_path / f).exists() for f in REQUIRED_FILES)


def _download_with_progress(url: str, dest: Path, filename: str) -> None:
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        last_percent = -1

        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)

                if total > 0:
                    percent = downloaded * 100 // total
                    if percent != last_percent:
                        sys.stdout.write(f"\r  {filename}: {downloaded / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB ({percent}%)")
                        sys.stdout.flush()
                        last_percent = percent

    if total > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()


def download_model(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename in REQUIRED_FILES:
        url = f"{BASE_URL}{filename}?download=true"
        dest = target_dir / filename

        if dest.exists():
            _LOGGER.debug("Already exists: %s", dest)
            continue

        _LOGGER.info("Downloading %s...", filename)
        try:
            _download_with_progress(url, dest, filename)
        except Exception as e:
            _LOGGER.error("Failed to download %s: %s", filename, e)
            if dest.exists():
                dest.unlink()
            raise

    _LOGGER.info("Model downloaded to %s", target_dir)
