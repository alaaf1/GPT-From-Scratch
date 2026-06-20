"""
Stage 1: Data download.

Downloads the raw text corpus used for training the char-level GPT model.
Designed to be a standalone DVC stage so the dataset is versioned and
re-fetched only when params.yaml's data.url changes.

Usage:
    python src/data_download.py
"""
import os
import sys
import urllib.request

import yaml


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def download_data(url: str, raw_path: str) -> None:
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    if os.path.exists(raw_path):
        print(f"[data_download] {raw_path} already exists, skipping download.")
        return

    print(f"[data_download] Downloading from {url} ...")
    try:
        urllib.request.urlretrieve(url, raw_path)
    except Exception as e:
        print(f"[data_download] ERROR: failed to download data: {e}", file=sys.stderr)
        raise

    size_kb = os.path.getsize(raw_path) / 1024
    print(f"[data_download] Saved {raw_path} ({size_kb:.1f} KB)")


def main():
    params = load_params()
    download_data(
        url=params["data"]["url"],
        raw_path=params["data"]["raw_path"],
    )


if __name__ == "__main__":
    main()
