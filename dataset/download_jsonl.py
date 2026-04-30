#!/usr/bin/env python3
"""Download MiniMind jsonl datasets into this directory."""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


REPO_ID = "jingyaogong/minimind_dataset"
DEFAULT_BASE_URL = "https://huggingface.co"

DATASET_FILES = (
    "agent_rl.jsonl",
    "agent_rl_math.jsonl",
    "dpo.jsonl",
    "lora_exam.jsonl",
    "lora_identity.jsonl",
    "lora_medical.jsonl",
    "pretrain_t2t.jsonl",
    "pretrain_t2t_mini.jsonl",
    "rlaif.jsonl",
    "sft_t2t.jsonl",
    "sft_t2t_mini.jsonl",
)


def dataset_url(base_url: str, filename: str) -> str:
    escaped_repo = urllib.parse.quote(REPO_ID, safe="/")
    escaped_file = urllib.parse.quote(filename)
    return f"{base_url.rstrip('/')}/datasets/{escaped_repo}/resolve/main/{escaped_file}"


def download_file(url: str, destination: Path) -> None:
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "MiniMind dataset downloader"})

    with urllib.request.urlopen(request) as response, tmp_path.open("wb") as output:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0

        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded * 100 / total
                print(f"\r  {destination.name}: {percent:5.1f}%", end="", flush=True)

    if total:
        print()
    tmp_path.replace(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("HF_ENDPOINT", DEFAULT_BASE_URL),
        help="Hugging Face compatible endpoint, e.g. https://hf-mirror.com",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to save jsonl files",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        choices=DATASET_FILES,
        default=list(DATASET_FILES),
        help="Subset of dataset files to download",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--list", action="store_true", help="List dataset files and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        print("\n".join(DATASET_FILES))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename in args.include:
        destination = args.output_dir / filename
        if destination.exists() and not args.force:
            print(f"skip existing: {destination}")
            continue

        url = dataset_url(args.base_url, filename)
        print(f"download: {url}")
        try:
            download_file(url, destination)
        except (urllib.error.URLError, OSError) as exc:
            print(f"failed: {filename}: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
