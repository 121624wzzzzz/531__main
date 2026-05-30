#!/usr/bin/env python3
"""Run static E/U geometry audit on extracted model matrices."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "geometry"))

from spectral import audit_model_extract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit embedding/unembedding geometry.")
    parser.add_argument(
        "--extract",
        type=Path,
        required=True,
        help="Path to a .safetensors extract or .npy matrix bundle.",
    )
    parser.add_argument("--info-json", type=Path, default=None, help="Optional *.info.json metadata.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/geometry"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    row = audit_model_extract(args.extract, info_json=args.info_json)
    json_path = args.output_dir / "geometry_audit.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    csv_path = args.output_dir / "geometry_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2))
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
