"""Aggregate run JSONs under runs/ into a tidy table.

Usage:
    python scripts/a100_summarize.py [--root runs] [--groups paper-small,...]
        [--csv summary.csv] [--md summary.md]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "repro_pytorch"))
try:
    from variants import actual_extra_params  # type: ignore
except Exception:  # pragma: no cover
    def actual_extra_params(*_args, **_kwargs):
        return 0


@dataclass
class RunRow:
    group: str
    model: str
    architecture: str
    variant: str
    rank: int
    scale: float
    seed: int
    extra_params: int
    best_valid_ppl: float
    best_valid_epoch: int
    final_valid_ppl: float
    test_ppl: float
    epochs: int
    train_sec: float
    eval_sec: float
    total_sec: float
    avg_train_wps: float
    paper_test_eval: bool
    tf32: bool
    legacy_weight_decay: float
    output_dir: str
    json_path: str

    @classmethod
    def from_json(cls, group: str, path: Path) -> Optional["RunRow"]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        metrics = data.get("metrics", {})
        if "test_ppl" not in metrics:
            return None
        args = data.get("args", {})
        config = data.get("config", {})
        valid_pairs = []
        epoch = 0
        for key, val in metrics.items():
            if key.startswith("epoch_") and key.endswith("_valid_ppl"):
                n = int(key.split("_")[1])
                valid_pairs.append((n, float(val)))
                epoch = max(epoch, n)
        valid_pairs.sort()
        if valid_pairs:
            best_epoch, best_valid = min(valid_pairs, key=lambda kv: kv[1])
            final_valid = valid_pairs[-1][1]
        else:
            best_epoch = 0
            best_valid = float("nan")
            final_valid = float("nan")
        variant = args.get("variant", "")
        vocab = int(config.get("vocab_size", 10000) or 10000)
        hidden = int(config.get("hidden_size", 0) or 0)
        rank = int(args.get("lora_rank", 0) or 0)
        try:
            extra_params = actual_extra_params(variant, vocab, hidden, rank)
        except Exception:
            extra_params = 0
        return cls(
            group=group,
            model=args.get("model", ""),
            architecture=data.get("architecture", ""),
            variant=args.get("variant", ""),
            rank=int(args.get("lora_rank", 0) or 0),
            scale=float(args.get("relaxation_scale", 0.0) or 0.0),
            seed=int(args.get("seed", 0) or 0),
            extra_params=extra_params,
            best_valid_ppl=best_valid,
            best_valid_epoch=best_epoch,
            final_valid_ppl=final_valid,
            test_ppl=float(metrics.get("test_ppl", float("nan"))),
            epochs=epoch,
            train_sec=float(metrics.get("timing_train_sec", 0.0)),
            eval_sec=float(metrics.get("timing_eval_sec", 0.0)),
            total_sec=float(metrics.get("timing_total_sec", 0.0)),
            avg_train_wps=float(metrics.get("timing_avg_train_wps", 0.0)),
            paper_test_eval=bool(args.get("paper_test_eval", False)),
            tf32=bool(args.get("tf32", False)),
            legacy_weight_decay=float(args.get("legacy_weight_decay") or 0.0),
            output_dir=str(path.parent),
            json_path=str(path),
        )


def collect(root: Path, groups: Optional[List[str]]) -> List[RunRow]:
    """Scan ``root`` recursively for ``ptb_*.json`` files.

    ``groups`` can name phase dirs (``01-paper-repro``), specific run groups
    (``paper-large``), or both — any path whose *parent* matches is included.
    """
    rows: List[RunRow] = []
    for json_path in sorted(root.rglob("ptb_*.json")):
        # Path: runs/<phase>/<group>/<unique_run_dir>/ptb_*.json
        rel = json_path.relative_to(root)
        phase = rel.parts[0] if len(rel.parts) >= 4 else ""
        group = rel.parts[1] if len(rel.parts) >= 4 else rel.parts[0]
        if groups:
            if group not in groups and phase not in groups:
                continue
        row = RunRow.from_json(group, json_path)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: (r.group, r.model, r.variant, r.scale, r.rank, r.seed))
    return rows


def write_csv(rows: List[RunRow], path: Path) -> None:
    fields = [
        "group",
        "model",
        "architecture",
        "variant",
        "rank",
        "scale",
        "seed",
        "extra_params",
        "best_valid_ppl",
        "best_valid_epoch",
        "final_valid_ppl",
        "test_ppl",
        "epochs",
        "train_sec",
        "eval_sec",
        "total_sec",
        "avg_train_wps",
        "paper_test_eval",
        "tf32",
        "legacy_weight_decay",
        "output_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: getattr(row, f) for f in fields})


def _fmt(val, nd: int = 2) -> str:
    if val is None:
        return "nan"
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return "nan"
    return f"{val:.{nd}f}"


def write_markdown(rows: List[RunRow], path: Path) -> None:
    lines = [
        "| group | model | variant | rank | scale | seed | extra_params | best_valid | best_ep | test | epochs | train_sec |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            "| {g} | {m} | {v} | {r} | {s} | {sd} | {xp} | {bv} | {be} | {tp} | {ep} | {ts} |".format(
                g=r.group,
                m=r.model,
                v=r.variant,
                r=r.rank,
                s=_fmt(r.scale, 3),
                sd=r.seed,
                xp=r.extra_params,
                bv=_fmt(r.best_valid_ppl),
                be=r.best_valid_epoch,
                tp=_fmt(r.test_ppl),
                ep=r.epochs,
                ts=_fmt(r.train_sec, 1),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_per_variant(rows: List[RunRow], metric: str = "best_valid_ppl") -> List[RunRow]:
    by: Dict[str, RunRow] = {}
    for row in rows:
        key = f"{row.model}/{row.variant}"
        cur = by.get(key)
        cur_val = getattr(cur, metric) if cur else float("inf")
        new_val = getattr(row, metric)
        if math.isnan(new_val):
            continue
        if new_val < cur_val:
            by[key] = row
    return sorted(by.values(), key=lambda r: (r.model, getattr(r, metric)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent / "runs")
    parser.add_argument("--groups", default=None, help="Comma-separated subdirs to scan")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--md", type=Path, default=None)
    parser.add_argument("--best_md", type=Path, default=None,
                        help="Also write a 'best per (model, variant)' markdown table")
    parser.add_argument("--best_metric", default="best_valid_ppl",
                        choices=["best_valid_ppl", "test_ppl"])
    args = parser.parse_args()
    groups = args.groups.split(",") if args.groups else None
    rows = collect(args.root, groups)
    print(f"collected {len(rows)} completed runs")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, args.csv)
        print(f"wrote {args.csv}")
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(rows, args.md)
        print(f"wrote {args.md}")
    if args.best_md:
        best_rows = best_per_variant(rows, metric=args.best_metric)
        args.best_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(best_rows, args.best_md)
        print(f"wrote {args.best_md}  ({len(best_rows)} best rows by {args.best_metric})")
    if not args.csv and not args.md and not args.best_md:
        print("\n".join(f"{r.group}/{r.model}/{r.variant} r{r.rank} s{r.scale} seed{r.seed} "
                        f"best_valid={_fmt(r.best_valid_ppl)} test={_fmt(r.test_ppl)}" for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
