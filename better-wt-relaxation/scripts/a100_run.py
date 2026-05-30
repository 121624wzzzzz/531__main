"""Run a list of A100 PTB jobs in parallel, one GPU per job.

Usage:
    python scripts/a100_run.py jobs.json \
        --gpus 1,2,3,4,5,6 \
        [--max_parallel 6] \
        [--dry_run] \
        [--force] \
        [--retries 1] \
        [--include_groups paper-small,paper-large]

Each running job has CUDA_VISIBLE_DEVICES set so the train script always
sees the assigned GPU as cuda:0. Stdout/stderr are tee'd into
<output_dir>/run.log so failures can be debugged after the batch ends.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from a100_jobs import Job, REPO_ROOT, jobs_from_json

PYTHON = os.environ.get("REPRO_PYTHON", "/home/wz/anaconda3/envs/torch24/bin/python")
TRAIN_SCRIPT = REPO_ROOT / "repro_pytorch" / "train_ptb.py"


@dataclass
class JobResult:
    job: Job
    gpu: int
    rc: int
    elapsed: float
    skipped: bool = False
    error: str = ""


def _parse_gpus(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _filter_jobs(jobs: List[Job], include_groups: Optional[Set[str]], force: bool) -> List[Job]:
    out = []
    for job in jobs:
        if include_groups and job.group not in include_groups:
            continue
        if not force and job.is_done():
            continue
        out.append(job)
    return out


def _run_one(job: Job, gpu: int, dry_run: bool) -> JobResult:
    out_dir = job.output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    cmd = [PYTHON, str(TRAIN_SCRIPT), *job.cli_args()]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"

    start = time.perf_counter()
    if dry_run:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"[DRY_RUN gpu={gpu}] " + " ".join(cmd) + "\n")
        return JobResult(job=job, gpu=gpu, rc=0, elapsed=0.0, skipped=True)

    with log_path.open("ab", buffering=0) as fh:
        fh.write(f"\n==== START gpu={gpu} ts={time.strftime('%Y-%m-%dT%H:%M:%S')} ====\n".encode())
        fh.write((" ".join(cmd) + "\n").encode())
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read(4096), b""):
            fh.write(chunk)
        proc.wait()
        rc = proc.returncode
        elapsed = time.perf_counter() - start
        fh.write(f"\n==== END gpu={gpu} rc={rc} elapsed={elapsed:.1f}s ====\n".encode())
    return JobResult(job=job, gpu=gpu, rc=rc, elapsed=elapsed)


def _worker(
    gpu: int,
    job_q: "queue.Queue[Optional[Job]]",
    results: List[JobResult],
    results_lock: threading.Lock,
    dry_run: bool,
    retries: int,
    progress: Dict[str, int],
    progress_lock: threading.Lock,
) -> None:
    while True:
        job = job_q.get()
        if job is None:
            job_q.task_done()
            return
        attempt = 0
        last_err = ""
        rc = 1
        elapsed = 0.0
        skipped = False
        while attempt <= retries:
            res = _run_one(job, gpu, dry_run)
            rc = res.rc
            elapsed = res.elapsed
            skipped = res.skipped
            if rc == 0 or skipped:
                break
            attempt += 1
            last_err = f"rc={rc} attempt={attempt}"
        with results_lock:
            results.append(
                JobResult(
                    job=job, gpu=gpu, rc=rc, elapsed=elapsed, skipped=skipped, error=last_err
                )
            )
        with progress_lock:
            progress["done"] += 1
            d = progress["done"]
            t = progress["total"]
            tag = "DRY" if dry_run else ("OK" if rc == 0 else "FAIL")
            print(
                f"[{d:>3}/{t}] gpu={gpu} {tag} elapsed={elapsed:5.1f}s  "
                f"{job.group}/{job.model}/{job.variant} r{job.lora_rank} "
                f"s{job.relaxation_scale} seed{job.seed}",
                flush=True,
            )
        job_q.task_done()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--gpus", default="1,2,3,4,5,6", help="Comma-separated GPU ids")
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=None,
        help="Max concurrent jobs (default: number of GPUs)",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if JSON already has test_ppl")
    parser.add_argument(
        "--retries", type=int, default=0, help="Retry failing jobs this many times (same GPU)"
    )
    parser.add_argument(
        "--include_groups", default=None, help="Only run jobs whose group is in this comma list"
    )
    parser.add_argument("--summary", type=Path, default=None, help="Where to write a JSON run summary")
    args = parser.parse_args()

    raw_jobs = jobs_from_json(args.jobs.read_text(encoding="utf-8"))
    include = set(args.include_groups.split(",")) if args.include_groups else None
    jobs = _filter_jobs(raw_jobs, include, args.force)
    gpus = _parse_gpus(args.gpus)
    max_parallel = args.max_parallel or len(gpus)
    if max_parallel > len(gpus):
        max_parallel = len(gpus)

    print(
        f"runner: total_jobs={len(raw_jobs)}  to_run={len(jobs)}  "
        f"gpus={gpus}  parallel={max_parallel}  dry_run={args.dry_run}",
        flush=True,
    )
    if not jobs:
        print("nothing to do.", flush=True)
        return 0

    job_q: "queue.Queue[Optional[Job]]" = queue.Queue()
    for job in jobs:
        job_q.put(job)
    for _ in range(max_parallel):
        job_q.put(None)

    results: List[JobResult] = []
    results_lock = threading.Lock()
    progress = {"done": 0, "total": len(jobs)}
    progress_lock = threading.Lock()

    threads = []
    for gpu in gpus[:max_parallel]:
        t = threading.Thread(
            target=_worker,
            args=(gpu, job_q, results, results_lock, args.dry_run, args.retries, progress, progress_lock),
            daemon=False,
        )
        t.start()
        threads.append(t)

    job_q.join()
    for t in threads:
        t.join()

    n_ok = sum(1 for r in results if r.rc == 0)
    n_fail = sum(1 for r in results if r.rc != 0 and not r.skipped)
    n_dry = sum(1 for r in results if r.skipped)
    total_sec = sum(r.elapsed for r in results)
    print(
        f"runner finished: ok={n_ok} fail={n_fail} dry={n_dry} total_wall_sec_unweighted={total_sec:.1f}",
        flush=True,
    )

    if args.summary:
        out = {
            "ok": n_ok,
            "fail": n_fail,
            "dry": n_dry,
            "results": [
                {
                    "group": r.job.group,
                    "model": r.job.model,
                    "variant": r.job.variant,
                    "rank": r.job.lora_rank,
                    "scale": r.job.relaxation_scale,
                    "seed": r.job.seed,
                    "gpu": r.gpu,
                    "rc": r.rc,
                    "elapsed": r.elapsed,
                    "output_dir": str(r.job.output_dir()),
                    "skipped": r.skipped,
                    "error": r.error,
                }
                for r in results
            ],
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(out, indent=2), encoding="utf-8")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
