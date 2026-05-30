"""Job specification and presets for Linux 8x A100 PTB experiments.

Each job is a fully-described single-GPU training run. The runner reads a
JSON list of these jobs, pins each to one GPU, and writes outputs to a
unique directory derived from the job fields so concurrent runs cannot
overwrite each other's checkpoints / metrics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / "runs"
DATA_PATH = REPO_ROOT / "data" / "ptb"


def _scale_tag(scale: float) -> str:
    text = f"{scale:.6g}"
    return text.replace("-", "m").replace(".", "p")


@dataclass
class Job:
    """One training run; deterministically yields a unique output_dir."""

    group: str
    model: str
    variant: str
    data_path: Optional[str] = None
    lora_rank: int = 8
    relaxation_scale: float = 1.0
    seed: int = 1
    extra_args: List[str] = field(default_factory=list)
    architecture: Optional[str] = None
    legacy_weight_decay: Optional[float] = None
    max_epochs: Optional[int] = None
    paper_test_eval: bool = False
    tf32: bool = False
    metrics_only: bool = True
    note: str = ""

    def output_dir(self) -> Path:
        parts = [self.model, self.variant, f"r{self.lora_rank}", f"s{_scale_tag(self.relaxation_scale)}"]
        if self.architecture:
            parts.append(self.architecture)
        if self.legacy_weight_decay is not None:
            parts.append(f"wd{_scale_tag(self.legacy_weight_decay)}")
        if self.paper_test_eval:
            parts.append("papertest")
        if self.tf32:
            parts.append("tf32")
        parts.append(f"seed{self.seed}")
        name = "__".join(parts)
        return RUNS_ROOT / self.group / name

    def expected_json(self) -> Path:
        arch = self.architecture or _default_arch(self.model)
        return self.output_dir() / f"ptb_{self.model}_{arch}_{self.variant}.json"

    def is_done(self) -> bool:
        path = self.expected_json()
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        metrics = data.get("metrics", {})
        return "test_ppl" in metrics

    def cli_args(self) -> List[str]:
        args = [
            "--data_path", self.data_path or str(DATA_PATH),
            "--model", self.model,
            "--variant", self.variant,
            "--lora_rank", str(self.lora_rank),
            "--relaxation_scale", f"{self.relaxation_scale:g}",
            "--seed", str(self.seed),
            "--device", "cuda:0",
            "--output_dir", str(self.output_dir()),
        ]
        if self.architecture:
            args += ["--architecture", self.architecture]
        if self.legacy_weight_decay is not None:
            args += ["--legacy_weight_decay", f"{self.legacy_weight_decay:g}"]
        if self.max_epochs is not None:
            args += ["--max_epochs", str(self.max_epochs)]
        if self.paper_test_eval:
            args.append("--paper_test_eval")
        if self.tf32:
            args.append("--tf32")
        if self.metrics_only:
            args.append("--metrics_only")
        args += list(self.extra_args)
        return args


def _default_arch(model: str) -> str:
    # Mirrors configs.CONFIGS without importing torch.
    if model in {"bayes1500", "bayes4090"}:
        return "variational"
    if model in {"rhn830", "rhn4090"}:
        return "rhn"
    if model in {"gpt512"}:
        return "transformer"
    return "zaremba"


VARIANTS_RANK_SENSITIVE = {
    "s4",
    "s5",
    "s6",
    "s7",
    "s9",
    "s10",
    "s12",
    "s13",
    "s14",
    "s15",
    "s16",
    "s17",
    "s18",
    "s19",
    "s20",
    "s21",
}
VARIANTS_SCALE_SENSITIVE = {
    "s3",
    "s4",
    "s5",
    "s6",
    "s7",
    "s8",
    "s9",
    "s10",
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s16",
    "s17",
    "s18",
    "s19",
    "s20",
    "s21",
}
# Keep the historical screen preset fixed to S1-S13 so old job counts/results
# stay reproducible. S14-S17 have separate tied-bilateral presets below.
ALL_S_VARIANTS = [f"s{i}" for i in range(1, 14)]
BILATERAL_TIED_VARIANTS = ["s14", "s15", "s16", "s17"]
SHIFT_MULT_FOLLOWUP_VARIANTS = ["s18", "s19", "s20", "s21"]


def preset_paper_small() -> List[Job]:
    out: List[Job] = []
    for variant in ["baseline", "wt", "pr", "wt_pr"]:
        out.append(Job(group="paper-small", model="small", variant=variant, metrics_only=False))
    return out


def preset_paper_large() -> List[Job]:
    out: List[Job] = []
    for variant in ["baseline", "wt"]:
        out.append(
            Job(
                group="paper-large",
                model="large",
                variant=variant,
                tf32=True,
                paper_test_eval=True,
                metrics_only=False,
            )
        )
    return out


def preset_paper_bayes() -> List[Job]:
    out: List[Job] = []
    for variant in ["baseline", "wt"]:
        out.append(
            Job(
                group="paper-bayes",
                model="bayes1500",
                variant=variant,
                legacy_weight_decay=1e-7,
                tf32=True,
                paper_test_eval=True,
                metrics_only=False,
            )
        )
    return out


def preset_paper_rhn() -> List[Job]:
    """Press & Wolf (2017) Table 5 RHN + BD (+ WT) rows on Mikolov PTB."""
    out: List[Job] = []
    for variant in ["baseline", "wt"]:
        out.append(
            Job(
                group="paper-rhn",
                model="rhn830",
                variant=variant,
                architecture="rhn",
                legacy_weight_decay=1e-7,
                tf32=True,
                paper_test_eval=True,
                metrics_only=False,
            )
        )
    return out


def preset_screen_rhn4090() -> List[Job]:
    """Fast RHN screen for checking whether S7 helps this architecture."""
    return [
        Job(
            group="screen-rhn512",
            model="rhn4090",
            variant="wt",
            architecture="rhn",
            legacy_weight_decay=1e-7,
            tf32=True,
        ),
        Job(
            group="screen-rhn512",
            model="rhn4090",
            variant="s7",
            lora_rank=8,
            relaxation_scale=0.05,
            architecture="rhn",
            legacy_weight_decay=1e-7,
            tf32=True,
        ),
        Job(
            group="screen-rhn512",
            model="rhn4090",
            variant="s7",
            lora_rank=32,
            relaxation_scale=0.03,
            architecture="rhn",
            legacy_weight_decay=1e-7,
            tf32=True,
        ),
    ]


def preset_screen_large4090(
    ranks: Iterable[int] = (1, 2, 4, 8, 16, 32),
    scales: Iterable[float] = (0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0),
) -> List[Job]:
    out: List[Job] = []
    ranks = list(ranks)
    scales = list(scales)
    for variant in ALL_S_VARIANTS:
        if variant in {"s1", "s2"}:
            out.append(
                Job(
                    group="screen-large4090",
                    model="large4090",
                    variant=variant,
                    lora_rank=ranks[0],
                    relaxation_scale=scales[0],
                    tf32=True,
                )
            )
            continue
        eff_ranks = ranks if variant in VARIANTS_RANK_SENSITIVE else [ranks[0]]
        eff_scales = scales if variant in VARIANTS_SCALE_SENSITIVE else [scales[0]]
        for rank in eff_ranks:
            for scale in eff_scales:
                out.append(
                    Job(
                        group="screen-large4090",
                        model="large4090",
                        variant=variant,
                        lora_rank=rank,
                        relaxation_scale=scale,
                        tf32=True,
                    )
                )
    return out


def preset_screen_low_scale() -> List[Job]:
    """Extra small-scale sweep for variants that exploded at scale>=0.1."""
    low_scales = [0.01, 0.03, 0.05, 0.15, 0.2]
    candidates = ["s4", "s5", "s6", "s7", "s8", "s10", "s12", "s13"]
    out: List[Job] = []
    for variant in candidates:
        for scale in low_scales:
            out.append(
                Job(
                    group="screen-large4090-lowscale",
                    model="large4090",
                    variant=variant,
                    lora_rank=8,
                    relaxation_scale=scale,
                    tf32=True,
                )
            )
    return out


def preset_screen_bilateral_tied(
    ranks: Iterable[int] = (1, 2, 4, 8, 16, 32),
    scales: Iterable[float] = (0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0),
) -> List[Job]:
    """Fast large4090 screen for tied bilateral relaxation variants."""
    out: List[Job] = []
    for variant in BILATERAL_TIED_VARIANTS:
        for rank in ranks:
            for scale in scales:
                out.append(
                    Job(
                        group="screen-bilateral-tied",
                        model="large4090",
                        variant=variant,
                        lora_rank=rank,
                        relaxation_scale=scale,
                        tf32=True,
                    )
                )
    return out


def preset_screen_bilateral_tied_lowscale() -> List[Job]:
    """Low-scale follow-up for S14-S17 using the same grid style as Phase 3."""
    out: List[Job] = []
    for variant in BILATERAL_TIED_VARIANTS:
        for scale in (0.01, 0.03, 0.05, 0.15, 0.2):
            out.append(
                Job(
                    group="screen-bilateral-tied-lowscale",
                    model="large4090",
                    variant=variant,
                    lora_rank=8,
                    relaxation_scale=scale,
                    tf32=True,
                )
            )
    return out


# -- Phase 5/6 candidate specs: single source of truth for (variant, rank, scale, note) --
BILATERAL_TIED_CANDIDATES = [
    ("s1", 8, 0.1, "tied control"),
    ("s2", 1, 1.0, "untied control"),
    ("s14", 4, 0.3, "tied bilateral additive; matched S5 paper-confirm"),
    ("s15", 8, 0.03, "tied bilateral multiplicative; matched best S7 scale"),
    ("s15", 8, 0.05, "tied bilateral multiplicative; matched S7 diagnostic scale"),
    ("s15", 32, 0.03, "tied bilateral multiplicative; matched S7 Pareto point"),
    ("s16", 8, 0.03, "tied input additive plus output multiplicative"),
    ("s16", 8, 0.05, "tied input additive plus output multiplicative"),
    ("s17", 8, 0.03, "tied S13-style input plus output multiplicative"),
]

SHIFT_MULT_FOLLOWUP_CANDIDATES = [
    ("s18", 8, 0.007, "S12-style input shift+mult plus S7 output mult; matched S12 stable scale"),
    ("s18", 8, 0.03, "S12-style input shift+mult plus S7 output mult; matched S7 scale"),
    ("s19", 8, 0.03, "output multiplicative plus hidden shift; matched S7 r8 scale"),
    ("s19", 32, 0.03, "output multiplicative plus hidden shift; matched S7 r32 scale"),
    ("s20", 8, 0.007, "bilateral shift+mult; matched S12 stable scale"),
    ("s20", 8, 0.03, "bilateral shift+mult; matched S7 scale"),
    ("s21", 8, 0.03, "output additive plus multiplicative; matched S7 scale"),
    ("s21", 16, 0.03, "output additive plus multiplicative; higher rank probe"),
]


def preset_paper_bilateral_tied(seeds: Iterable[int] = (1, 2, 3, 4, 5)) -> List[Job]:
    """Paper-scale 5-seed diagnostic for tied bilateral variants."""
    specs = BILATERAL_TIED_CANDIDATES
    out: List[Job] = []
    for variant, rank, scale, note in specs:
        for seed in seeds:
            out.append(
                Job(
                    group="paper-bilateral-tied",
                    model="large",
                    variant=variant,
                    lora_rank=rank,
                    relaxation_scale=scale,
                    seed=seed,
                    tf32=True,
                    paper_test_eval=True,
                    metrics_only=False,
                    note=note,
                )
            )
    return out


def preset_paper_shift_mult_followup(seeds: Iterable[int] = (1, 2, 3, 4, 5)) -> List[Job]:
    """Paper-scale 5-seed follow-up for missing shift/mult output variants."""
    specs = SHIFT_MULT_FOLLOWUP_CANDIDATES
    out: List[Job] = []
    for variant, rank, scale, note in specs:
        for seed in seeds:
            out.append(
                Job(
                    group="paper-shift-mult-followup",
                    model="large",
                    variant=variant,
                    lora_rank=rank,
                    relaxation_scale=scale,
                    seed=seed,
                    tf32=True,
                    paper_test_eval=True,
                    metrics_only=False,
                    note=note,
                )
            )
    return out


def preset_paper_confirm(candidates: Iterable[str], rank: int, scale: float) -> List[Job]:
    """Run the best screening winners under paper-scale `large` config."""
    out: List[Job] = []
    for variant in candidates:
        out.append(
            Job(
                group="paper-confirm",
                model="large",
                variant=variant,
                lora_rank=rank,
                relaxation_scale=scale,
                tf32=True,
                paper_test_eval=True,
                metrics_only=False,
            )
        )
    return out


def preset_multi_seed(
    candidates: Iterable[str],
    rank: int,
    scale: float,
    seeds: Iterable[int] = (1, 2, 3, 4, 5),
    model: str = "large",
) -> List[Job]:
    out: List[Job] = []
    for variant in candidates:
        for seed in seeds:
            out.append(
                Job(
                    group="multi-seed",
                    model=model,
                    variant=variant,
                    lora_rank=rank,
                    relaxation_scale=scale,
                    seed=seed,
                    tf32=True,
                    paper_test_eval=True,
                    metrics_only=False,
                )
            )
    return out


PRESETS = {
    "paper-small": preset_paper_small,
    "paper-large": preset_paper_large,
    "paper-bayes": preset_paper_bayes,
    "paper-rhn": preset_paper_rhn,
    "screen-rhn4090": preset_screen_rhn4090,
    "screen-large4090": preset_screen_large4090,
    "screen-large4090-lowscale": preset_screen_low_scale,
    "screen-bilateral-tied": preset_screen_bilateral_tied,
    "screen-bilateral-tied-lowscale": preset_screen_bilateral_tied_lowscale,
    "paper-bilateral-tied": preset_paper_bilateral_tied,
    "paper-shift-mult-followup": preset_paper_shift_mult_followup,
}


def jobs_to_json(jobs: List[Job]) -> str:
    serial = [asdict(job) for job in jobs]
    return json.dumps(serial, indent=2)


def jobs_from_json(text: str) -> List[Job]:
    data = json.loads(text)
    return [Job(**item) for item in data]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate or inspect Job lists")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("gen", help="Generate a preset job list as JSON")
    p_gen.add_argument("preset", choices=sorted(PRESETS.keys()))
    p_gen.add_argument("--out", type=Path, required=True)

    p_count = sub.add_parser("count", help="Count jobs and how many are already done")
    p_count.add_argument("jobs", type=Path)

    args = parser.parse_args()

    if args.cmd == "gen":
        jobs = PRESETS[args.preset]()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(jobs_to_json(jobs), encoding="utf-8")
        done = sum(1 for j in jobs if j.is_done())
        print(f"wrote {len(jobs)} jobs to {args.out}  (done={done}, todo={len(jobs)-done})")
        return 0

    if args.cmd == "count":
        jobs = jobs_from_json(args.jobs.read_text(encoding="utf-8"))
        done = sum(1 for j in jobs if j.is_done())
        print(f"{len(jobs)} jobs, {done} done, {len(jobs)-done} todo")
        by_group: Dict[str, List[Job]] = {}
        for job in jobs:
            by_group.setdefault(job.group, []).append(job)
        for group, items in sorted(by_group.items()):
            d = sum(1 for j in items if j.is_done())
            print(f"  {group}: {d}/{len(items)} done")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
