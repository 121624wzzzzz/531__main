#!/usr/bin/env python3
"""Plot FineEdu/GPT-2 S1 vs S2 gradient norms (3-seed mean) as vector SVG/PDF."""

from __future__ import annotations

import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
SEEDS = (42, 123, 2026)


def load_grad_series(variant: str) -> dict[str, list[float]]:
    by_key: dict[str, list[list[float]]] = defaultdict(list)
    steps_ref: list[int] | None = None

    for seed in SEEDS:
        path = LOGS / f"fineedu-gpt2-6b-seed{seed}" / f"{variant}_grad_stats.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        steps = [row["step"] for row in rows]
        if steps_ref is None:
            steps_ref = steps
        elif steps != steps_ref:
            raise ValueError(f"step mismatch for {variant} seed={seed}")

        by_key["embed"].append([row["embed_grad"]["norm"] for row in rows])
        by_key["unemb"].append([row["unemb_grad"]["norm"] for row in rows])
        if variant == "s1":
            by_key["shared"].append([row["shared_total_grad"]["norm"] for row in rows])

    out: dict[str, list[float]] = {"step_k": [s / 1000.0 for s in steps_ref]}
    for key, runs in by_key.items():
        n = len(runs[0])
        out[key] = [sum(runs[s][i] for s in range(len(runs))) / len(runs) for i in range(n)]
    return out


def polyline_points(xs: list[float], ys: list[float], x_map, y_map) -> str:
    return " ".join(f"{x_map(x):.2f},{y_map(y):.2f}" for x, y in zip(xs, ys))


def render_svg(s1: dict[str, list[float]], s2: dict[str, list[float]], out_path: Path) -> None:
    width, height = 1050, 420
    margin = {"l": 58, "r": 18, "t": 42, "b": 52}
    gap = 36
    panel_w = (width - margin["l"] - margin["r"] - gap) / 2
    panel_h = height - margin["t"] - margin["b"]
    y_max = 0.13
    x_max = s1["step_k"][-1]

    def panel_x0(idx: int) -> float:
        return margin["l"] + idx * (panel_w + gap)

    def x_map(x: float, x0: float) -> float:
        return x0 + (x / x_max) * panel_w

    def y_map(y: float) -> float:
        return margin["t"] + panel_h - (y / y_max) * panel_h

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: 'DejaVu Sans', 'Arial', sans-serif; fill: #222; }",
        ".axis { stroke: #444; stroke-width: 1; }",
        ".grid { stroke: #ccc; stroke-width: 0.6; }",
        ".line { fill: none; stroke-width: 1.8; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="white"/>',
    ]

    y_ticks = [i * 0.02 for i in range(7)]
    x_ticks = [i * 5 for i in range(int(x_max // 5) + 1)]

    for idx, (title, data, series) in enumerate([
        ("S1 tied baseline", s1, [
            ("input embedding path", "embed", "#1f77b4"),
            ("output unembedding path", "unemb", "#ff7f0e"),
            ("shared total gradient", "shared", "#2ca02c"),
        ]),
        ("S2 untied baseline", s2, [
            ("input embedding gradient", "embed", "#1f77b4"),
            ("output head gradient", "unemb", "#ff7f0e"),
        ]),
    ]):
        x0 = panel_x0(idx)
        x1 = x0 + panel_w
        y0 = margin["t"]
        y1 = margin["t"] + panel_h

        lines.append(f'<text x="{(x0 + x1) / 2:.1f}" y="24" text-anchor="middle" font-size="15" font-weight="600">{title}</text>')

        for yt in y_ticks:
            yy = y_map(yt)
            lines.append(f'<line class="grid" x1="{x0:.1f}" y1="{yy:.1f}" x2="{x1:.1f}" y2="{yy:.1f}"/>')
        for xt in x_ticks:
            xx = x_map(xt, x0)
            lines.append(f'<line class="grid" x1="{xx:.1f}" y1="{y0:.1f}" x2="{xx:.1f}" y2="{y1:.1f}"/>')

        lines.append(f'<line class="axis" x1="{x0:.1f}" y1="{y1:.1f}" x2="{x1:.1f}" y2="{y1:.1f}"/>')
        lines.append(f'<line class="axis" x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y1:.1f}"/>')

        if idx == 0:
            lines.append(f'<text x="{(margin["l"] + panel_w + gap / 2) / 2:.1f}" y="{height - 12}" text-anchor="middle" font-size="13">Step (k)</text>')
            lines.append(
                f'<text x="16" y="{(margin["t"] + margin["b"] + panel_h) / 2:.1f}" '
                f'transform="rotate(-90 16 {(margin["t"] + margin["b"] + panel_h) / 2:.1f})" '
                f'text-anchor="middle" font-size="13">Gradient norm</text>'
            )
        else:
            lines.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{height - 12}" text-anchor="middle" font-size="13">Step (k)</text>')

        for yt in y_ticks:
            yy = y_map(yt)
            label = f"{yt:.2f}"
            tx = x0 - 8 if idx == 0 else x0 - 8
            lines.append(f'<text x="{tx:.1f}" y="{yy + 4:.1f}" text-anchor="end" font-size="11">{label}</text>')
        for xt in x_ticks:
            xx = x_map(xt, x0)
            lines.append(f'<text x="{xx:.1f}" y="{y1 + 18:.1f}" text-anchor="middle" font-size="11">{xt:g}</text>')

        legend_y = y0 + 12
        for i, (label, key, color) in enumerate(series):
            ly = legend_y + i * 18
            lines.append(f'<line class="line" x1="{x0 + 8:.1f}" y1="{ly:.1f}" x2="{x0 + 34:.1f}" y2="{ly:.1f}" stroke="{color}"/>')
            lines.append(f'<text x="{x0 + 40:.1f}" y="{ly + 4:.1f}" font-size="10">{label}</text>')
            pts = polyline_points(
                data["step_k"],
                data[key],
                lambda x, x0=x0: x_map(x, x0),
                y_map,
            )
            lines.append(f'<polyline class="line" stroke="{color}" points="{pts}"/>')

    lines.append("</svg>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def try_pdf(svg_path: Path, pdf_path: Path) -> None:
    for cmd in (["rsvg-convert", "-f", "pdf", "-o", str(pdf_path), str(svg_path)],
                ["inkscape", str(svg_path), "--export-type=pdf", f"--export-filename={pdf_path}"]):
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"wrote {pdf_path}")
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    print("pdf skipped (rsvg-convert/inkscape not available)")


def main() -> None:
    s1 = load_grad_series("s1")
    s2 = load_grad_series("s2")
    out_svg = LOGS / "fineedu-tied-vs-untied-grad-mean.svg"
    out_pdf = LOGS / "fineedu-tied-vs-untied-grad-mean.pdf"
    render_svg(s1, s2, out_svg)
    print(f"wrote {out_svg}")
    try_pdf(out_svg, out_pdf)


if __name__ == "__main__":
    main()
