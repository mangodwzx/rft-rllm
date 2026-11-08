"""Portable comparison tables and SVG figures from verified summaries."""
from __future__ import annotations

import csv
import math
from html import escape
from pathlib import Path


COLORS = {"base": "#64748b", "rft": "#2563eb", "grpo": "#d97706"}
LABELS = {"base": "Base", "rft": "RFT", "grpo": "GRPO"}
SCOPES = (("all", "overall", "All tasks"), ("level_4_5", "level_4_5", "Level 4-5"))


def _points(values: dict | None, attempts: int, *, delta: bool = False) -> list[tuple[int, float]]:
    """Validate plotted values, including k <= N and unknown (empty) subsets."""
    points = []
    for key, value in (values or {}).items():
        try:
            k = int(key)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid pass@k key: {key!r}") from exc
        if isinstance(key, bool) or str(k) != str(key) or not 1 <= k <= attempts:
            raise ValueError(f"Plot k must be an integer in 1..{attempts}: {key!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid metric value for k={k}: {value!r}")
        if not (-1.0 if delta else 0.0) <= value <= 1.0:
            raise ValueError(f"Metric value out of range for k={k}: {value}")
        points.append((k, float(value)))
    return sorted(points)


def _text(x: float, y: float, value: str, **attributes) -> str:
    attrs = " ".join(f'{name.replace("_", "-")}="{escape(str(val), quote=True)}"' for name, val in attributes.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {attrs}>{escape(value)}</text>'


def _panel(
    x: float, y: float, title: str, tasks: int, attempts: int,
    series: list[tuple[str, str, list[tuple[int, float]]]],
    ks: list[int], *, delta_limit: float | None = None,
) -> str:
    left, top, width, height = x + 60, y + 46, 438, 214
    bottom = top + height
    parts = [f'<rect x="{x}" y="{y}" width="530" height="322" rx="8" fill="#ffffff" stroke="#e2e8f0"/>',
             _text(x + 18, y + 24, title, font_size=15, font_weight=600),
             _text(x + 512, y + 24, f"n={tasks}, N={attempts}", text_anchor="end", font_size=11, fill="#64748b")]
    minimum, maximum = (-delta_limit, delta_limit) if delta_limit is not None else (0.0, 1.0)
    log_min, log_max = (math.log2(ks[0]), math.log2(ks[-1])) if ks else (0.0, 0.0)

    def px(k: int) -> float:
        return left + width * (math.log2(k) - log_min) / (log_max - log_min) if log_max != log_min else left + width / 2

    def py(value: float) -> float:
        return bottom - height * (value - minimum) / (maximum - minimum)

    ticks = [minimum + (maximum - minimum) * i / 4 for i in range(5)]
    for value in ticks:
        yy = py(value)
        zero = delta_limit is not None and abs(value) < 1e-12
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left + width}" y2="{yy:.2f}" '
                     f'stroke="{"#94a3b8" if zero else "#e2e8f0"}"/>')
        if delta_limit is not None:
            label = f"{100 * value:+g} pp" if abs(value) > 1e-12 else "0 pp"
        else:
            label = f"{value:.0%}"
        parts.append(_text(left - 10, yy + 4, label, text_anchor="end", font_size=11, fill="#475569"))
    tick_ks = []
    for k in ks:
        if not tick_ks or k == ks[-1] or (px(k) - px(tick_ks[-1]) >= 38 and px(ks[-1]) - px(k) >= 30):
            tick_ks.append(k)
    for k in tick_ks:
        xx = px(k)
        parts.append(f'<line x1="{xx:.2f}" y1="{bottom}" x2="{xx:.2f}" y2="{bottom + 5}" stroke="#94a3b8"/>')
        parts.append(_text(xx, bottom + 20, str(k), text_anchor="middle", font_size=11, fill="#475569"))
    parts.append(_text(left + width / 2, bottom + 44, "k (log2 scale)", text_anchor="middle", font_size=12, fill="#475569"))
    if tasks == 0 or not any(points for _, _, points in series):
        parts.append(_text(left + width / 2, top + height / 2, "No tasks" if tasks == 0 else "No observations",
                           text_anchor="middle", font_size=15, fill="#64748b"))
    else:
        for label, color, points in series:
            if not points:
                continue
            coordinates = " ".join(f"{px(k):.2f},{py(value):.2f}" for k, value in points)
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.3" points="{coordinates}"/>')
            for k, value in points:
                tooltip = escape(f"{label}: k={k}, {'delta=' if delta_limit is not None else 'pass@k='}{value:.6f}")
                parts.append(f'<circle cx="{px(k):.2f}" cy="{py(value):.2f}" r="3.3" fill="{color}"><title>{tooltip}</title></circle>')
    return "\n".join(parts)


def write_comparison_artifacts(output: Path, summaries: dict, comparisons: dict) -> None:
    """Write comparison.csv and pass_at_k.svg, without fitting or inferring CIs.

    Curves use only the reported k values. Empty subsets retain blank CSV
    values and a labeled empty panel. Delta is GRPO minus RFT, in probability
    units (percentage-point differences on the figure).
    """
    output = Path(output)
    curves = {}
    deltas = {}
    all_ks = set()
    for scope, comparison_scope, _ in SCOPES:
        curves[scope] = []
        for group in ("base", "rft", "grpo"):
            summary = summaries[group] if scope == "all" else summaries[group]["subsets"][scope]
            points = _points(summary["pass_at"], summary["attempts"])
            curves[scope].append((LABELS[group], COLORS[group], points))
            all_ks.update(k for k, _ in points)
        comparison = comparisons["grpo_minus_rft"][comparison_scope]
        deltas[scope] = _points(comparison["pass_at_delta"], comparison["attempts"], delta=True)
        all_ks.update(k for k, _ in deltas[scope])

    ks = sorted(all_ks)
    csv_rows = []
    for scope, comparison_scope, _ in SCOPES:
        for name, comparison in comparisons.items():
            metrics = comparison[comparison_scope]
            points = _points(metrics["pass_at_delta"], metrics["attempts"], delta=True)
            values = dict(points)
            row_ks = sorted(values) if values else [k for k in ks if k <= metrics["attempts"]]
            for k in row_ks:
                csv_rows.append({"scope": scope, "comparison": name, "k": k,
                                 "pass_at_delta": values.get(k), "tasks": metrics["tasks"], "attempts": metrics["attempts"]})

    # Both delta panels use the same symmetric scale, making them comparable.
    max_delta = max((abs(value) for values in deltas.values() for _, value in values), default=0.0)
    delta_limit = max(.05, min(1.0, math.ceil(max_delta * 20) / 20))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="800" viewBox="0 0 1120 800" role="img" aria-labelledby="title desc">',
           '<title id="title">RFT and GRPO pass@k comparison</title>',
           '<desc id="desc">Overall and Level 4-5 pass@k curves, with GRPO minus RFT differences. No confidence intervals or significance estimates.</desc>',
           '<rect width="1120" height="800" fill="#f8fafc"/>',
           '<g font-family="Arial, Helvetica, sans-serif" fill="#0f172a">',
           _text(24, 32, "RFT / GRPO evaluation", font_size=22, font_weight=600),
           _text(24, 55, "Same test tasks and sampling protocol; training budgets are recorded separately.", font_size=12, fill="#475569")]
    for index, group in enumerate(("base", "rft", "grpo")):
        x = 820 + index * 94
        svg.append(f'<line x1="{x}" y1="43" x2="{x + 20}" y2="43" stroke="{COLORS[group]}" stroke-width="3"/>')
        svg.append(_text(x + 26, 47, LABELS[group], font_size=12))
    for index, (scope, _, title) in enumerate(SCOPES):
        summary = summaries["base"] if scope == "all" else summaries["base"]["subsets"][scope]
        x = 20 + 550 * index
        svg.append(_panel(x, 78, f"{title}: pass@k", summary["tasks"], summary["attempts"], curves[scope], ks))
        svg.append(_panel(x, 416, f"{title}: GRPO - RFT", summary["tasks"], summary["attempts"],
                          [("GRPO - RFT", "#7c3aed", deltas[scope])], ks, delta_limit=delta_limit))
    svg.append(_text(24, 762, "N = samples per task. At k = N, pass@k equals observed task coverage.", font_size=12, fill="#475569"))
    svg.append(_text(24, 782, "Delta panels show percentage-point differences. No confidence intervals or significance estimates.", font_size=12, fill="#475569"))
    svg.append('</g></svg>')

    output.mkdir(parents=True, exist_ok=True)
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scope", "comparison", "k", "pass_at_delta", "tasks", "attempts"])
        writer.writeheader()
        writer.writerows(csv_rows)
    (output / "pass_at_k.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")
