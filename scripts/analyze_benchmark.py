#
# Copyright © 2015 The Gravitee team (http://gravitee.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=2.2",
#     "matplotlib>=3.8",
#     "typer>=0.15",
#     "rich>=13",
# ]
# ///
"""Analyze JMH benchmark output from one or more CPU-pinned runs.

Each input JSON is expected to follow the JMH `-rf json` schema. The filename's
trailing digit (e.g. ``bench-cpu4.json``) is used to label the CPU pinning.
Pass multiple JSONs to produce CPU-scaling comparisons.

Outputs (written under ``--out``):
  - ``summary.csv``  : flat, tidy DataFrame of every (profile, textLength,
                       batchSize, entityCount, cpus) combination with
                       latency / per-text latency / throughput.
  - ``report.md``    : human-readable markdown summary (top/bottom configs,
                       per-profile breakdown, CPU scaling table).
  - ``plots/*.png``  : latency vs batch size, throughput vs CPU count, and
                       (when multiple CPU configs are present) scaling
                       efficiency from 1 → N cores.

Usage:
    uv run scripts/analyze_benchmark.py results/bench-cpu1.json results/bench-cpu2.json results/bench-cpu4.json
    uv run scripts/analyze_benchmark.py results/bench-cpu4.json --out reports/
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(add_completion=False)


def _cpus_from_filename(path: Path) -> int:
    """Extract the CPU count from a filename like ``bench-cpu4.json``.

    Returns 0 if no trailing digit is found — the caller can treat that as
    "unknown CPU config" and label runs by filename instead.
    """
    m = re.search(r"cpu(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def _load_one(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text())
    if not raw:
        raise ValueError(f"Empty benchmark JSON: {path}")
    cpus = _cpus_from_filename(path)
    rows = []
    for entry in raw:
        p = entry.get("params", {})
        m = entry.get("primaryMetric", {})
        batch_size = int(p.get("batchSize", 1))
        score = m.get("score")
        rows.append({
            "profile": p.get("profile"),
            "textLength": p.get("textLength"),
            "batchSize": batch_size,
            "entityCount": int(p.get("entityCount", 0)),
            "cpus": cpus,
            "source": path.name,
            "score_ms": score,
            "score_err_ms": m.get("scoreError"),
            "per_text_ms": score / batch_size if score is not None else None,
            "throughput_tps": (batch_size * 1000.0 / score) if score else None,
        })
    return pd.DataFrame(rows)


def _load_all(paths: list[Path]) -> pd.DataFrame:
    frames = [_load_one(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    # Preferred column order
    return df[
        [
            "cpus",
            "profile",
            "textLength",
            "batchSize",
            "entityCount",
            "score_ms",
            "score_err_ms",
            "per_text_ms",
            "throughput_tps",
            "source",
        ]
    ]


# ── Reporting helpers ──────────────────────────────────────────────────────


_LENGTH_ORDER = ["tiny", "short", "medium", "long"]


def _length_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["textLength"] = pd.Categorical(df["textLength"], categories=_LENGTH_ORDER, ordered=True)
    return df.sort_values(["profile", "textLength", "batchSize", "entityCount", "cpus"])


def _markdown_table(df: pd.DataFrame, cols: list, floatfmt: str = "{:,.2f}") -> str:
    headers = [str(c) for c in cols]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _scaling_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot so each row is one config, columns are CPU counts.

    Adds a ``speedup_NxM`` column comparing the lowest CPU count to each higher
    one. Returns wide-format DataFrame.
    """
    keys = ["profile", "textLength", "batchSize", "entityCount"]
    pivot = df.pivot_table(index=keys, columns="cpus", values="score_ms", aggfunc="mean")
    pivot = pivot.sort_index()
    cpu_cols = sorted(pivot.columns)
    if len(cpu_cols) >= 2:
        base = cpu_cols[0]
        for c in cpu_cols[1:]:
            pivot[f"speedup_{c}/{base}"] = pivot[base] / pivot[c]
    return pivot.reset_index()


# ── Plots ──────────────────────────────────────────────────────────────────


def _plot_latency_vs_batch(df: pd.DataFrame, out: Path) -> None:
    df = _length_sort(df)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
    cpu_values = sorted(df["cpus"].unique())
    for r, profile in enumerate(["base", "pii"]):
        for c, length in enumerate(_LENGTH_ORDER):
            ax = axes[r, c]
            sub = df[(df["profile"] == profile) & (df["textLength"] == length)]
            for cpu in cpu_values:
                d = sub[sub["cpus"] == cpu].groupby("batchSize")["score_ms"].mean()
                if not d.empty:
                    ax.plot(d.index, d.values, marker="o", label=f"{cpu} cpu")
            ax.set_title(f"{profile} / {length}")
            ax.set_xlabel("batch size")
            ax.set_ylabel("latency (ms/op)")
            ax.grid(alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Latency vs batch size — by profile × text length", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_throughput_vs_cpus(df: pd.DataFrame, out: Path) -> None:
    df = _length_sort(df)
    cpu_values = sorted(df["cpus"].unique())
    if len(cpu_values) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, profile in zip(axes, ["base", "pii"], strict=False):
        sub = df[df["profile"] == profile]
        for length in _LENGTH_ORDER:
            d = sub[sub["textLength"] == length].groupby("cpus")["throughput_tps"].mean()
            if not d.empty:
                ax.plot(d.index, d.values, marker="o", label=length)
        ax.set_title(f"throughput vs CPU count — {profile}")
        ax.set_xlabel("CPUs")
        ax.set_ylabel("throughput (texts/s)")
        ax.set_xticks(cpu_values)
        ax.grid(alpha=0.3)
        ax.legend(title="textLength", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_scaling_efficiency(df: pd.DataFrame, out: Path) -> None:
    cpu_values = sorted(df["cpus"].unique())
    if len(cpu_values) < 2:
        return
    base_cpu = cpu_values[0]
    df = _length_sort(df)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, profile in zip(axes, ["base", "pii"], strict=False):
        sub = df[df["profile"] == profile]
        for length in _LENGTH_ORDER:
            d = sub[sub["textLength"] == length].groupby("cpus")["score_ms"].mean()
            if d.empty or base_cpu not in d.index:
                continue
            speedup = d[base_cpu] / d
            ax.plot(speedup.index, speedup.values, marker="o", label=length)
        ax.plot(cpu_values, [c / base_cpu for c in cpu_values], "k--", label="ideal", alpha=0.4)
        ax.set_title(f"speedup vs {base_cpu}-cpu baseline — {profile}")
        ax.set_xlabel("CPUs")
        ax.set_ylabel(f"speedup (relative to {base_cpu} cpu)")
        ax.set_xticks(cpu_values)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────


JOIN_KEYS = ["profile", "textLength", "batchSize", "entityCount", "cpus"]


def _overlay_latency(df: pd.DataFrame, label_col: str, out_path: Path) -> None:
    """One subplot per (profile × textLength); one line per run label, x = batchSize."""
    df = _length_sort(df)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
    labels = sorted(df[label_col].unique())
    for r, profile in enumerate(["base", "pii"]):
        for c, length in enumerate(_LENGTH_ORDER):
            ax = axes[r, c]
            sub = df[(df["profile"] == profile) & (df["textLength"] == length)]
            for lbl in labels:
                d = sub[sub[label_col] == lbl].groupby("batchSize")["score_ms"].mean()
                if not d.empty:
                    ax.plot(d.index, d.values, marker="o", label=lbl)
            ax.set_title(f"{profile} / {length}")
            ax.set_xlabel("batch size")
            ax.set_ylabel("latency (ms/op)")
            ax.grid(alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("A/B comparison — latency vs batch size", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


@app.command()
def analyze(
    inputs: Annotated[
        list[Path],
        typer.Argument(help="JMH JSON files (e.g. results/bench-cpu1.json results/bench-cpu4.json)"),
    ],
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("results"),
) -> None:
    """Flatten + summarize JMH benchmark output."""
    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "plots"
    plots_dir.mkdir(exist_ok=True)

    df = _load_all([Path(p) for p in inputs])
    df = _length_sort(df)

    # 1) Flat CSV
    csv_path = out / "summary.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"[green]✓[/green] wrote {csv_path}  ({len(df)} rows)")

    # 2) Markdown report
    md = []
    md.append("# GLiNER4j benchmark — analysis\n")
    md.append(f"Loaded {len(inputs)} run(s), {len(df)} measurements.\n")
    md.append(f"CPU configs: {sorted(df['cpus'].unique())}\n")

    md.append("\n## Fastest 10 configurations (by per-text ms)\n")
    md.append(
        _markdown_table(
            df.sort_values("per_text_ms").head(10)[
                ["cpus", "profile", "textLength", "batchSize", "entityCount", "score_ms", "per_text_ms", "throughput_tps"]
            ],
            cols=["cpus", "profile", "textLength", "batchSize", "entityCount", "score_ms", "per_text_ms", "throughput_tps"],
        )
    )

    md.append("\n\n## Slowest 10 configurations (by per-text ms)\n")
    md.append(
        _markdown_table(
            df.sort_values("per_text_ms", ascending=False).head(10)[
                ["cpus", "profile", "textLength", "batchSize", "entityCount", "score_ms", "per_text_ms", "throughput_tps"]
            ],
            cols=["cpus", "profile", "textLength", "batchSize", "entityCount", "score_ms", "per_text_ms", "throughput_tps"],
        )
    )

    # CPU scaling
    if len(df["cpus"].unique()) >= 2:
        md.append("\n\n## CPU scaling — latency (ms/op) per config\n")
        scaling = _scaling_table(df)
        cols = list(scaling.columns)
        md.append(_markdown_table(scaling, cols=cols))

    report_path = out / "report.md"
    report_path.write_text("\n".join(md))
    console.print(f"[green]✓[/green] wrote {report_path}")

    # 3) Plots
    _plot_latency_vs_batch(df, plots_dir / "latency_vs_batch.png")
    console.print(f"[green]✓[/green] wrote {plots_dir / 'latency_vs_batch.png'}")
    if len(df["cpus"].unique()) >= 2:
        _plot_throughput_vs_cpus(df, plots_dir / "throughput_vs_cpus.png")
        console.print(f"[green]✓[/green] wrote {plots_dir / 'throughput_vs_cpus.png'}")
        _plot_scaling_efficiency(df, plots_dir / "scaling_efficiency.png")
        console.print(f"[green]✓[/green] wrote {plots_dir / 'scaling_efficiency.png'}")

    # Console summary
    console.print()
    table = Table(title="Run summary")
    table.add_column("CPUs", justify="right")
    table.add_column("rows")
    table.add_column("min ms/op", justify="right")
    table.add_column("median ms/op", justify="right")
    table.add_column("max ms/op", justify="right")
    for cpus, sub in df.groupby("cpus"):
        table.add_row(
            str(cpus),
            str(len(sub)),
            f"{sub['score_ms'].min():.2f}",
            f"{sub['score_ms'].median():.2f}",
            f"{sub['score_ms'].max():.2f}",
        )
    console.print(table)


def _parse_run_spec(spec: str) -> tuple[str, list[Path]]:
    """Parse one ``LABEL=PATH[,PATH...]`` argument into a (label, paths) tuple."""
    if "=" not in spec:
        raise typer.BadParameter(
            f"Run spec must look like LABEL=PATH[,PATH...] (got '{spec}')"
        )
    label, paths_str = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise typer.BadParameter(f"Empty label in run spec '{spec}'")
    paths = [Path(p.strip()) for p in paths_str.split(",") if p.strip()]
    if not paths:
        raise typer.BadParameter(f"No paths in run spec '{spec}'")
    return label, paths


@app.command()
def compare(
    runs: Annotated[
        list[str],
        typer.Argument(
            help=(
                "One or more 'LABEL=PATH[,PATH...]' specs. "
                "Examples: 'baseline=results/bench-cpu4.json' "
                "'tuned=runs/new/bench-cpu4.json,runs/new/bench-cpu2.json'"
            )
        ),
    ],
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("results/compare"),
    top_n: Annotated[int, typer.Option("--top", help="Number of regressions/improvements to list")] = 10,
) -> None:
    """Compare N labeled run sets (≥2), joined on the param combination.

    The first run is treated as the baseline; deltas are computed against it.
    Per-pair delta columns are added for every (baseline → other) pair.
    """
    if len(runs) < 2:
        raise typer.BadParameter("Need at least two runs to compare")

    parsed = [_parse_run_spec(spec) for spec in runs]
    labels = [label for label, _ in parsed]
    if len(set(labels)) != len(labels):
        raise typer.BadParameter(f"Run labels must be unique (got {labels})")
    baseline = labels[0]

    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "plots"
    plots_dir.mkdir(exist_ok=True)

    frames = [_load_all(paths).assign(run=label) for label, paths in parsed]
    long_df = pd.concat(frames, ignore_index=True)
    long_df = _length_sort(long_df)

    # Wide: one row per param combo, one column per run label.
    wide = long_df.pivot_table(
        index=JOIN_KEYS,
        columns="run",
        values="score_ms",
        aggfunc="mean",
    ).reset_index()
    missing = [lbl for lbl in labels if lbl not in wide.columns]
    if missing:
        raise typer.BadParameter(
            f"Could not join runs — labels with no matched rows: {missing}. "
            f"Labels seen in data: {[c for c in wide.columns if c not in JOIN_KEYS]}"
        )

    # Delta columns: every non-baseline vs baseline.
    for lbl in labels[1:]:
        wide[f"delta_pct[{lbl}_vs_{baseline}]"] = 100.0 * (wide[lbl] - wide[baseline]) / wide[baseline]

    # Sort by the first non-baseline delta column.
    sort_col = f"delta_pct[{labels[1]}_vs_{baseline}]"
    wide = wide.sort_values(sort_col)

    # CSV
    csv_path = out / "comparison.csv"
    wide.to_csv(csv_path, index=False)
    console.print(f"[green]✓[/green] wrote {csv_path}  ({len(wide)} joined rows, {len(labels)} runs)")

    # Markdown
    md = []
    md.append(f"# Comparison — {len(labels)} runs\n")
    md.append("Runs: " + ", ".join(f"`{lbl}`" for lbl in labels))
    md.append(f"\nBaseline: `{baseline}`. Joined on `{', '.join(JOIN_KEYS)}` — {len(wide)} matched configs.\n")
    md.append("Negative `delta_pct` means the run is **faster** than the baseline.\n")

    score_cols = labels
    delta_cols = [f"delta_pct[{lbl}_vs_{baseline}]" for lbl in labels[1:]]

    md.append(f"\n## Top {top_n} improvements vs `{baseline}` (by `{sort_col}`)\n")
    md.append(_markdown_table(wide.head(top_n), cols=[*JOIN_KEYS, *score_cols, *delta_cols]))

    md.append(f"\n\n## Top {top_n} regressions vs `{baseline}` (by `{sort_col}`)\n")
    md.append(_markdown_table(wide.tail(top_n).iloc[::-1], cols=[*JOIN_KEYS, *score_cols, *delta_cols]))

    # Per-profile mean delta across all non-baseline runs.
    md.append("\n\n## Per-profile mean delta vs baseline\n")
    per_profile = wide.groupby("profile")[delta_cols].mean().reset_index()
    md.append(_markdown_table(per_profile, cols=["profile", *delta_cols]))

    (out / "comparison.md").write_text("\n".join(md))
    console.print(f"[green]✓[/green] wrote {out / 'comparison.md'}")

    # Plot: overlay latency curves by run.
    _overlay_latency(long_df, label_col="run", out_path=plots_dir / "comparison_latency.png")
    console.print(f"[green]✓[/green] wrote {plots_dir / 'comparison_latency.png'}")

    # Console summary table.
    console.print()
    table = Table(title=f"compare — baseline: {baseline}")
    table.add_column("metric")
    for lbl in labels:
        table.add_column(lbl, justify="right")
    for lbl in labels[1:]:
        table.add_column(f"Δ% {lbl}", justify="right")
    for stat_name, fn in [
        ("min ms/op", "min"),
        ("median ms/op", "median"),
        ("mean ms/op", "mean"),
        ("max ms/op", "max"),
    ]:
        row = [stat_name]
        vals = {lbl: getattr(wide[lbl], fn)() for lbl in labels}
        for lbl in labels:
            row.append(f"{vals[lbl]:.2f}")
        base_val = vals[baseline]
        for lbl in labels[1:]:
            row.append(f"{100.0 * (vals[lbl] - base_val) / base_val:+.1f}%")
        table.add_row(*row)
    console.print(table)


if __name__ == "__main__":
    app()
