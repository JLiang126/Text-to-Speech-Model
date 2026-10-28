#!/usr/bin/env python
"""Compares systems scored by scripts/evaluate.py and makes the report table and figures.

Run from the repo root:

  python scripts/compare.py results/ground_truth results/pretrained results/expA results/expB \
      --pair expA expB --out results/comparison

Writes to --out:
  comparison_table.csv / .md   one row per system: mean [95% CI] for every metric
  paired_tests.csv             B - A on the same sentences (bootstrap CI + Wilcoxon p) for each --pair
  comparison.png               dot plot with 95% CIs, one panel per metric
  learning_curves.png          metric vs training step (only if an experiment was scored at >= 2 steps)

Without --pair, every pair of non-ground-truth systems is compared.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.evaluation.stats import bootstrap_ci, paired_comparison  # noqa: E402

PAIRED_METRICS = ("mcd_dtw", "wer", "cer", "secs", "secs_paired", "utmos", "dur_ratio")
# (key, panel title, scale for display, lower is better?)
PLOT_METRICS = [
    ("mcd_dtw", "MCD-DTW (dB)", 1.0, True),
    ("wer", "WER (%)", 100.0, True),
    ("secs", "Speaker similarity", 1.0, False),
    ("utmos", "UTMOS (predicted MOS)", 1.0, False),
    ("rtf", "Real-time factor", 1.0, True),
]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated, CVD-safe)
REFERENCE = "#8a8984"  # neutral ink for the ground-truth / pretrained reference systems


class System:
    def __init__(self, folder: Path):
        self.folder = folder
        self.summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        self.per_utt = pd.read_csv(folder / "per_utterance.csv")
        runs = folder / "rtf_runs.csv"
        self.rtf_runs = pd.read_csv(runs) if runs.is_file() else None
        self.name = str(self.summary["experiment"])
        self.step = self.summary.get("step")
        self.is_reference = bool(self.summary.get("ground_truth")) or self.name in ("ground_truth", "pretrained")
        self.label = self.name

    def value(self, key: str) -> Tuple[float, float, float]:
        """(point, lo, hi). WER/CER use corpus rates; RTF bootstraps the per-sentence medians."""
        nan = (float("nan"),) * 3
        if key in ("wer", "cer") and key in self.summary.get("corpus", {}):
            c = self.summary["corpus"][key]
            return c["value"], c["ci_low"], c["ci_high"]
        if key == "rtf":
            if self.rtf_runs is None:
                return nan
            per_text = self.rtf_runs.groupby("text_id")["rtf"].median().to_numpy()
            return bootstrap_ci(per_text)
        m = self.summary.get("metrics", {}).get(key)
        return (m["mean"], m["ci_low"], m["ci_high"]) if m else nan


def fmt(point: float, lo: float, hi: float, scale: float = 1.0, digits: int = 2) -> str:
    if np.isnan(point):
        return "-"
    return f"{point * scale:.{digits}f} [{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]"


def results_table(systems: List[System]) -> pd.DataFrame:
    rows = []
    for s in systems:
        lat = s.summary.get("latency") or {}
        device = s.summary.get("device", {})
        rows.append({
            "system": s.label, "step": s.step if s.step is not None else "",
            "MCD-DTW (dB)": fmt(*s.value("mcd_dtw")),
            "WER (%)": fmt(*s.value("wer"), scale=100, digits=1),
            "CER (%)": fmt(*s.value("cer"), scale=100, digits=1),
            "Speaker sim.": fmt(*s.value("secs"), digits=3),
            "UTMOS": fmt(*s.value("utmos")),
            "Duration ratio": fmt(*s.value("dur_ratio")),
            "RTF": fmt(*s.value("rtf"), digits=3),
            "Latency (s)": f"{lat['latency_mean_s']:.3f} ± {lat['latency_std_s']:.3f}" if lat else "-",
            "Timed on": (device.get("gpu") or device.get("cpu", "")) if lat else "-",
            "n": s.summary.get("n_utterances", ""),
        })
    return pd.DataFrame(rows)


def paired_tests(systems: List[System], pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    by_label = {s.label: s for s in systems}
    rows = []
    for a_name, b_name in pairs:
        if a_name not in by_label or b_name not in by_label:
            print(f"[warn] skipping pair {a_name} vs {b_name}: not among {sorted(by_label)}")
            continue
        merged = by_label[a_name].per_utt.merge(by_label[b_name].per_utt, on="utt_id", suffixes=("_a", "_b"))
        for metric in PAIRED_METRICS:
            if f"{metric}_a" in merged and f"{metric}_b" in merged:
                result = paired_comparison(merged[f"{metric}_a"], merged[f"{metric}_b"])
                if result["n"] > 0:
                    rows.append({"a": a_name, "b": b_name, "metric": metric, **result})
    return pd.DataFrame(rows)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_comparison(systems: List[System], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [m for m in PLOT_METRICS if any(not np.isnan(s.value(m[0])[0]) for s in systems)]
    if not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.9 * len(metrics), 0.5 * len(systems) + 1.6),
                             sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    y = np.arange(len(systems))[::-1]
    for ax, (key, title, scale, lower_better) in zip(axes, metrics):
        _style(ax)
        for yi, s in zip(y, systems):
            point, lo, hi = (v * scale for v in s.value(key))
            if np.isnan(point):
                continue
            color = REFERENCE if s.is_reference else SERIES[0]
            ax.plot([lo, hi], [yi, yi], color=color, linewidth=2, solid_capstyle="round")
            ax.plot(point, yi, "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_title(f"{title}\n{'lower' if lower_better else 'higher'} is better", fontsize=9.5,
                     color=INK, loc="left")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([s.label for s in systems], fontsize=9.5, color=INK)
    fig.suptitle("Objective metrics on the test split (dot = mean, line = 95% CI)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_learning_curves(systems: List[System], path: Path) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups: Dict[str, List[System]] = {}
    for s in systems:
        if not s.is_reference and s.step is not None:
            groups.setdefault(s.name, []).append(s)
    groups = {k: sorted(v, key=lambda s: s.step) for k, v in groups.items() if len({s.step for s in v}) >= 2}
    if not groups:
        return False
    references = [s for s in systems if s.is_reference]
    metrics = [m for m in PLOT_METRICS if m[0] != "rtf"
               and any(not np.isnan(s.value(m[0])[0]) for g in groups.values() for s in g)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.2), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (key, title, scale, lower_better) in zip(axes, metrics):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for color, (name, runs) in zip(SERIES, groups.items()):
            steps = [s.step for s in runs]
            vals = np.array([s.value(key) for s in runs]) * scale
            ax.fill_between(steps, vals[:, 1], vals[:, 2], color=color, alpha=0.15, linewidth=0)
            ax.plot(steps, vals[:, 0], color=color, linewidth=2, marker="o", markersize=5, label=name)
            ax.annotate(name, (steps[-1], vals[-1, 0]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8.5, color=INK_2, va="center")
        for ref, style in zip(references, ("--", ":")):
            point = ref.value(key)[0] * scale
            if not np.isnan(point):
                ax.axhline(point, color=REFERENCE, linewidth=1.2, linestyle=style, label=ref.label)
        ax.set_title(f"{title} ({'lower' if lower_better else 'higher'} is better)", fontsize=9.5,
                     color=INK, loc="left")
        ax.set_xlabel("training step", fontsize=9, color=INK_2)
        ax.margins(x=0.15)  # room for the direct label at the end of each line
    axes[0].legend(fontsize=8, frameon=False, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return True


def to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *body]) + "\n"


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+", help="results/<exp> folders written by evaluate.py")
    p.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"), help="compare B against A (repeatable)")
    p.add_argument("--out", default="results/comparison")
    args = p.parse_args(argv)

    systems = [System(Path(r)) for r in args.results]
    names = [s.name for s in systems]
    for s in systems:  # same experiment scored at several steps -> label with the step
        if names.count(s.name) > 1:
            s.label = f"{s.name}@{s.step}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    table = results_table(systems)
    table.to_csv(out / "comparison_table.csv", index=False)
    (out / "comparison_table.md").write_text(to_markdown(table), encoding="utf-8")
    print(to_markdown(table))

    candidates = [s.label for s in systems if not s.summary.get("ground_truth")]
    pairs = [tuple(pair) for pair in args.pair] if args.pair else list(itertools.combinations(candidates, 2))
    tests = paired_tests(systems, pairs)
    if not tests.empty:
        tests.to_csv(out / "paired_tests.csv", index=False)
        print(tests[["a", "b", "metric", "n", "mean_diff", "ci_low", "ci_high", "wilcoxon_p"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    latest = {}
    for s in systems:  # dot plot: one point per system, latest step when several were scored
        if s.name not in latest or (s.step or 0) >= (latest[s.name].step or 0):
            latest[s.name] = s
    plot_comparison(list(latest.values()), out / "comparison.png")
    if plot_learning_curves(systems, out / "learning_curves.png"):
        print(f"[info] wrote {out / 'learning_curves.png'}")
    print(f"[info] wrote {out}/comparison_table.csv, comparison_table.md, paired_tests.csv, comparison.png")


if __name__ == "__main__":
    main()
