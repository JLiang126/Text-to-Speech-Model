#!/usr/bin/env python
"""Compares systems scored by scripts/evaluate.py and makes the report's table and figures.

Run from the repo root:

  python scripts/compare.py results/ground_truth results/pretrained results/expA results/expB \
      --pair expA expB --out results/comparison

Writes to --out:
  comparison_table.csv / .md   one row per system: mean [95% CI] for every metric
  paired_tests.csv             B - A on the same sentences, per metric: mean difference with a
                               95% CI, Wilcoxon p (raw and Holm-adjusted across metrics),
                               Cohen's d_z, and a plain-language verdict
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

from scripts.evaluation.stats import bootstrap_ratio_ci, holm_adjust, paired_comparison  # noqa: E402

# metric -> True if lower is better. dur_dev = |duration ratio - 1| (speaking-rate error).
PAIRED_METRICS = {"mcd_dtw": True, "wer": True, "cer": True, "secs": False, "secs_paired": False,
                  "utmos": False, "dur_dev": True}
# (key, panel title, display scale, lower is better)
QUALITY_PANELS = [
    ("mcd_dtw", "MCD-DTW (dB)", 1.0, True),
    ("wer", "WER (%)", 100.0, True),
    ("secs", "Speaker similarity", 1.0, False),
    ("utmos", "UTMOS (predicted MOS)", 1.0, False),
]
DEVICE_NAMES = {"cuda": "GPU", "cpu": "CPU", "mps": "Apple GPU"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated, colour-blind safe)
REFERENCE = "#8a8984"  # neutral ink for the ground-truth / pretrained reference systems
NAN3 = (float("nan"),) * 3


class System:
    """Everything evaluate.py wrote for one system (any file may be missing)."""

    def __init__(self, folder: Path):
        self.folder = folder
        summary_path = folder / "summary.json"
        self.summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        per_utt = folder / "per_utterance.csv"
        self.per_utt = pd.read_csv(per_utt) if per_utt.is_file() else pd.DataFrame({"utt_id": []})
        if "dur_ratio" in self.per_utt.columns:
            self.per_utt["dur_dev"] = (self.per_utt["dur_ratio"] - 1.0).abs()
        self.latency: Dict[str, dict] = {}
        self.rtf_runs: Dict[str, pd.DataFrame] = {}
        for path in sorted(folder.glob("latency_*.json")):
            tag = path.stem.split("_", 1)[1]
            self.latency[tag] = json.loads(path.read_text(encoding="utf-8"))
            runs = folder / f"rtf_runs_{tag}.csv"
            if runs.is_file():
                self.rtf_runs[tag] = pd.read_csv(runs)
        first_latency = next(iter(self.latency.values()), {})
        self.name = str(self.summary.get("experiment") or first_latency.get("experiment") or folder.name)
        self.step = self.summary.get("step", first_latency.get("step"))
        self.is_reference = bool(self.summary.get("ground_truth")) or self.name in ("ground_truth", "pretrained")
        self.label = self.name

    def value(self, key: str) -> Tuple[float, float, float]:
        """(point, lo, hi). WER/CER use corpus rates; rtf_<device> bootstraps per-sentence medians."""
        if key in ("wer", "cer") and key in self.summary.get("corpus", {}):
            c = self.summary["corpus"][key]
            return c["value"], c["ci_low"], c["ci_high"]
        if key.startswith("rtf_"):
            # Aggregate RTF = total synthesis time / total audio time (robust to very short sentences),
            # with a bootstrap CI over the timing sentences.
            runs = self.rtf_runs.get(key[len("rtf_"):])
            if runs is None or runs.empty:
                return NAN3
            per_text = runs.groupby("text_id")[["latency_s", "audio_s"]].mean()
            return bootstrap_ratio_ci(per_text["latency_s"].to_numpy(), per_text["audio_s"].to_numpy())
        m = self.summary.get("metrics", {}).get(key)
        return (m["mean"], m["ci_low"], m["ci_high"]) if m else NAN3


def fmt(point: float, lo: float, hi: float, scale: float = 1.0, digits: int = 2) -> str:
    if np.isnan(point):
        return "-"
    return f"{point * scale:.{digits}f} [{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]"


def device_tags(systems: List[System]) -> List[str]:
    tags = {tag for s in systems for tag in s.latency}
    return [t for t in ("cuda", "mps", "cpu") if t in tags] + sorted(tags - {"cuda", "mps", "cpu"})


def results_table(systems: List[System]) -> pd.DataFrame:
    tags = device_tags(systems)
    rows = []
    for s in systems:
        row = {
            "system": s.label, "step": s.step if s.step is not None else "",
            "MCD-DTW (dB)": fmt(*s.value("mcd_dtw")),
            "WER (%)": fmt(*s.value("wer"), scale=100, digits=1),
            "CER (%)": fmt(*s.value("cer"), scale=100, digits=1),
            "Speaker sim.": fmt(*s.value("secs"), digits=3),
            "UTMOS": fmt(*s.value("utmos")),
            "Duration ratio": fmt(*s.value("dur_ratio")),
        }
        for tag in tags:
            name = DEVICE_NAMES.get(tag, tag)
            lat = s.latency.get(tag)
            row[f"RTF {name}"] = fmt(*s.value(f"rtf_{tag}"), digits=3)
            row[f"Latency {name} (s)"] = (f"{lat['latency_mean_s']:.3f} ± {lat['latency_std_s']:.3f}"
                                          if lat else "-")
        row["n sentences"] = s.summary.get("n_utterances", "")
        rows.append(row)
    return pd.DataFrame(rows)


def timing_hardware(systems: List[System]) -> Dict[str, str]:
    """Which hardware each timing column came from (should be one machine per column)."""
    hardware: Dict[str, set] = {}
    for s in systems:
        for tag, lat in s.latency.items():
            dev = lat.get("device", {})
            hardware.setdefault(tag, set()).add(dev.get("gpu") or dev.get("cpu", "unknown"))
    return {tag: ", ".join(sorted(names)) for tag, names in hardware.items()}


def verdict(row: dict, lower_is_better: bool, alpha: float = 0.05) -> str:
    if not (row["ci_low"] > 0 or row["ci_high"] < 0) or not row["wilcoxon_p_holm"] < alpha:
        return "no clear difference"
    b_lower = row["mean_diff"] < 0
    return f"{row['b']} better" if b_lower == lower_is_better else f"{row['a']} better"


def paired_tests(systems: List[System], pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    by_label = {s.label: s for s in systems}
    rows = []
    for a_name, b_name in pairs:
        if a_name not in by_label or b_name not in by_label:
            print(f"[warn] skipping pair {a_name} vs {b_name}: not among {sorted(by_label)}")
            continue
        merged = by_label[a_name].per_utt.merge(by_label[b_name].per_utt, on="utt_id", suffixes=("_a", "_b"))
        pair_rows = []
        for metric, lower_is_better in PAIRED_METRICS.items():
            if f"{metric}_a" in merged and f"{metric}_b" in merged:
                result = paired_comparison(merged[f"{metric}_a"], merged[f"{metric}_b"])
                if result["n"] > 0:
                    pair_rows.append({"a": a_name, "b": b_name, "metric": metric,
                                      "better_if": "lower" if lower_is_better else "higher", **result})
        for row, p_holm in zip(pair_rows, holm_adjust([r["wilcoxon_p"] for r in pair_rows])):
            row["wilcoxon_p_holm"] = p_holm
            row["verdict"] = verdict(row, PAIRED_METRICS[row["metric"]])
        rows.extend(pair_rows)
    return pd.DataFrame(rows)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _panels(systems: List[System]):
    panels = list(QUALITY_PANELS)
    for tag in device_tags(systems):
        panels.append((f"rtf_{tag}", f"Real-time factor ({DEVICE_NAMES.get(tag, tag)})", 1.0, True))
    return [p for p in panels if any(not np.isnan(s.value(p[0])[0]) for s in systems)]


def plot_comparison(systems: List[System], path: Path) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = _panels(systems)
    if not panels:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(max(2.9 * len(panels), 6.5), 0.5 * len(systems) + 1.9),
                             sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    y = np.arange(len(systems))[::-1]
    for ax, (key, title, scale, lower_better) in zip(axes, panels):
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
    fig.suptitle("Objective metrics on the test split\ndot = mean, line = 95% CI, grey = reference systems",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return True


def plot_learning_curves(systems: List[System], path: Path) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups: Dict[str, List[System]] = {}
    for s in systems:
        if not s.is_reference and s.step is not None and s.summary:
            groups.setdefault(s.name, []).append(s)
    groups = {k: sorted(v, key=lambda s: s.step) for k, v in groups.items() if len({s.step for s in v}) >= 2}
    if not groups:
        return False
    references = [s for s in systems if s.is_reference]
    panels = [p for p in QUALITY_PANELS
              if any(not np.isnan(s.value(p[0])[0]) for g in groups.values() for s in g)]
    if not panels:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.2), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (key, title, scale, lower_better) in zip(axes, panels):
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


def main(argv: Optional[List[str]] = None) -> Dict[str, object]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+", help="result folders written by evaluate.py")
    p.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"), help="compare B against A (repeatable)")
    p.add_argument("--out", default="results/comparison")
    args = p.parse_args(argv)

    systems = [System(Path(r)) for r in args.results]
    names = [s.name for s in systems]
    for s in systems:  # the same experiment scored at several steps -> label with the step
        if names.count(s.name) > 1:
            s.label = f"{s.name}@{s.step}"
    labels = [s.label for s in systems]
    for s in systems:  # still ambiguous (e.g. the same step evaluated twice) -> add the folder name
        if labels.count(s.label) > 1:
            s.label = f"{s.label} ({s.folder.name})"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    table = results_table(systems)
    table.to_csv(out / "comparison_table.csv", index=False)
    hardware = timing_hardware(systems)
    notes = "".join(f"\nTiming ({DEVICE_NAMES.get(t, t)}): {h}" for t, h in hardware.items())
    (out / "comparison_table.md").write_text(to_markdown(table) + notes + "\n", encoding="utf-8")
    print(to_markdown(table) + notes)
    for tag, names_used in hardware.items():
        if "," in names_used:
            print(f"[warn] {DEVICE_NAMES.get(tag, tag)} timings come from different hardware ({names_used}); "
                  f"re-time on one machine before comparing RTF.")

    candidates = [s.label for s in systems if not s.summary.get("ground_truth") and s.summary]
    pairs = [tuple(pair) for pair in args.pair] if args.pair else list(itertools.combinations(candidates, 2))
    tests = paired_tests(systems, pairs)
    if not tests.empty:
        tests.to_csv(out / "paired_tests.csv", index=False)
        print(tests[["a", "b", "metric", "n", "mean_diff", "ci_low", "ci_high", "wilcoxon_p_holm",
                     "cohens_dz", "verdict"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    latest: Dict[str, System] = {}
    for s in systems:  # dot plot: one point per system, the latest step when several were scored
        if s.name not in latest or (s.step or 0) >= (latest[s.name].step or 0):
            latest[s.name] = s
    wrote_plot = plot_comparison(list(latest.values()), out / "comparison.png")
    wrote_curves = plot_learning_curves(systems, out / "learning_curves.png")
    written = ["comparison_table.csv", "comparison_table.md"] + (["paired_tests.csv"] if not tests.empty else []) \
        + (["comparison.png"] if wrote_plot else []) + (["learning_curves.png"] if wrote_curves else [])
    print(f"[info] wrote to {out}: {', '.join(written)}")
    return {"table": table, "tests": tests}


if __name__ == "__main__":
    main()
