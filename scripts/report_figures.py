#!/usr/bin/env python
"""The two summary figures of the report, drawn from saved results (no checkpoints needed).

  python scripts/report_figures.py

writes to results/experiments_v2/report/figures/:
  before_after.png  every round-2 run on four test metrics: its starting point (the pretrained model
                    with the run's speaker index) and where fine-tuning took it
  losses.png        training and validation losses by epoch for the speaker-embedding runs, from the
                    training logs; validation starts at epoch 0 (the pretrained model) once
                    scripts/pretrained_validation_losses.py has written its CSV

The run IDs (B, W, E, ...) are the ones the report's tables use.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

# (report ID, run name, label); the order is the figure's row order
RUNS = [
    ("B", "best_speaker_embedding", "speaker 7 · full"),
    ("W", "worst_speaker_embedding", "speaker 24 · full"),
    ("B10", "best_speaker_embedding_10xLR", "speaker 7 · full, 10× emb. LR"),
    ("W10", "worst_speaker_embedding_10xLR", "speaker 24 · full, 10× emb. LR"),
    ("F", "freeze_text_encoder", "speaker 7 · text encoder frozen"),
    ("F10", "freeze_text_encoder_10xLR", "speaker 7 · encoder frozen, 10× LR"),
    ("O", "original_vits_config", "speaker 7 · layer drop 0.1"),
    ("E", "unfreeze_speech_embeddings_only", "speaker 7 · embedding only"),
    ("EW", "unfreeze_speech_embeddings_only_WSE", "speaker 24 · embedding only"),
    ("E10", "unfreeze_speech_embeddings_only_10xLR", "speaker 7 · emb. only, 10× LR"),
    ("EW10", "unfreeze_speech_embeddings_only_WSE_10xLR", "speaker 24 · emb. only, 10× LR"),
]
FULL_RUNS = 7  # the first seven train the whole model, the last four only the speaker embedding
LOSS_RUNS = ("B", "W", "E", "EW", "E10", "EW10")

BEFORE, AFTER = "#86b6ef", "#1c5cab"  # one hue, two shades: survives greyscale printing
CONNECT, REF, GRID, INK, INK2 = "#c3c2b7", "#52514e", "#e1e0d9", "#0b0b0b", "#52514e"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # validated categorical order
MARKERS = ["o", "o", "s", "s", "^", "^"]  # full, embedding only, embedding only at 10x


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": CONNECT,
                         "xtick.color": INK2, "ytick.color": INK})
    return plt


def _metric(results: Path, system: str, key: str):
    d = json.loads((results / system / "summary.json").read_text(encoding="utf-8"))["metrics"][key]
    return d["mean"], d["ci_low"], d["ci_high"]


def before_after(results: Path, experiments: pd.DataFrame, out: Path) -> Path:
    """Dumbbell rows, one per run, one panel per metric: pretrained start (light) to fine-tuned (dark)."""
    plt = _style()
    from matplotlib.lines import Line2D

    panels = [("secs", "Speaker similarity to LJ\n(higher = more like LJ)", True),
              ("utmos", "UTMOS (predicted, 1–5)\n(higher = more natural)", True),
              ("f0_hz", "Median pitch (Hz)\n(LJ recordings: 207 Hz)", True),
              ("mcd_dtw", "MCD (dB)\n(lower = closer to the recording)", False)]
    rows = [r for r in RUNS if (results / r[1] / "summary.json").is_file()]
    ypos = [i + (0.7 if i >= FULL_RUNS else 0) for i in range(len(rows))]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.05, 3.5), sharey=True, gridspec_kw={"wspace": 0.16})
    for ax, (key, title, has_reference) in zip(axes, panels):
        if has_reference:
            ax.axvline(_metric(results, "ground_truth", key)[0], color=REF, lw=1, ls=(0, (3, 2)), zorder=1)
        for y, (_, run, _) in zip(ypos, rows):
            start = _metric(results, f"pretrained_spk{int(experiments.loc[run, 'speaker_id'])}", key)[0]
            end, low, high = _metric(results, run, key)
            ax.plot([start, end], [y, y], color=CONNECT, lw=1.5, zorder=2, solid_capstyle="round")
            ax.plot([low, high], [y, y], color=AFTER, lw=2.2, alpha=0.45, zorder=3, solid_capstyle="round")
            ax.plot(start, y, "o", ms=5, color=BEFORE, mec="white", mew=0.8, zorder=4)
            ax.plot(end, y, "o", ms=5, color=AFTER, mec="white", mew=0.8, zorder=5)
        ax.set_title(title, fontsize=7.5, loc="left", color=INK, pad=6)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=2, labelsize=7)
        ax.margins(x=0.08)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([f"{rid}   {label}" for rid, _, label in rows])
    axes[0].set_ylim(max(ypos) + 0.6, -0.6)
    handles = [Line2D([], [], marker="o", ls="", ms=5, color=BEFORE, mec="white", label="before fine-tuning (pretrained voice)"),
               Line2D([], [], marker="o", ls="-", lw=2.2, ms=5, color=AFTER, mec="white", label="after fine-tuning (bar = 95% CI)"),
               Line2D([], [], color=REF, lw=1, ls=(0, (3, 2)), label="real LJ recordings")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.6, -0.01))
    fig.subplots_adjust(left=0.27, right=0.985, top=0.86, bottom=0.14)
    return _save(fig, out)


def _epoch0(path: Optional[Path]) -> Dict[int, pd.Series]:
    """Pretrained-model validation losses by speaker ID, if scripts/pretrained_validation_losses.py ran."""
    if path is None or not path.is_file():
        return {}
    table = pd.read_csv(path)
    return {int(row["speaker_id"]): row for _, row in table.iterrows()}


def losses(runs_dir: Path, experiments: pd.DataFrame, out: Path, epoch0_csv: Optional[Path] = None) -> Path:
    """One row of panels (mel, KL, duration): validation solid, training dashed, one colour per run."""
    plt = _style()
    from matplotlib.lines import Line2D
    from scripts.evaluation.training_logs import find_run_folder, load_history

    start = _epoch0(epoch0_csv)
    panels = [("mel", "Mel L1 loss", False), ("kl", "KL loss (log scale)", True), ("duration", "Duration loss", False)]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.05, 2.6), gridspec_kw={"wspace": 0.25})
    labels = {rid: (run, label) for rid, run, label in RUNS}
    handles = []
    for i, rid in enumerate(LOSS_RUNS):
        run, label = labels[rid]
        folder = find_run_folder(run, None, runs_dir)
        if folder is None:
            continue
        train, val = load_history(folder)
        speaker = int(experiments.loc[run, "speaker_id"])
        if speaker in start:
            val = pd.concat([start[speaker][["epoch", *(p[0] for p in panels)]].to_frame().T, val], ignore_index=True)
        colour, marker = SERIES[i], MARKERS[i]
        for ax, (key, _, _) in zip(axes, panels):
            ax.plot(val["epoch"], val[key], color=colour, lw=1.6, marker=marker, ms=3.2, mec="white", mew=0.5)
            if train is not None and key in train:
                ax.plot(train["epoch"], train[key], color=colour, lw=1, ls=(0, (3, 2)), alpha=0.75)
        handles.append(Line2D([], [], color=colour, marker=marker, ms=4, mec="white", lw=1.6, label=f"{rid}  {label}"))
    for ax, (key, title, log) in zip(axes, panels):
        if log:
            from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, StrMethodFormatter
            ax.set_yscale("log")
            low, high = ax.get_ylim()
            ax.yaxis.set_major_locator(FixedLocator([t for t in (1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100)
                                                     if low <= t <= high]))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, fontsize=7.5, loc="left", color=INK, pad=5)
        ax.set_xlabel("epoch" + (" (0 = pretrained model)" if start else ""), fontsize=7, color=INK2)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=2, labelsize=7)
    handles += [Line2D([], [], color=INK2, lw=1.6, label="validation"),
                Line2D([], [], color=INK2, lw=1, ls=(0, (3, 2)), label="training")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.31)
    return _save(fig, out)


def _save(fig, out: Path) -> Path:
    import matplotlib.pyplot as plt
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


def main(argv: Optional[List[str]] = None) -> Dict[str, Path]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default="results/experiments_v2")
    p.add_argument("--runs_dir", default="outputs_v2", help="the runs' training logs")
    p.add_argument("--experiments", default="experiments/experiments_v2.csv")
    p.add_argument("--epoch0", default="outputs_v2/pretrained_validation_losses.csv",
                   help="pretrained-model validation losses (scripts/pretrained_validation_losses.py)")
    p.add_argument("--out_dir", default="results/experiments_v2/report/figures")
    args = p.parse_args(argv)
    experiments = pd.read_csv(args.experiments).set_index("experiment")
    out = Path(args.out_dir)
    paths = {"before_after": before_after(Path(args.results_dir), experiments, out / "before_after.png"),
             "losses": losses(Path(args.runs_dir), experiments, out / "losses.png", Path(args.epoch0))}
    for path in paths.values():
        print(f"Wrote {path}")
    return paths


if __name__ == "__main__":
    main()
