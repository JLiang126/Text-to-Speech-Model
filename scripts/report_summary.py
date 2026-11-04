#!/usr/bin/env python
"""The summary tables and figures of notebook section 1.4 and of the report, from saved results.

  python scripts/report_summary.py   # writes the two figures to results/experiments_v2/report/figures/

  results_table        every round-2 system: metric means and how far its speaker vector moved
  ci_note              the widest 95% confidence interval of each metric, in one line
  differences_table    which changes made a clear difference (paired tests on the same sentences)
  learning_rate_table  the four learning-rate runs of round 1, against its baseline
  before_after         figure: every round-2 run before and after fine-tuning
  losses               figure: training and validation losses by epoch, from epoch 0 (the pretrained model)

No checkpoints are needed. The run IDs (B, W, E, ...) are the ones the report uses.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

# (ID, run, what it changes, short label); full fine-tuning first, then the embedding-only runs
FULL = [
    ("B", "best_speaker_embedding", "Base case: speaker 7, full fine-tuning", "speaker 7 (base case)"),
    ("W", "worst_speaker_embedding", "B, starting from speaker 24", "speaker 24"),
    ("B10", "best_speaker_embedding_10xLR", "B, 10× embedding learning rate", "speaker 7, 10× embedding LR"),
    ("W10", "worst_speaker_embedding_10xLR", "W, 10× embedding learning rate", "speaker 24, 10× embedding LR"),
    ("F", "freeze_text_encoder", "B, text encoder frozen", "speaker 7, text encoder frozen"),
    ("F10", "freeze_text_encoder_10xLR", "F, 10× embedding learning rate", "speaker 7, frozen encoder, 10× LR"),
    ("O", "original_vits_config", "B, text-encoder layer drop 0.1 (original VITS)", "speaker 7, layer drop 0.1"),
]
EMBEDDING_ONLY = [
    ("E", "unfreeze_speech_embeddings_only", "B, only the speaker embedding trains", "speaker 7"),
    ("EW", "unfreeze_speech_embeddings_only_WSE", "E, starting from speaker 24", "speaker 24"),
    ("E10", "unfreeze_speech_embeddings_only_10xLR", "E, 10× embedding learning rate", "speaker 7, 10× LR"),
    ("EW10", "unfreeze_speech_embeddings_only_WSE_10xLR", "EW, 10× embedding learning rate", "speaker 24, 10× LR"),
]
RUNS = FULL + EMBEDDING_ONLY
RUN_NAME = {rid: run for rid, run, _, _ in RUNS}
SPEAKER = {"B": 7, "W": 24, "B10": 7, "W10": 24, "F": 7, "F10": 7, "O": 7, "E": 7, "EW": 24, "E10": 7, "EW10": 24}
REFERENCES = [("", "ground_truth", "Real LJ recordings"), ("", "pretrained_spk7", "Pretrained, speaker 7 (before fine-tuning)"),
              ("", "pretrained_spk24", "Pretrained, speaker 24 (before fine-tuning)")]
# (A, B, what B changes): each run against the run it changes one setting of
COMPARISONS = [("B", "W", "starts from speaker 24 instead of 7"), ("B", "E", "trains only the speaker embedding"),
               ("W", "EW", "trains only the speaker embedding"), ("B", "B10", "10× embedding learning rate"),
               ("W", "W10", "10× embedding learning rate"), ("E", "E10", "10× embedding learning rate"),
               ("EW", "EW10", "10× embedding learning rate"), ("B", "F", "freezes the text encoder"),
               ("F", "F10", "10× embedding learning rate"), ("B", "O", "text-encoder layer drop 0.1")]
SECTIONS = ("speaker_embedding", "embedding_learning_rate", "ruling_out")
LEARNING_RATE_RUNS = [("1e-5, linear decay", "halve_learning_rate"), ("2e-5, linear decay", "no_droplayer"),
                      ("4e-5, linear decay (used in round 2)", "double_learninge_rate"), ("2e-5, constant", "no_scheduler")]

BEFORE, AFTER = "#86b6ef", "#1c5cab"  # one hue, two shades: survives greyscale printing
REF, GRID, PANEL, INK, INK2 = "#52514e", "#e1e0d9", "#f6f6f3", "#0b0b0b", "#52514e"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # validated categorical order
MARKERS = ["o", "o", "s", "s", "^", "^"]  # full, embedding only, embedding only at 10x
LOSS_RUNS = ("B", "W", "E", "EW", "E10", "EW10")
GOOD, BAD = "#dff3e6", "#fbe1e1"


# ------------------------------------------------------------------ tables
def _summary(results: Path, system: str) -> dict:
    return json.loads((results / system / "summary.json").read_text(encoding="utf-8"))


def results_frame(results: Path) -> pd.DataFrame:
    """Means over the test sentences (WER is the corpus rate) and the speaker-vector change of each run."""
    checks = pd.read_csv(results / "report" / "checks.csv").set_index("experiment")
    rows = []
    for rid, run, what in [(r, n, w) for r, n, w in REFERENCES] + [(r, n, w) for r, n, w, _ in RUNS]:
        s = _summary(results, run)
        m = s["metrics"]
        moved = 100 * checks.loc[run, "speaker_rel_l2"] if run in checks.index else float("nan")
        rows.append({"ID": rid, "Run": what, "MCD (dB) ↓": m["mcd_dtw"]["mean"] if "mcd_dtw" in m else float("nan"),
                     "WER (%) ↓": 100 * s["corpus"]["wer"]["value"], "Speaker sim. ↑": m["secs"]["mean"],
                     "UTMOS ↑": m["utmos"]["mean"], "Pitch (Hz)": m["f0_hz"]["mean"],
                     "Duration ratio": m["dur_ratio"]["mean"] if "dur_ratio" in m else float("nan"),
                     "Vector moved (%)": moved})
    return pd.DataFrame(rows)


TEXT_COLUMNS = ("ID", "Run", "Change", "What B changes", "Schedule")


def _styled(frame: pd.DataFrame, formats: Dict[str, str]):
    """Index hidden, text columns (and their headers) left-aligned, numbers right-aligned."""
    text = [c for c in frame.columns if c in TEXT_COLUMNS]
    headers = [{"selector": f"th.col_heading.col{frame.columns.get_loc(c)}", "props": "text-align: left"} for c in text]
    return (frame.style.hide(axis="index").format(formats, na_rep="–")
            .set_properties(subset=text, **{"text-align": "left"}).set_table_styles(headers))


def _signed(fmt: str, value: float) -> str:
    """The value with its sign, but no minus on a value that rounds to zero ("-0.00" becomes "+0.00")."""
    out = fmt.format(value)
    return "+" + out[1:] if out.startswith("-") and not out.strip("-0.") else out


def results_table(results: Path):
    """Table 1 of the notebook: every round-2 system, with a rule above each group of runs."""
    frame = results_frame(results)
    starts = {len(REFERENCES), len(REFERENCES) + len(FULL)}
    rule = "border-top: 1.5px solid #b8b7b0"
    return (_styled(frame, {"MCD (dB) ↓": "{:.2f}", "WER (%) ↓": "{:.1f}", "Speaker sim. ↑": "{:.3f}",
                            "UTMOS ↑": "{:.2f}", "Pitch (Hz)": "{:.0f}", "Duration ratio": "{:.2f}",
                            "Vector moved (%)": "{:.1f}"})
            .apply(lambda row: [rule if row.name in starts else ""] * len(row), axis=1))


CI_METRICS = [("mcd_dtw", "MCD", 1, "±{:.2f} dB"), ("wer", "WER", 100, "±{:.1f} points"),
              ("secs", "speaker similarity", 1, "±{:.3f}"), ("utmos", "UTMOS", 1, "±{:.2f}"),
              ("f0_hz", "pitch", 1, "±{:.0f} Hz"), ("dur_ratio", "duration ratio", 1, "±{:.2f}")]


def ci_note(results: Path) -> str:
    """The widest 95% bootstrap CI (half-width) of each Table 1 metric over the round-2 runs."""
    widest = {}
    for _, run, _, _ in RUNS:
        s = _summary(results, run)
        for key, label, scale, _ in CI_METRICS:
            ci = s["corpus"]["wer"] if key == "wer" else s["metrics"][key]
            widest[label] = max(widest.get(label, 0.0), scale * (ci["ci_high"] - ci["ci_low"]) / 2)
    return ("95% confidence intervals (bootstrap over the 150 sentences), widest over the 11 runs: "
            + ", ".join(f"{label} {fmt.format(widest[label])}" for _, label, _, fmt in CI_METRICS) + ".")


DIFF_METRICS = [("mcd_dtw", "MCD (dB)", 1, "{:+.2f}"), ("wer", "WER (points)", 100, "{:+.1f}"),
                ("secs", "Speaker sim.", 1, "{:+.3f}"), ("utmos", "UTMOS", 1, "{:+.2f}"),
                ("f0_dev", "Pitch error (Hz)", 1, "{:+.1f}"), ("dur_dev", "Duration error", 1, "{:+.3f}")]


def _paired(results: Path, sections) -> pd.DataFrame:
    return pd.concat([pd.read_csv(results / "report" / s / "paired_tests.csv") for s in sections], ignore_index=True)


def _verdict(tests: pd.DataFrame, a: str, b: str, metric: str) -> int:
    """+1 = b clearly better than a, -1 = clearly worse, 0 = no clear difference (95% CI and Holm-corrected Wilcoxon)."""
    row = tests[(tests["a"] == a) & (tests["b"] == b) & (tests["metric"] == metric)].iloc[0]
    return 1 if row["verdict"] == f"{b} better" else (-1 if row["verdict"] == f"{a} better" else 0)


MARK = {1: " ✓", -1: " ✗", 0: ""}


def differences_frame(results: Path):
    """(values, verdicts): B minus A for each comparison; verdict +1 = clearly better, -1 = clearly worse, 0 = no clear difference."""
    tests = _paired(results, SECTIONS)
    values, verdicts = [], []
    for a, b, what in COMPARISONS:
        ra, rb = RUN_NAME[a], RUN_NAME[b]
        v = {"Change": f"{a} → {b}", "What B changes": what}
        d = {"Change": 0, "What B changes": 0}
        for key, label, scale, fmt in DIFF_METRICS:
            row = tests[(tests["a"] == ra) & (tests["b"] == rb) & (tests["metric"] == key)].iloc[0]
            d[label] = _verdict(tests, ra, rb, key)
            v[label] = _signed(fmt, scale * row["mean_diff"]) + MARK[d[label]]
        values.append(v)
        verdicts.append(d)
    return pd.DataFrame(values), pd.DataFrame(verdicts)


def _shaded(values: pd.DataFrame, verdicts: pd.DataFrame, formats: Dict[str, str]):
    colours = verdicts.apply(lambda col: col.map({1: f"background-color: {GOOD}", -1: f"background-color: {BAD}"}).fillna(""))
    return _styled(values, formats).apply(lambda _: colours, axis=None)


def differences_table(results: Path):
    """Table 2 of the notebook: shaded = clear difference (95% CI excludes 0 and Holm p < 0.05)."""
    return _shaded(*differences_frame(results), {})


LR_METRICS = [("mcd_dtw", "MCD (dB) ↓", "{:.2f}"), ("wer", "WER (%) ↓", "{:.1f}"),
              ("secs", "Speaker sim. ↑", "{:.3f}"), ("utmos", "UTMOS ↑", "{:.2f}")]


def learning_rate_table(results_round1: Path, runs_dir: Path = Path("outputs")):
    """Round 1: the four learning-rate runs (90/5/5 split, speaker 0, 15 epochs), each against the 2e-5 baseline."""
    from scripts.evaluation.training_logs import find_run_folder, load_history

    tests = _paired(results_round1, ["learning_rate"])
    baseline = "no_droplayer"
    values, verdicts = [], []
    for label, run in LEARNING_RATE_RUNS:
        s = _summary(results_round1, run)
        v = {"Schedule": label + (" (baseline)" if run == baseline else "")}
        d = {"Schedule": 0}
        for key, column, fmt in LR_METRICS:
            mean = 100 * s["corpus"]["wer"]["value"] if key == "wer" else s["metrics"][key]["mean"]
            d[column] = 0 if run == baseline else _verdict(tests, baseline, run, key)
            v[column] = fmt.format(mean) + MARK[d[column]]
        folder = find_run_folder(run, None, runs_dir)
        v["Val. mel loss ↓"] = f"{load_history(folder)[1]['mel'].iloc[-1]:.3f}" if folder else "–"
        d["Val. mel loss ↓"] = 0
        values.append(v)
        verdicts.append(d)
    return _shaded(pd.DataFrame(values), pd.DataFrame(verdicts), {})


# ------------------------------------------------------------------ figures
STYLE = {"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": GRID,
         "xtick.color": INK2, "ytick.color": INK, "axes.labelcolor": INK2}


def _styled_figure(draw):
    """Draws inside a local matplotlib style, on a Figure that is not pyplot's: a notebook's plots stay as they are."""
    @functools.wraps(draw)
    def wrapper(*args, **kwargs):
        import matplotlib
        with matplotlib.rc_context(STYLE):
            return draw(*args, **kwargs)
    return wrapper


def _panel(ax, title: str, subtitle: str, xlabel: str):
    ax.set_facecolor(PANEL)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="white", lw=1.2)
    ax.set_axisbelow(True)
    ax.set_title(f"{title}\n", fontsize=9, loc="left", color=INK, fontweight="bold", pad=2)
    ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=7.5, color=INK2, va="bottom")
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, labelsize=7.5)


@_styled_figure
def before_after(results: Path, out: Path) -> Path:
    """Arrows from each run's starting point (the pretrained model with its speaker) to where fine-tuning took it."""
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    panels = [("secs", "Speaker similarity to LJ", "higher = more like LJ", "cosine similarity", (-0.25, 1.0)),
              ("utmos", "Naturalness (UTMOS)", "higher = more natural", "predicted MOS (1–5)", (3.3, 4.5)),
              ("f0_hz", "Median pitch", "LJ recordings: 207 Hz", "Hz", (100, 225))]
    labels, y = [], 0.0
    groups = [("Full fine-tuning", FULL), ("Only the speaker embedding trains", EMBEDDING_ONLY)]
    headers = []
    for title, runs in groups:
        headers.append((y, title))
        y += 1
        for rid, run, _, label in runs:
            labels.append((y, f"{rid}   {label}", run, SPEAKER[rid]))
            y += 1
        y += 0.4
    fig = Figure(figsize=(7.4, 4.4))
    axes = fig.subplots(1, len(panels), sharey=True, gridspec_kw={"wspace": 0.32})
    for ax, (key, title, subtitle, xlabel, xlim) in zip(axes, panels):
        _panel(ax, title, subtitle, xlabel)
        real = _summary(results, "ground_truth")["metrics"][key]["mean"]
        ax.axvline(real, color=REF, lw=1, ls=(0, (3, 2)), zorder=1)
        for yy, _, run, speaker in labels:
            start = _summary(results, f"pretrained_spk{speaker}")["metrics"][key]["mean"]
            end = _summary(results, run)["metrics"][key]["mean"]
            if abs(end - start) > 0.12 * (xlim[1] - xlim[0]):  # shorter moves leave no room for an arrow
                ax.annotate("", xy=(end, yy), xytext=(start, yy), zorder=3,
                            arrowprops=dict(arrowstyle="-|>", color=AFTER, lw=1.3, shrinkA=4.2, shrinkB=3.2, mutation_scale=7))
            # the starting dot is larger, so it shows as a halo when a run barely moved
            ax.plot(start, yy, "o", ms=7, color=BEFORE, mec="white", mew=0.8, zorder=4)
            ax.plot(end, yy, "o", ms=4.8, color=AFTER, mec="white", mew=0.8, zorder=5)
        ax.set_xlim(*xlim)
    axes[0].set_yticks([h for h, _ in headers] + [yy for yy, *_ in labels])
    axes[0].set_yticklabels([t for _, t in headers] + [t for _, t, *_ in labels])
    for tick in axes[0].get_yticklabels()[:len(headers)]:
        tick.set_fontweight("bold")
    axes[0].set_ylim(y - 0.2, -0.8)
    handles = [Line2D([], [], marker="o", ls="", ms=7, color=BEFORE, mec="white", label="pretrained voice (before)"),
               Line2D([], [], marker="o", ls="", ms=4.8, color=AFTER, mec="white", label="after fine-tuning"),
               Line2D([], [], color=REF, lw=1, ls=(0, (3, 2)), label="real LJ recordings")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.6, -0.005))
    fig.subplots_adjust(left=0.25, right=0.99, top=0.88, bottom=0.15)
    return _save(fig, out)


def _epoch0(path: Optional[Path]) -> Dict[int, pd.Series]:
    """Pretrained-model validation losses by speaker ID, once scripts/pretrained_validation_losses.py has run."""
    if path is None or not path.is_file():
        return {}
    return {int(row["speaker_id"]): row for _, row in pd.read_csv(path).iterrows()}


@_styled_figure
def losses(runs_dir: Path, out: Path, epoch0_csv: Optional[Path] = None) -> Path:
    """One row of panels (mel, KL, duration): validation solid, training dashed, one colour per run."""
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, StrMethodFormatter
    from scripts.evaluation.training_logs import find_run_folder, load_history

    start = _epoch0(epoch0_csv)
    panels = [("mel", "Mel reconstruction loss", "spectrogram L1; lower = closer", False),
              ("kl", "KL loss (log scale)", "prior vs posterior; lower = better fit", True),
              ("duration", "Duration loss", "duration predictor; lower = better", False)]
    labels = {rid: label for rid, _, _, label in RUNS}
    fig = Figure(figsize=(7.4, 2.9))
    axes = fig.subplots(1, len(panels), gridspec_kw={"wspace": 0.32})
    handles = []
    for i, rid in enumerate(LOSS_RUNS):
        folder = find_run_folder(RUN_NAME[rid], None, runs_dir)
        if folder is None:
            continue
        train, val = load_history(folder)
        if SPEAKER[rid] in start:
            first = start[SPEAKER[rid]][["epoch", *(p[0] for p in panels)]].to_frame().T
            val = pd.concat([first, val], ignore_index=True)
        group = "full" if rid in ("B", "W") else "embedding only"
        for ax, (key, *_) in zip(axes, panels):
            ax.plot(val["epoch"], val[key], color=SERIES[i], lw=1.6, marker=MARKERS[i], ms=3.2, mec="white", mew=0.5, zorder=3)
            if train is not None and key in train:
                ax.plot(train["epoch"], train[key], color=SERIES[i], lw=1, ls=(0, (3, 2)), alpha=0.7, zorder=2)
        handles.append(Line2D([], [], color=SERIES[i], marker=MARKERS[i], ms=4, mec=SERIES[i], lw=1.6,  # no white gap: reads solid
                              label=f"{rid}  {labels[rid]}, {group}"))
    for ax, (key, title, subtitle, log) in zip(axes, panels):
        _panel(ax, title, subtitle, "epoch" + (" (0 = pretrained model)" if start else ""))
        ax.grid(axis="y", color="white", lw=1.2)
        ax.xaxis.set_major_locator(FixedLocator([0, 5, 10, 15] if start else [1, 5, 10, 15]))
        if log:
            ax.set_yscale("log")
            low, high = ax.get_ylim()
            ax.yaxis.set_major_locator(FixedLocator([t for t in (1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100) if low <= t <= high]))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="y", length=0, labelsize=7.5)
    handles += [Line2D([], [], color=INK2, lw=1.6, label="validation"), Line2D([], [], color=INK2, lw=1, ls=(0, (3, 2)), label="training")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.33)
    return _save(fig, out)


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    return out


def figures(results: Path = Path("results/experiments_v2"), runs_dir: Path = Path("outputs_v2"),
            epoch0: Path = Path("outputs_v2/pretrained_validation_losses.csv"),
            out_dir: Path = Path("results/experiments_v2/report/figures")) -> Dict[str, Path]:
    return {"before_after": before_after(results, out_dir / "before_after.png"),
            "losses": losses(runs_dir, out_dir / "losses.png", epoch0)}


def main(argv: Optional[List[str]] = None) -> Dict[str, Path]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default="results/experiments_v2")
    p.add_argument("--runs_dir", default="outputs_v2", help="the runs' training logs")
    p.add_argument("--epoch0", default="outputs_v2/pretrained_validation_losses.csv",
                   help="pretrained-model validation losses (scripts/pretrained_validation_losses.py)")
    p.add_argument("--out_dir", default="results/experiments_v2/report/figures")
    args = p.parse_args(argv)
    paths = figures(Path(args.results_dir), Path(args.runs_dir), Path(args.epoch0), Path(args.out_dir))
    for path in paths.values():
        print(f"Wrote {path}")
    return paths


if __name__ == "__main__":
    main()
