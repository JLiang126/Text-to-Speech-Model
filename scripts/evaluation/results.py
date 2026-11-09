"""Tables, figures and audio for the systems in the notebook's run table (section 1.4).

run_table() builds the run table from experiment_variables.csv (one row per experiment) and the
reference systems. Every other function takes that table and the IDs to show, so each table and figure
says exactly which runs and metrics it uses. Only score(), check_checkpoints() and make_examples() need
the checkpoints.
"""
from __future__ import annotations

import json
import textwrap
from functools import lru_cache
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluation.manifest import split_fingerprint
from scripts.evaluation.prompts import PROMPTS
from scripts.evaluation.stats import holm_adjust, paired_comparison
from scripts.evaluation.training_logs import load_history

BASE_MODEL = "ylacombe/vits-vctk-with-discriminator"
SCORED_SENTENCES = "e705482d0bf0"  # fingerprint of the 150 test sentences every system was scored on
EPOCH0 = Path("results/pretrained_validation_losses.csv")  # validation losses before fine-tuning
CHECKS = Path("results/checks.csv")

# metric: (column label, number format)
METRICS = {
    "mcd": ("MCD (dB) ↓", "{:.2f}"),
    "wer": ("WER (%) ↓", "{:.1f}"),
    "secs": ("Speaker sim. ↑", "{:.3f}"),
    "utmos": ("UTMOS ↑", "{:.2f}"),
    "pitch": ("Pitch (Hz)", "{:.0f}"),
    "duration": ("Duration ratio", "{:.2f}"),
    "val_mel": ("Val. mel ↓", "{:.3f}"),
    "val_kl": ("Val. KL ↓", "{:.2f}"),
    "moved": ("Vector moved (%)", "{:.1f}"),
}
SUMMARY_KEYS = {"mcd": "mcd_dtw", "secs": "secs", "utmos": "utmos", "pitch": "f0_hz", "duration": "dur_ratio"}

# Paired tests: per-sentence column, lower is better, label, format. Holm's correction always runs
# over all eight metrics (as in our earlier reports), even when a table shows fewer.
PAIRED = {
    "mcd": ("mcd_dtw", True, "MCD (dB)", "{:+.2f}"),
    "wer": ("wer", True, "WER (points)", "{:+.1f}"),
    "cer": ("cer", True, "CER (points)", "{:+.1f}"),
    "secs": ("secs", False, "Speaker sim.", "{:+.3f}"),
    "secs_paired": ("secs_paired", False, "Speaker sim. (same sentence)", "{:+.3f}"),
    "utmos": ("utmos", False, "UTMOS", "{:+.2f}"),
    "duration": ("dur_dev", True, "Duration error", "{:+.3f}"),
    "pitch": ("f0_dev", True, "Pitch error (Hz)", "{:+.1f}"),
}
GROUP_TITLES = {"full": "Full fine-tuning", "embedding only": "Only the speaker embedding trains"}

BEFORE, AFTER, REF = "#86b6ef", "#1c5cab", "#52514e"
INK, INK2, PANEL = "#0b0b0b", "#52514e", "#f6f6f3"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7d4fd6", "#8a8984"]
GOOD, BAD = "#dff3e6", "#fbe1e1"


def _available(checkpoint) -> bool:
    """The real recordings (no checkpoint), the pretrained model, or a weights file that is on this machine."""
    return pd.isna(checkpoint) or checkpoint == BASE_MODEL or Path(checkpoint).is_file()


def checkpoint_path(name: str) -> str:
    """A run's weights, model.<name>.safetensors, in outputs/<name>/run/ (from training or the shared zip of the
    runs); <name>.safetensors is accepted too."""
    folder = Path("outputs") / name / "run"
    for file in (f"model.{name}.safetensors", f"{name}.safetensors"):
        if (folder / file).is_file():
            return str(folder / file)
    return str(Path("outputs") / name / "run" / f"model.{name}.safetensors")


def run_table(experiments_csv, references: pd.DataFrame) -> pd.DataFrame:
    """One row per system, indexed by its ID: the reference systems (id, name, label, speaker), then every
    experiment in the CSV under its experiment_label, with where its scores, logs and checkpoint are."""
    exp = pd.read_csv(experiments_csv, dtype={"experiment_label": str}, encoding="utf-8-sig")
    exp = pd.DataFrame({
        "id": exp["experiment_label"],
        "name": exp["file_name"],
        "label": exp["short_description"].str.rstrip(".").str.replace(". 10X", ", 10×", regex=False),
        "speaker": exp["speaker_id"],
        "start": "P" + exp["speaker_id"].astype(str),
        "group": np.where(exp["freeze_mode"] == "unfreeze_speech_embeddings_only", "embedding only", "full"),
        "freeze": exp["freeze_mode"],
        "logs": "outputs/" + exp["file_name"] + "/run",
        "checkpoint": [checkpoint_path(name) for name in exp["file_name"]],
    })
    refs = references.assign(checkpoint=[None if pd.isna(s) else BASE_MODEL for s in references["speaker"]])
    runs = pd.concat([refs, exp], ignore_index=True).set_index("id")
    runs["speaker"] = runs["speaker"].astype("Int64")
    runs["results"] = "results/" + runs["name"]
    return runs[["name", "label", "speaker", "start", "group", "freeze", "checkpoint", "logs", "results"]]


# ------------------------------------------------------------------ loading
@lru_cache(maxsize=None)
def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summary(runs: pd.DataFrame, rid: str) -> dict:
    return _read_json(str(Path(runs.loc[rid, "results"]) / "summary.json"))


def per_sentence(runs: pd.DataFrame, rid: str) -> pd.DataFrame:
    """Every metric for every scored sentence, plus the duration error and the pitch error."""
    table = pd.read_csv(Path(runs.loc[rid, "results"]) / "per_utterance.csv")
    if "dur_ratio" in table:
        table["dur_dev"] = (table["dur_ratio"] - 1).abs()
        table["f0_dev"] = (table["f0_hz"] - table["f0_ref_hz"]).abs()
    return table


def losses(runs: pd.DataFrame, rid: str):
    """(training, validation) losses by epoch. The validation curve starts at epoch 0 when the pretrained
    model's losses for the run's speaker are in results/pretrained_validation_losses.csv."""
    train, val = load_history(runs.loc[rid, "logs"])
    if EPOCH0.is_file():
        start = pd.read_csv(EPOCH0)
        start = start[start["speaker_id"] == runs.loc[rid, "speaker"]]
        val = pd.concat([start[val.columns.intersection(start.columns)], val], ignore_index=True)
    return train, val


def value(runs: pd.DataFrame, rid: str, metric: str) -> float:
    """One number of the metric table (NaN where the metric does not apply to that system)."""
    row = runs.loc[rid]
    if metric == "wer":
        return 100 * summary(runs, rid)["corpus"]["wer"]["value"]
    if metric in SUMMARY_KEYS:
        found = summary(runs, rid)["metrics"].get(SUMMARY_KEYS[metric])
        return found["mean"] if found else np.nan
    if metric in ("val_mel", "val_kl"):
        loss = metric[4:]
        if isinstance(row["logs"], str):  # a run: its last epoch
            return float(load_history(row["logs"])[1][loss].iloc[-1])
        if row["checkpoint"] == BASE_MODEL and EPOCH0.is_file():  # a pretrained voice: epoch 0
            start = pd.read_csv(EPOCH0).set_index("speaker_id")
            return float(start.loc[row["speaker"], loss]) if row["speaker"] in start.index else np.nan
        return np.nan
    if metric == "moved":
        checks = pd.read_csv(CHECKS).set_index("experiment")
        return 100 * checks.loc[row["name"], "speaker_rel_l2"] if row["name"] in checks.index else np.nan
    raise KeyError(f"unknown metric {metric!r}, choose from {list(METRICS)}")


# ------------------------------------------------------------------ tables
def _styled(frame: pd.DataFrame, text_columns=("ID", "System", "Runs", "What changes")):
    text = [c for c in frame.columns if c in text_columns]
    left = [{"selector": f"th.col_heading.col{frame.columns.get_loc(c)}", "props": "text-align: left"} for c in text]
    return frame.style.hide(axis="index").set_properties(subset=text, **{"text-align": "left"}).set_table_styles(left)


def metric_table(runs: pd.DataFrame, ids, metrics):
    """Means over the scored sentences (WER pooled over all of them), one row per system."""
    frame = pd.DataFrame([{"ID": rid, "System": runs.loc[rid, "label"],
                           **{METRICS[m][0]: value(runs, rid, m) for m in metrics}} for rid in ids])
    return _styled(frame).format({METRICS[m][0]: METRICS[m][1] for m in metrics}, na_rep="–")


def paired_tests(runs: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """B minus A on the sentences both scored: mean difference, 95% bootstrap CI, Wilcoxon p (Holm-corrected
    over the eight metrics) and a verdict: +1 B clearly better, -1 clearly worse, 0 no clear difference.
    A difference is clear when its CI excludes 0 and the corrected p < 0.05."""
    both = per_sentence(runs, a).merge(per_sentence(runs, b), on="utt_id", suffixes=("_a", "_b"))
    rows = [{"metric": m, **paired_comparison(both[f"{col}_a"], both[f"{col}_b"])}
            for m, (col, _, _, _) in PAIRED.items()]
    for row, p in zip(rows, holm_adjust([r["wilcoxon_p"] for r in rows])):
        clear = (row["ci_low"] > 0 or row["ci_high"] < 0) and p < 0.05
        b_better = (row["mean_diff"] < 0) == PAIRED[row["metric"]][1]
        row.update(p_holm=p, verdict=(1 if b_better else -1) if clear else 0)
    return pd.DataFrame(rows).set_index("metric")


def _signed(fmt: str, x: float) -> str:
    """x with its sign, but no minus on a difference that rounds to zero ("-0.0")."""
    text = fmt.format(x)
    return fmt.format(0.0) if float(text) == 0 else text


def paired_table(runs: pd.DataFrame, pairs, metrics):
    """What each change did: for each (first, second, change), the second system minus the first, paired over
    the same sentences. Clear differences are bold, on green when the second system is better and on red when
    it is worse."""
    values, verdicts = [], []
    for a, b, change in pairs:
        tests = paired_tests(runs, a, b)
        scale = {m: 100 if m in ("wer", "cer") else 1 for m in metrics}
        values.append({"Runs": f"{a} → {b}", "What changes": change,
                       **{PAIRED[m][2]: _signed(PAIRED[m][3], scale[m] * tests.loc[m, "mean_diff"]) for m in metrics}})
        verdicts.append({"Runs": 0, "What changes": 0, **{PAIRED[m][2]: tests.loc[m, "verdict"] for m in metrics}})
    css = {1: f"background-color: {GOOD}; font-weight: bold", -1: f"background-color: {BAD}; font-weight: bold"}
    shading = pd.DataFrame(verdicts).apply(lambda col: col.map(css).fillna(""))
    return _styled(pd.DataFrame(values)).apply(lambda _: shading, axis=None)


def check_held_out(runs: pd.DataFrame, ids) -> str:
    """Every system was scored on the same sentences, and none of them was in a split the model trained or
    validated on (evaluate.py records the overlap with each training manifest and stops if there is any)."""
    tests = {rid: summary(runs, rid)["test_set"] for rid in ids}
    sets = {(t["n"], t["fingerprint"]) for t in tests.values()}
    overlap = {rid: sum(t["overlap"].values()) for rid, t in tests.items()}
    assert len(sets) == 1 and not any(overlap.values()), f"sentence sets {sets}, overlap {overlap}"
    n, fingerprint = sets.pop()
    return (f"All {len(ids)} systems were scored on the same {n} test sentences (fingerprint {fingerprint}), "
            "and none of them is in a training or validation split.")


# ------------------------------------------------------------------ figures
def _figure(width: float, height: float, n_panels: int):
    from matplotlib.figure import Figure  # not pyplot, so nothing leaks into the notebook's own plots

    fig = Figure(figsize=(width, height))
    return fig, fig.subplots(1, n_panels, gridspec_kw={"wspace": 0.32}, squeeze=False)[0]


def _panel(ax, title: str, subtitle: str, xlabel: str) -> None:
    ax.set_facecolor(PANEL)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.grid(color="white", lw=1.2)
    ax.set_axisbelow(True)
    ax.set_title(f"{title}\n", fontsize=9, loc="left", color=INK, fontweight="bold", pad=2)
    ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=7.5, color=INK2, va="bottom")
    ax.set_xlabel(xlabel, fontsize=7.5, color=INK2)
    ax.tick_params(length=0, labelsize=7.5, labelcolor=INK2)


BEFORE_AFTER = {  # metric: (title, subtitle, x label)
    "secs": ("Speaker similarity to LJ", "higher = more like LJ", "cosine similarity"),
    "utmos": ("Naturalness (UTMOS)", "higher = more natural", "predicted MOS (1-5)"),
    "pitch": ("Median pitch", "LJ recordings: 207 Hz", "Hz"),
    "mcd": ("Spectral distance (MCD)", "lower = closer to the recording", "dB"),
    "wer": ("Word error rate", "lower = more intelligible", "%"),
    "duration": ("Duration ratio", "1 = LJ's speaking rate", "synthetic / real length"),
}


def plot_before_after(runs: pd.DataFrame, ids, metrics=("secs", "utmos", "pitch"), reference="GT", labels=None):
    """Each run before fine-tuning (large light dot: the pretrained model with its starting speaker) and
    after it (small dark dot), against the real recordings (dashed line). One row per run in ids, in that
    order, and one panel per metric (any of mcd, wer, secs, utmos, pitch, duration). labels, e.g.
    {"2": "Base case"}, renames runs in this figure only; the run table's label is used otherwise."""
    from matplotlib.lines import Line2D

    rows, y, group = [], 0.0, None
    for rid in ids:
        if runs.loc[rid, "group"] != group:
            group = runs.loc[rid, "group"]
            rows.append((y, GROUP_TITLES.get(group, group), None))
            y += 1
        rows.append((y, textwrap.fill(f"{rid}   {(labels or {}).get(rid, runs.loc[rid, 'label'])}", 32), rid))
        y += 1
    fig, axes = _figure(7.4, 1.1 + 0.3 * len(rows), len(metrics))
    for ax, metric in zip(axes, metrics):
        _panel(ax, *BEFORE_AFTER[metric])
        ax.grid(axis="y", visible=False)
        # the real recordings have no MCD against themselves, and their duration ratio is 1 by definition
        real = 1.0 if metric == "duration" else value(runs, reference, metric)
        if not np.isnan(real):
            ax.axvline(real, color=REF, lw=1, ls=(0, (3, 2)), zorder=1)
        points = [(yy, value(runs, runs.loc[rid, "start"], metric), value(runs, rid, metric))
                  for yy, _, rid in rows if rid is not None]
        span = np.nanmax([v for _, start, end in points for v in (start, end, real)]) - \
            np.nanmin([v for _, start, end in points for v in (start, end, real)]) or 1.0
        for yy, start, end in points:
            if abs(end - start) > 0.08 * span:  # shorter moves leave no room for an arrow
                ax.annotate("", xy=(end, yy), xytext=(start, yy), zorder=3, arrowprops=dict(
                    arrowstyle="-|>", color=AFTER, lw=1.3, shrinkA=4.2, shrinkB=3.2, mutation_scale=7))
            # the starting dot is larger, so it shows as a halo when a run barely moved
            ax.plot(start, yy, "o", ms=7, color=BEFORE, mec="white", mew=0.8, zorder=4)
            ax.plot(end, yy, "o", ms=4.8, color=AFTER, mec="white", mew=0.8, zorder=5)
        ax.margins(x=0.08)
        ax.set_ylim(y - 0.4, -0.8)
        ax.set_yticks([])
    axes[0].set_yticks([yy for yy, _, _ in rows], [text for _, text, _ in rows])
    for tick, (_, _, rid) in zip(axes[0].get_yticklabels(), rows):
        tick.set_fontweight("normal" if rid else "bold")
        tick.set_color(INK)
    handles = [Line2D([], [], marker="o", ls="", ms=7, color=BEFORE, mec="white", label="pretrained voice (before)"),
               Line2D([], [], marker="o", ls="", ms=4.8, color=AFTER, mec="white", label="after fine-tuning"),
               Line2D([], [], color=REF, lw=1, ls=(0, (3, 2)), label="real LJ recordings")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    height = fig.get_figheight()
    longest = max(len(line) for _, text, _ in rows for line in text.splitlines())  # about 0.06 inch a character
    fig.subplots_adjust(left=(0.2 + 0.06 * longest) / 7.4, right=0.99, top=1 - 0.5 / height, bottom=0.65 / height)
    return fig


LOSSES = {  # loss: (title, subtitle, log scale)
    "mel": ("Mel reconstruction loss", "spectrogram L1; lower = closer", False),
    "kl": ("KL loss (log scale)", "prior vs posterior; lower = better fit", True),
    "duration": ("Duration loss", "duration predictor; lower = better", False),
}


def plot_losses(runs: pd.DataFrame, ids, names=("mel", "kl", "duration"), labels=None):
    """Losses by epoch on the validation split (solid) and the training split (dashed), one line per run in
    ids and one panel per loss (any of mel, kl, duration). Epoch 0 is the pretrained model with the run's
    starting speaker. Circles are full fine-tuning, squares embedding only. Each axis fits the validation
    curves, and a training loss above it is marked with a triangle at the top edge. labels renames runs in
    this figure only, as in plot_before_after."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator, StrMethodFormatter

    curves = {rid: losses(runs, rid) for rid in ids}
    # each run keeps its colour in every figure: the colours follow the run table's order of the runs
    colours = dict(zip([rid for rid in runs.index if isinstance(runs.loc[rid, "logs"], str)], SERIES))
    fig, axes = _figure(7.4, 2.9, len(names))
    for ax, name in zip(axes, names):
        title, subtitle, log = LOSSES[name]
        _panel(ax, title, subtitle, "epoch (0 = pretrained model)")
        # The axis fits the validation curves. A training loss above it is left out of its line and marked
        # with a triangle at the top edge, with its value.
        low = min(val[name].min() for _, val in curves.values())
        high = max(val[name].max() for _, val in curves.values())
        if log:
            ax.set_yscale("log")
            bottom, top, edge = low / 1.15, high * 1.15, high * 1.15 / 1.06
            ax.yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 5, 7, 10, 15, 20]))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
        else:
            bottom, top = low - 0.08 * (high - low), high + 0.08 * (high - low)
            edge = top - 0.03 * (top - bottom)
        ax.set_ylim(bottom, top)
        for rid in ids:
            train, val = curves[rid]
            colour, marker = colours[rid], "o" if runs.loc[rid, "group"] == "full" else "s"
            ax.plot(val["epoch"], val[name], color=colour, lw=1.6, marker=marker, ms=3.2, mec="white", mew=0.5)
            if train is None:
                continue
            ax.plot(train["epoch"], train[name].where(train[name] <= top), color=colour, lw=1, ls=(0, (3, 2)),
                    alpha=0.7)
            for epoch, loss in train.loc[train[name] > top, ["epoch", name]].itertuples(index=False):
                ax.plot(epoch, edge, "^", color=colour, ms=4.5, zorder=6)
                ax.annotate(f"run {rid} training: {loss:,.0f}", (epoch, edge), xytext=(5, 0),
                            textcoords="offset points", va="center", fontsize=6.5, color=INK2)
    handles = [Line2D([], [], color=colours[rid], marker="o" if runs.loc[rid, "group"] == "full" else "s", ms=4,
                      mec=colours[rid], lw=1.6, label=f"{rid}  {(labels or {}).get(rid, runs.loc[rid, 'label'])}")
               for rid in ids]
    handles += [Line2D([], [], color=INK2, lw=1.6, label="validation"),
                Line2D([], [], color=INK2, lw=1, ls=(0, (3, 2)), label="training")]
    legend_rows = (len(handles) + 2) // 3
    fig.set_figheight(2.4 + 0.22 * legend_rows)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    height = fig.get_figheight()
    fig.subplots_adjust(left=0.06, right=0.99, top=1 - 0.45 / height, bottom=(0.55 + 0.2 * legend_rows) / height)
    return fig


def speaker_ranking(ranking_csv, mel_csv=None) -> pd.DataFrame:
    """Every trained VCTK voice with its ECAPA similarity to LJ (scripts/rank_speakers.py) and, once computed,
    the pretrained model's mel reconstruction loss on LJ with that voice (scripts/pretrained_validation_losses.py
    --all_speakers), each with its rank."""
    table = pd.read_csv(ranking_csv)
    table = table[table["eligible"]].set_index("speaker_index")
    table = table[["vctk_id", "f0_median_hz", "similarity"]].rename(columns={"f0_median_hz": "pitch_hz"})
    table["similarity_rank"] = table["similarity"].rank(ascending=False).astype(int)
    if mel_csv is not None and Path(mel_csv).is_file():
        table["mel_loss"] = pd.read_csv(mel_csv).set_index("speaker_id")["mel"]
        table["mel_rank"] = table["mel_loss"].rank().astype("Int64")
    table["pitch_hz"] = table["pitch_hz"].round().astype("Int64")
    return table.round({"similarity": 3, "mel_loss": 3}).sort_values("similarity_rank")


def plot_speaker_ranking(table: pd.DataFrame, highlight=(7, 24)):
    """ECAPA similarity of every voice against its pitch and, once computed, its mel loss on LJ."""
    from scipy.stats import spearmanr

    panels = [("pitch_hz", "Median pitch of each voice", "Hz (LJ: 207 Hz)")]
    if "mel_loss" in table:
        panels.append(("mel_loss", "Mel loss on LJ before fine-tuning", "validation mel L1 (lower = closer)"))
    fig, axes = _figure(3.7 * len(panels), 3.0, len(panels))
    low = table["pitch_hz"] < 160
    for ax, (column, title, ylabel) in zip(axes, panels):
        rho, p = spearmanr(table["similarity"], table[column], nan_policy="omit")
        _panel(ax, title, f"{table[column].notna().sum()} voices; Spearman rho = {rho:.2f} (p = {p:.1g})",
               "ECAPA similarity to LJ (higher = closer)")
        ax.set_ylabel(ylabel, fontsize=7.5, color=INK2)
        if column == "pitch_hz":
            ax.axhline(207, color=REF, lw=1, ls=(0, (3, 2)), zorder=1)
        for mask, colour, label in ((low, SERIES[0], "below 160 Hz"), (~low, SERIES[1], "160 Hz or above")):
            ax.scatter(table.loc[mask, "similarity"], table.loc[mask, column], s=14, color=colour, label=label,
                       edgecolors="white", linewidths=0.4)
        for speaker in highlight:
            x, yy = table.loc[speaker, ["similarity", column]]
            right = x > table["similarity"].median()  # label on the inner side, so it stays in the panel
            ax.annotate(f"speaker {speaker}", (x, yy), xytext=(-8 if right else 8, 8), textcoords="offset points",
                        ha="right" if right else "left", fontsize=7.5, color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    axes[0].legend(frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.14 if len(panels) == 1 else 0.08, right=0.98, top=0.82, bottom=0.17, wspace=0.3)
    return fig


# ------------------------------------------------------------------ audio
def audio_grid(runs: pd.DataFrame, ids, prompts, wav_dir="wav_examples"):
    """One row of players per prompt and one column per system (the wavs from make_examples)."""
    from IPython.display import HTML, Audio

    head = "".join(f"<th style='text-align:center'>{escape(rid)}<br><small>{escape(runs.loc[rid, 'label'])}</small></th>"
                   for rid in ids)
    body = ""
    for n in prompts:  # the prompt on one row, its players on the next
        body += (f"<tr><td colspan='{len(ids)}' style='text-align:left; padding-top:10px'>"
                 f"<b>Prompt {n}:</b> {escape(PROMPTS[n - 1][1])}</td></tr>")
        cells = ""
        for rid in ids:
            wav = Path(wav_dir) / f"{runs.loc[rid, 'name']}_{n:02d}.wav"
            cells += f"<td>{Audio(filename=str(wav))._repr_html_()}</td>" if wav.is_file() else "<td>–</td>"
        body += f"<tr>{cells}</tr>"
    style = "<style>.audio-grid audio {width: 125px} .audio-grid td {vertical-align: middle}</style>"
    return HTML(f"{style}<table class='audio-grid'><tr>{head}</tr>{body}</table>")


def make_examples(runs: pd.DataFrame, ids, out_dir="wav_examples") -> None:
    """The 12 prompts for each system as out_dir/<name>_01.wav to _12.wav (scripts/synth.py). Systems that
    already have them are skipped, and so are runs whose checkpoint is not on this machine."""
    from contextlib import redirect_stdout
    from io import StringIO

    from scripts import synth

    made, missing = [], []
    for rid in ids:
        row = runs.loc[rid]
        if all((Path(out_dir) / f"{row['name']}_{n:02d}.wav").is_file() for n in range(1, len(PROMPTS) + 1)):
            continue
        if pd.isna(row["checkpoint"]) or not _available(row["checkpoint"]):
            missing.append(rid)
            continue
        with redirect_stdout(StringIO()):  # synth.py prints one line per prompt
            synth.main(["--prompts", "--exp", row["name"], "--model", row["checkpoint"],
                        "--speaker_id", str(int(row["speaker"])), "--out_dir", str(out_dir)])
        made.append(rid)
    if made:
        print(f"Wrote the {len(PROMPTS)} prompts for {', '.join(made)} to {out_dir}/")
    if missing:
        print(
            f"No checkpoint on this machine for {', '.join(missing)}: put "
            "group55_VITS_checkpoints.zip in the repository root and rerun the notebook"
        )


# ------------------------------------------------------------------ scoring (slow, needs the checkpoints)
def _scored(runs: pd.DataFrame, rid: str) -> bool:
    done = Path(runs.loc[rid, "results"]) / "summary.json"
    return done.is_file() and json.loads(done.read_text(encoding="utf-8"))["test_set"]["fingerprint"] == SCORED_SENTENCES


def score(runs: pd.DataFrame, ids, data_root="dataset/LJSpeech-1.1", device="auto") -> None:
    """Scores each system with scripts/evaluate.py. Systems already scored on the same sentences are skipped,
    and so are runs whose checkpoint is not on this machine.

    The sentences are the first 150 of the test part of a 90/5/5 cut of 1.1's seeded shuffle (so they are
    in 1.1's test split too), and the speaker reference is 50 clips from its training part. Rebuilding that
    cut keeps a re-run identical to the saved scores.
    """
    todo = [rid for rid in ids if not _scored(runs, rid) and _available(runs.loc[rid, "checkpoint"])]
    if not todo:
        return
    from scripts import evaluate
    from scripts.DataConfig import DataConfig
    from scripts.DataLoader import build_splits

    split = Path("dataset/eval_split")
    if not (split / "test_manifest.csv").is_file():
        build_splits(DataConfig(data_root=data_root, output_split_dir=split, split_ratios=(0.90, 0.05, 0.05)))
    test = split / "test_manifest.csv"
    if split_fingerprint(pd.read_csv(test)["utt_id"].astype(str).iloc[:150]) != SCORED_SENTENCES:
        raise RuntimeError(f"{test} does not start with the 150 sentences the saved scores used")
    for rid in todo:
        row, folder = runs.loc[rid], Path(runs.loc[rid, "results"])
        argv = ["--exp", row["name"], "--manifest", str(test), "--limit", "150", "--data_root", data_root,
                "--metrics", "mcd,wer,secs,utmos", "--no_save_wavs", "--device", device, "--out_dir", str(folder)]
        if pd.isna(row["checkpoint"]):  # the real recordings
            argv.append("--ground_truth")
        else:
            argv += ["--model", row["checkpoint"], "--speaker_id", str(int(row["speaker"]))]
        evaluate.main(argv)
    _read_json.cache_clear()


def check_checkpoints(runs: pd.DataFrame, ids) -> pd.DataFrame:
    """Compares each run's weights with the pretrained model: did only its speaker row train, and did the
    frozen modules stay unchanged? New rows are added to results/checks.csv; runs already in it, and runs
    whose checkpoint is not on this machine, are skipped."""
    from scripts.evaluation.checkpoints import check_experiment, state_of
    from scripts.evaluation.synthesizer import load_hf_vits

    checked = set(pd.read_csv(CHECKS)["experiment"]) if CHECKS.is_file() else set()
    todo = [rid for rid in ids if runs.loc[rid, "name"] not in checked and _available(runs.loc[rid, "checkpoint"])]
    if not todo:
        return pd.read_csv(CHECKS) if CHECKS.is_file() else pd.DataFrame()
    base = state_of(load_hf_vits(BASE_MODEL, BASE_MODEL)[0])
    rows = []
    for rid in todo:
        row = runs.loc[rid]
        state = state_of(load_hf_vits(row["checkpoint"], BASE_MODEL)[0])
        rows.append({"experiment": row["name"], "speaker_id": int(row["speaker"]), "freeze_mode": row["freeze"],
                     **check_experiment(state, base, int(row["speaker"]), row["freeze"])})
    if not rows:
        return pd.read_csv(CHECKS) if CHECKS.is_file() else pd.DataFrame()
    new = pd.DataFrame(rows).set_index("experiment")
    old = pd.read_csv(CHECKS).set_index("experiment") if CHECKS.is_file() else None
    checks = new if old is None else pd.concat([old.drop(new.index, errors="ignore"), new])
    checks.to_csv(CHECKS)
    return checks
