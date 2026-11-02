#!/usr/bin/env python
"""Evaluates every fine-tuning experiment and builds the report's tables and figures.

One command for the whole batch. It reads the experiment list (experiments/experiments.csv:
one row per checkpoint with its settings) and the report plan (experiments/report_plan.json:
which systems and pairs answer which question), then:

1. checks each checkpoint against the pretrained model: did the configured speaker row
   train, and are the frozen modules unchanged? With the training logs (--runs_dir), also:
   did the run use the settings its config says, and was it trained on a split whose
   test sentences are the ones scored here?                          -> report/checks.csv
2. scores every system on the test split with evaluate.py: each experiment, the real
   recordings (ceiling) and the pretrained model with a run's speaker index (floor).
   Finished systems are skipped, so the command resumes after a stop; a system scored on
   other sentences (an older split) counts as unfinished.             -> <results>/<system>/
3. times one system (every run has the same architecture, so one timing covers all)
4. synthesises the 12 fixed prompts per system for listening          -> <results>/<system>/prompts/
5. writes the report: per-question tables, paired tests and figures, and the validation
   loss curves of the runs in each section (when --runs_dir has them) -> <results>/report/report.md

Run from the repo root, with the checkpoints where experiments.csv says (checkpoints/) and the
test split the runs trained with (rebuild it with `python scripts/data_checks.py`):

  python scripts/evaluate_experiments.py                  # everything, resumable
  python scripts/evaluate_experiments.py --limit 20       # quick preview on 20 test sentences
  python scripts/evaluate_experiments.py --report_only    # rebuild the report from finished results

New checkpoints: add a row to experiments.csv (and the name to a section of report_plan.json),
then run again; only the new systems are scored.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # lets `python scripts/evaluate_experiments.py` import the package
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts import compare, evaluate  # noqa: E402
from scripts import synth as synth_script  # noqa: E402
from scripts.evaluation import training_logs  # noqa: E402
from scripts.evaluation.checkpoints import check_experiment, state_of  # noqa: E402
from scripts.evaluation.manifest import overlap_with, split_fingerprint  # noqa: E402
from scripts.evaluation.synthesizer import DEFAULT_BASE_MODEL, load_hf_vits  # noqa: E402

GROUND_TRUTH = "ground_truth"
FLOOR = re.compile(r"^pretrained_spk(\d+)$")
QUALITY_METRICS = "mcd,wer,secs,utmos"
METRIC_LABELS = {"mcd_dtw": ("MCD-DTW (dB)", 1.0, 2), "wer": ("WER (pp)", 100.0, 1), "cer": ("CER (pp)", 100.0, 1),
                 "secs": ("Speaker sim.", 1.0, 3), "secs_paired": ("Speaker sim. (paired)", 1.0, 3),
                 "utmos": ("UTMOS", 1.0, 2), "dur_dev": ("Duration error", 1.0, 3),
                 "f0_dev": ("Pitch error (Hz)", 1.0, 1)}
# settings compared between a run's config (what we think it used) and its training_config.json
SETTING_KEYS = ("speaker_id", "learning_rate", "use_linear_lr_schedule", "freeze_mode", "speaker_embedding_lr_ratio",
                "encoder_layerdrop", "weight_fmaps", "num_train_epochs", "seed", "subset_size", "split_ratios",
                "min_duration_sec", "max_duration_sec")


# ------------------------------------------------------------------ inputs
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", default="experiments/experiments.csv", help="one row per checkpoint")
    p.add_argument("--plan", default="experiments/report_plan.json", help="sections, pairs and order")
    p.add_argument("--results_dir", default="results/experiments")
    p.add_argument("--base_model", default=DEFAULT_BASE_MODEL,
                   help="pretrained model the runs started from: floors, config, tokenizer, checks")
    p.add_argument("--manifest", default="dataset/splits/test_manifest.csv", help="test split manifest")
    p.add_argument("--data_root", default="dataset/LJSpeech-1.1", help="fallback folder for wav paths")
    p.add_argument("--runs_dir", default="outputs",
                   help="training run folders (<runs_dir>/<config output_dir name>/run/*_loss_history.csv, "
                        "training_config.json, ../splits/): loss curves, settings and split checks")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    p.add_argument("--limit", type=int, default=None, help="only the first N test sentences (CPU budget)")
    p.add_argument("--only", default=None, help="comma list of systems to score (default: all in the plan)")
    p.add_argument("--force", action="store_true", help="score again even if a system is finished")
    p.add_argument("--report_only", action="store_true", help="skip scoring; rebuild the report from results")
    p.add_argument("--skip_checks", action="store_true", help="don't compare checkpoints with the base model")
    p.add_argument("--skip_prompts", action="store_true", help="don't synthesise the 12 listening prompts")
    p.add_argument("--skip_timing", action="store_true", help="don't time the timing system")
    return p


def load_experiments(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"experiment", "speaker_id", "checkpoint"} - set(table.columns)
    if missing:
        raise SystemExit(f"{path} lacks the column(s) {sorted(missing)}")
    table["experiment"] = table["experiment"].str.strip()
    duplicated = table["experiment"][table["experiment"].duplicated()].tolist()
    if duplicated:
        raise SystemExit(f"{path}: experiment names must be unique; repeated: {duplicated}")
    bad = [n for n in table["experiment"] if n == GROUND_TRUTH or FLOOR.match(n)]
    if bad:
        raise SystemExit(f"{path}: {bad} are reserved names (ground_truth, pretrained_spk<k>)")
    return table.set_index("experiment", drop=False)


def load_plan(path: Path, experiments: pd.DataFrame) -> Dict[str, object]:
    plan = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    plan.setdefault("baseline", experiments.index[0])
    plan.setdefault("sections", [])
    order = list(plan.get("order", []))
    order += [plan["baseline"], GROUND_TRUTH]
    for section in plan["sections"]:
        order += section.get("systems", [])
    order += list(experiments.index)  # every checkpoint is scored, listed in a section or not
    plan["order"] = list(dict.fromkeys(order))
    for name in plan["order"]:
        system_spec(name, experiments, "unused")  # fails early on a typo in the plan
    return plan


def resolve_checkpoint(name: str, listed: str, run_folder: Optional[Path] = None) -> str:
    """The listed checkpoint, else the names Michael's training writes (model.<exp>.safetensors,
    in checkpoints/ or the run folder)."""
    candidates = [Path(listed), Path(listed).with_name(f"model.{name}.safetensors")]
    if run_folder is not None:
        candidates += [run_folder / "run" / f"model.{name}.safetensors", run_folder / "run"]
    for candidate in candidates:
        if candidate.is_file() or (candidate.is_dir() and (candidate / "config.json").is_file()):
            return str(candidate)
    return listed


def run_folder(name: str, experiments: pd.DataFrame, runs_dir: Optional[str]) -> Optional[Path]:
    """The experiment's training run folder (loss histories), if present."""
    if name not in experiments.index:
        return None
    config = experiments.loc[name].get("config") or None
    return training_logs.find_run_folder(name, config, runs_dir)


def system_spec(name: str, experiments: pd.DataFrame, base_model: str) -> Dict[str, object]:
    """What to score for a system name: an experiment, the real recordings, or a pretrained floor."""
    if name == GROUND_TRUTH:
        return {"name": name, "kind": "ground_truth"}
    floor = FLOOR.match(name)
    if floor:
        return {"name": name, "kind": "floor", "model": base_model, "speaker_id": int(floor.group(1))}
    if name in experiments.index:
        row = experiments.loc[name]
        return {"name": name, "kind": "experiment", "model": resolve_checkpoint(name, row["checkpoint"]),
                "speaker_id": int(row["speaker_id"])}
    raise SystemExit(f"Unknown system '{name}': not in the experiments CSV, not ground_truth, not pretrained_spk<k>")


def expected_test_set(args) -> Tuple[int, str]:
    """(number of sentences, fingerprint) that evaluate.py will score with these arguments."""
    ids = pd.read_csv(args.manifest, usecols=["utt_id"])["utt_id"].astype(str)
    ids = ids.iloc[: args.limit] if args.limit else ids
    return len(ids), split_fingerprint(ids)


def is_finished(folder: Path, expected: Tuple[int, str]) -> bool:
    """Scored on exactly these sentences, with every metric available."""
    path = folder / "summary.json"
    if not path.is_file():
        return False
    summary = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = (summary.get("test_set") or {}).get("fingerprint")
    return summary.get("n_utterances") == expected[0] and fingerprint == expected[1] \
        and bool(summary.get("metrics")) and not summary.get("unavailable_metrics")


def read_checks(path: Path) -> Optional[pd.DataFrame]:
    """report/checks.csv, or None if absent or empty (an interrupted run leaves it empty)."""
    try:
        return pd.read_csv(path) if path.is_file() else None
    except pd.errors.EmptyDataError:
        return None


def run_split_manifests(folder: Optional[Path]) -> List[Path]:
    """The train and validation manifests a run trained with (its splits/ folder), if saved."""
    if folder is None:
        return []
    return [m for m in (folder / "splits" / "train_manifest.csv", folder / "splits" / "val_manifest.csv") if m.is_file()]


# ------------------------------------------------------------------ steps 1-4
def training_record_checks(name: str, row: pd.Series, folder: Optional[Path], test_ids: Sequence[str]
                           ) -> Dict[str, object]:
    """From the run's own files: settings that differ from its config, and how many of the
    scored test sentences were in the split it trained on."""
    if folder is None:
        return {"run_logs": "not found", "settings_differ": "", "test_in_training": float("nan")}
    out: Dict[str, object] = {"run_logs": str(folder)}
    actual = training_logs.run_settings(folder)
    claimed = {}
    if row.get("config") and Path(row["config"]).is_file():
        import yaml
        claimed = yaml.safe_load(Path(row["config"]).read_text(encoding="utf-8")) or {}
    differ = training_logs.settings_mismatches(actual, claimed, SETTING_KEYS) if actual else {}
    out["settings_differ"] = "; ".join(f"{k}: run {a} vs config {b}" for k, (a, b) in differ.items()) \
        if actual else "no training_config.json"
    manifests = run_split_manifests(folder)
    out["test_in_training"] = sum(overlap_with(test_ids, m) for m in manifests) if manifests else float("nan")
    for key, (a, b) in differ.items():
        print(f"[checks] {name}: {key} = {a} in training_config.json but {b} in {row.get('config')}")
    if manifests and out["test_in_training"]:
        print(f"[checks] {name}: {out['test_in_training']} scored test sentences are in its training split!")
    return out


def run_checks(experiments: pd.DataFrame, base_model: str, out_dir: Path, args=None) -> pd.DataFrame:
    """Step 1: every checkpoint against the base model (no audio, a minute or two), and the
    run's settings and split against its config and the test sentences (when its logs exist)."""
    print(f"[checks] loading the base model {base_model}")
    base, _ = load_hf_vits(base_model, base_model)
    base_state = state_of(base)
    del base
    runs_dir = getattr(args, "runs_dir", None)
    test_ids: List[str] = []
    if args is not None and Path(args.manifest).is_file():
        ids = pd.read_csv(args.manifest, usecols=["utt_id"])["utt_id"].astype(str)
        test_ids = (ids.iloc[: args.limit] if args.limit else ids).tolist()
    rows = []
    for name, row in experiments.iterrows():
        checkpoint = resolve_checkpoint(name, row["checkpoint"])
        if not Path(checkpoint).exists():
            print(f"[checks] {name}: checkpoint not found ({row['checkpoint']}), skipped")
            continue
        model, _ = load_hf_vits(checkpoint, base_model)
        result = check_experiment(state_of(model), base_state, int(row["speaker_id"]),
                                  row.get("freeze_mode", "none") or "none")
        del model
        gc.collect()
        result.update(training_record_checks(name, row, run_folder(name, experiments, runs_dir), test_ids))
        rows.append({"experiment": name, "speaker_id": int(row["speaker_id"]),
                     "freeze_mode": row.get("freeze_mode", "none"), **result})
        print(f"[checks] {name}: trained row {result.get('trained_row')} "
              f"({100 * result.get('trained_row_change', float('nan')):.2f}% change), "
              f"unchanged: {result['unchanged_modules']}, speaker ok: {result.get('speaker_ok')}, "
              f"freeze ok: {result['freeze_ok']}")
    checks = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    checks.to_csv(out_dir / "checks.csv", index=False)
    return checks


def score_system(spec: Dict[str, object], args, folder: Path, train_manifests: Sequence[Path] = ()) -> None:
    """Step 2: one system through evaluate.py. ``train_manifests``: the run's own training split,
    checked (with the manifests beside --manifest) for overlap with the test sentences."""
    argv = ["--exp", spec["name"], "--manifest", args.manifest, "--data_root", args.data_root,
            "--device", args.device, "--out_dir", str(folder), "--metrics", QUALITY_METRICS, "--no_save_wavs"]
    if train_manifests:
        beside = [Path(args.manifest).with_name(f"{s}_manifest.csv") for s in ("train", "val")]
        for manifest in list(train_manifests) + [m for m in beside if m.is_file()]:
            argv += ["--train_manifest", str(manifest)]
    if spec["kind"] == "ground_truth":
        argv.append("--ground_truth")
    else:
        argv += ["--model", str(spec["model"]), "--base_model", args.base_model, "--speaker_id", str(spec["speaker_id"])]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    evaluate.main(argv)


def time_system(spec: Dict[str, object], args, folder: Path) -> None:
    """Step 3: latency/RTF for one system (the architecture is identical across runs)."""
    evaluate.main(["--exp", spec["name"], "--model", str(spec["model"]), "--base_model", args.base_model,
                   "--speaker_id", str(spec["speaker_id"]), "--device", args.device, "--out_dir", str(folder),
                   "--metrics", "rtf"])


def synthesise_prompts(spec: Dict[str, object], args, folder: Path) -> None:
    """Step 4: the 12 fixed prompts, for listening."""
    synth_script.main(["--prompts", "--exp", spec["name"], "--model", str(spec["model"]),
                       "--base_model", args.base_model, "--speaker_id", str(spec["speaker_id"]),
                       "--device", args.device, "--out_dir", str(folder / "prompts")])


# ------------------------------------------------------------------ step 5: report
def _fmt_diff(row: pd.Series) -> str:
    _, scale, digits = METRIC_LABELS.get(row["metric"], (row["metric"], 1.0, 3))
    return (f"{row['mean_diff'] * scale:+.{digits}f} "
            f"[{row['ci_low'] * scale:+.{digits}f}, {row['ci_high'] * scale:+.{digits}f}]")


def paired_markdown(tests: pd.DataFrame, only_clear: bool = False) -> str:
    """Paired tests as a markdown table: B minus A, on the sentences both systems share."""
    if tests is None or tests.empty:
        return "_No paired tests (a system in each pair is not scored yet)._\n"
    rows = tests if not only_clear else tests[tests["verdict"] != "no clear difference"]
    if rows.empty:
        return "_No clear differences (every CI includes 0 or Holm p >= 0.05)._\n"
    lines = ["| A | B | metric | B - A [95% CI] | Holm p | d_z | verdict |", "|---|---|---|---|---|---|---|"]
    for _, row in rows.iterrows():
        label = METRIC_LABELS.get(row["metric"], (row["metric"],))[0]
        p_holm = row.get("wilcoxon_p_holm", float("nan"))
        p_text = "-" if pd.isna(p_holm) else ("<0.001" if p_holm < 0.001 else f"{p_holm:.3f}")
        dz = row.get("cohens_dz", float("nan"))
        lines.append(f"| {row['a']} | {row['b']} | {label} | {_fmt_diff(row)} | {p_text} | "
                     f"{'-' if pd.isna(dz) else f'{dz:.2f}'} | {row['verdict']} |")
    return "\n".join(lines) + "\n"


def run_compare(folders: Sequence[Path], pairs: Sequence[Sequence[str]], out: Path) -> Dict[str, object]:
    argv = [str(f) for f in folders]
    for a, b in pairs:
        argv += ["--pair", a, b]
    return compare.main(argv + ["--out", str(out)])


def plot_before_after(pairs: Sequence[Sequence[str]], folders: Dict[str, Path], path: Path) -> bool:
    """Each run's starting point (pretrained, same speaker index) next to its end point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [(a, b) for a, b in pairs if a in folders and b in folders]
    if not pairs:
        return False
    systems = {name: compare.System(folders[name]) for pair in pairs for name in pair}
    truth = compare.System(folders[GROUND_TRUTH]) if GROUND_TRUTH in folders else None
    panels = [("secs", "Speaker similarity to LJ (higher is better)", 1.0),
              ("utmos", "UTMOS (higher is better)", 1.0),
              ("wer", "WER % (lower is better)", 100.0),
              ("f0_hz", "Median pitch (Hz)", 1.0)]
    panels = [p for p in panels if any(not np.isnan(s.value(p[0])[0]) for s in systems.values())]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 0.55 * len(pairs) + 1.8),
                             sharey=True, facecolor=compare.SURFACE)
    axes = np.atleast_1d(axes)
    y = np.arange(len(pairs))[::-1]
    for ax, (key, title, scale) in zip(axes, panels):
        compare._style(ax)
        for yi, (start, end) in zip(y, pairs):
            p0, p1 = systems[start].value(key)[0] * scale, systems[end].value(key)[0] * scale
            lo, hi = (v * scale for v in systems[end].value(key)[1:])
            if np.isnan(p0) or np.isnan(p1):
                continue
            ax.annotate("", xy=(p1, yi), xytext=(p0, yi),
                        arrowprops=dict(arrowstyle="-|>", color=compare.SERIES[0], lw=1.6, shrinkA=4, shrinkB=4))
            ax.plot([lo, hi], [yi, yi], color=compare.SERIES[0], lw=1, alpha=0.5)
            ax.plot(p0, yi, "o", color=compare.REFERENCE, markersize=6)
            ax.plot(p1, yi, "o", color=compare.SERIES[0], markersize=7, markeredgecolor=compare.SURFACE)
        if truth is not None and not np.isnan(truth.value(key)[0]):
            ax.axvline(truth.value(key)[0] * scale, color=compare.INK_2, lw=1, ls="--")
        ax.set_title(title, fontsize=9.5, color=compare.INK, loc="left")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{end}\n(from {start})" for start, end in pairs], fontsize=8.5, color=compare.INK)
    fig.suptitle("Before vs after fine-tuning: grey = pretrained with the same speaker index, blue = fine-tuned "
                 "(line = 95% CI), dashed = real recordings", x=0.01, ha="left", fontsize=10, color=compare.INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=compare.SURFACE)
    plt.close(fig)
    return True


def checks_markdown(checks: pd.DataFrame) -> str:
    if checks is None or checks.empty:
        return "_Checks not run (use the default, without --skip_checks)._\n"
    lines = ["| experiment | speaker_id | most-changed speaker row (change) | next row (change) | speaker row OK | "
             "unchanged modules | freeze OK | settings vs config | test sentences in its training split |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in checks.iterrows():
        def pct(v):
            return "-" if pd.isna(v) else f"{100 * float(v):.2f}%"
        freeze_ok = "-" if pd.isna(r.get("freeze_ok")) else ("yes" if r.get("freeze_ok") in (True, "True") else "NO")
        speaker_ok = "yes" if r.get("speaker_ok") in (True, "True") else "NO"
        if pd.isna(r.get("run_logs")) or r.get("run_logs") == "not found":
            settings, leaked = "no run logs", "-"
        else:
            settings = r.get("settings_differ") if isinstance(r.get("settings_differ"), str) and r.get("settings_differ") \
                else "as configured"
            leaked = "split not saved" if pd.isna(r.get("test_in_training")) else str(int(r.get("test_in_training")))
        lines.append(f"| {r['experiment']} | {r['speaker_id']} | {r.get('trained_row')} ({pct(r.get('trained_row_change'))})"
                     f" | {r.get('next_row')} ({pct(r.get('next_row_change'))}) | {speaker_ok} | "
                     f"{r['unchanged_modules']} | {freeze_ok} | {settings} | {leaked} |")
    return "\n".join(lines) + "\n"


LOSS_COLUMNS = (("mel_final", "val mel, final", 3), ("mel_min", "val mel, min", 3), ("mel_best_epoch", "epoch of min", 0),
                ("mel_slope_last3", "val mel slope, last 3 epochs", 4), ("kl_final", "val KL, final", 3),
                ("duration_final", "val duration, final", 3), ("mel_gap_final", "val minus train mel", 3))


def losses_markdown(table: pd.DataFrame) -> str:
    """Validation-loss summary per run (see scripts/evaluation/training_logs.py)."""
    columns = [c for c in LOSS_COLUMNS if c[0] in table.columns]
    lines = ["| run | epochs | " + " | ".join(label for _, label, _ in columns) + " |",
             "|---|---|" + "---|" * len(columns)]
    for _, r in table.iterrows():
        cells = ["-" if pd.isna(r[key]) else (f"{int(r[key])}" if digits == 0 else f"{r[key]:.{digits}f}")
                 for key, _, digits in columns]
        lines.append(f"| {r['experiment']} | {r['epochs']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def section_losses(members: Sequence[str], experiments: pd.DataFrame, runs_dir: Optional[str], report: Path,
                   name: str, title: str) -> List[str]:
    """Loss curves and table for the section's runs whose training logs are present."""
    configs = {n: (experiments.loc[n].get("config") or None) for n in members if n in experiments.index}
    histories = training_logs.load_histories(configs, runs_dir)
    if not histories:
        return []
    table = training_logs.loss_table(histories)
    table.to_csv(report / f"{name}_losses.csv", index=False)
    training_logs.plot_loss_curves(histories, report / f"{name}_losses.png", title=title)
    missing = [n for n in configs if n not in histories]
    return ["\n**Training logs.** Validation losses are reconstruction losses on the validation split with the model "
            "in training mode (posterior encoder, alignment from the real recording), so they complement the test "
            "metrics above rather than replace them; only mel, KL and duration are comparable between runs. A "
            "negative slope over the last epochs means the run was still improving when it stopped; under a linear "
            "schedule the learning rate is near zero by then, so a flat end is not evidence of convergence."
            + (f" No logs found for: {', '.join(missing)}." if missing else "") + "\n",
            losses_markdown(table), f"\n![Loss curves]({name}_losses.png)\n"]


def config_markdown(experiments: pd.DataFrame) -> str:
    columns = [c for c in experiments.columns if c != "checkpoint"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in experiments[columns].itertuples(index=False)]
    return "\n".join(lines) + "\n"


def build_report(args, plan: Dict[str, object], experiments: pd.DataFrame, results: Path,
                 checks: Optional[pd.DataFrame]) -> Path:
    report = results / "report"
    report.mkdir(parents=True, exist_ok=True)
    folders = {name: results / name for name in plan["order"] if (results / name / "summary.json").is_file()}
    baseline = plan["baseline"]
    summaries = {n: json.loads((f / "summary.json").read_text(encoding="utf-8")) for n, f in folders.items()}
    n_sentences = sorted({s.get("n_utterances") for s in summaries.values()})
    test_sets = sorted({str((s.get("test_set") or {}).get("fingerprint")) for s in summaries.values()})
    md = [f"# Fine-tuning experiments: evaluation report\n",
          f"Generated {datetime.now():%Y-%m-%d %H:%M}. Test sentences per system: "
          f"{', '.join(str(n) for n in n_sentences) or '-'} (`{args.manifest}`, test-set fingerprint "
          f"{', '.join(test_sets) or '-'}). Base model: `{args.base_model}`.\n",
          *(["**Warning: the systems were scored on different test sets** (fingerprints above); "
             "re-score with the same manifest before comparing.\n"] if len(test_sets) > 1 else []),
          "Every number is a mean with a 95% bootstrap CI over the test sentences; WER and CER are corpus rates. "
          "The ceiling is the real recordings (`ground_truth`); a floor is the pretrained model with a run's "
          "speaker index (`pretrained_spk<k>`). Paired tests compare B with A on the same sentences "
          "(B minus A, Wilcoxon with Holm correction across metrics, effect size d_z). "
          f"Unfinished systems are left out: {', '.join(n for n in plan['order'] if n not in folders) or 'none'}.\n",
          "## Experiments\n", config_markdown(experiments),
          "## Checks: did each run do what its settings say?\n",
          "Each checkpoint compared with the base model. The trained speaker row should move far more than any "
          "other row (the rest only shrink through weight decay); a frozen module should be unchanged.\n",
          checks_markdown(checks)]

    # overview order: references first (ceiling, floors by speaker), then the baseline, then the experiments
    floors = sorted((n for n in folders if FLOOR.match(n)), key=lambda n: int(FLOOR.match(n).group(1)))
    names = [n for n in [GROUND_TRUTH] + floors + [baseline] + list(experiments.index) if n in folders]
    names = list(dict.fromkeys(names))
    if names:
        pairs = [(baseline, n) for n in names if n in experiments.index and n != baseline and baseline in folders]
        result = run_compare([folders[n] for n in names], pairs, report / "all")
        md += ["## Overview: every system\n", (report / "all" / "comparison_table.md").read_text(encoding="utf-8"),
               "\n![Every system](all/comparison.png)\n",
               f"\nClear differences from the baseline (`{baseline}`):\n\n",
               paired_markdown(result["tests"], only_clear=True)]

    for section in plan["sections"]:
        members = [n for n in section.get("systems", []) if n in folders]
        md.append(f"## {section.get('title', section['name'])}\n")
        if section.get("description"):
            md.append(section["description"] + "\n")
        if len(members) < 2:
            md.append("_Not enough finished systems yet._\n")
            continue
        pairs = [p for p in section.get("pairs", []) if p[0] in folders and p[1] in folders]
        result = run_compare([folders[n] for n in members], pairs, report / section["name"])
        md += [(report / section["name"] / "comparison_table.md").read_text(encoding="utf-8"),
               f"\n![{section.get('title', section['name'])}]({section['name']}/comparison.png)\n\n",
               paired_markdown(result["tests"])]
        if section.get("before_after") and plot_before_after(section["before_after"], folders,
                                                              report / f"{section['name']}_before_after.png"):
            md.append(f"\n![Before vs after fine-tuning]({section['name']}_before_after.png)\n")
        md += section_losses(section.get("systems", []), experiments, getattr(args, "runs_dir", None), report,
                             section["name"], section.get("title", section["name"]))

    listen = [n for n in names if (results / n / "prompts").is_dir()]
    if listen:
        md += ["## Listening\n", "The 12 fixed prompts per system are in `<results>/<system>/prompts/` "
               f"({', '.join(listen)}).\n"]
    path = report / "report.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path


# ------------------------------------------------------------------ main
def main(argv: Optional[List[str]] = None) -> Path:
    args = build_parser().parse_args(argv)
    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    experiments = load_experiments(Path(args.experiments))
    plan = load_plan(Path(args.plan), experiments)
    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    report_dir = results / "report"

    checks = read_checks(report_dir / "checks.csv")
    if not args.report_only and not args.skip_checks:
        checks = run_checks(experiments, args.base_model, report_dir, args)

    if not args.report_only:
        wanted = [s.strip() for s in args.only.split(",")] if args.only else plan["order"]
        expected = expected_test_set(args)
        todo = [n for n in plan["order"] if n in wanted]
        failures: Dict[str, str] = {}
        start, scored = time.perf_counter(), 0
        for i, name in enumerate(todo, 1):
            spec = system_spec(name, experiments, args.base_model)
            folder = results / name
            if spec["kind"] == "experiment" and not Path(spec["model"]).exists():
                failures[name] = f"checkpoint not found: {spec['model']}"
                print(f"[skip] {name}: {failures[name]}")
                continue
            if not args.force and is_finished(folder, expected):
                print(f"[done] {name} already scored on these {expected[0]} sentences")
            else:
                print(f"\n[{i}/{len(todo)}] scoring {name} ({spec['kind']})", flush=True)
                try:
                    score_system(spec, args, folder, run_split_manifests(run_folder(name, experiments, args.runs_dir)))
                    scored += 1
                except (Exception, SystemExit) as exc:  # one broken system must not stop the batch
                    failures[name] = f"{type(exc).__name__}: {exc}"
                    traceback.print_exc()
                gc.collect()
                elapsed = time.perf_counter() - start
                remaining = sum(1 for n in todo[i:] if args.force or not is_finished(results / n, expected))
                if scored:
                    print(f"[time] {elapsed / 60:.1f} min so far, about {remaining * elapsed / scored / 60:.0f} min "
                          f"left for {remaining} system(s)", flush=True)
            if spec["kind"] != "ground_truth" and not args.skip_prompts and name not in failures \
                    and not (folder / "prompts").is_dir():
                try:
                    synthesise_prompts(spec, args, folder)
                except (Exception, SystemExit) as exc:
                    print(f"[warn] prompts for {name} failed: {exc}")
        timing = system_spec(plan.get("timing_system", plan["baseline"]), experiments, args.base_model)
        timing_folder = results / timing["name"]
        if not args.skip_timing and timing["kind"] != "ground_truth" and timing["name"] not in failures \
                and not any(timing_folder.glob("latency_*.json")):
            try:
                time_system(timing, args, timing_folder)
            except (Exception, SystemExit) as exc:
                print(f"[warn] timing failed: {exc}")
        if failures:
            print("\n[failures] " + "; ".join(f"{k}: {v}" for k, v in failures.items()))

    path = build_report(args, plan, experiments, results, checks)
    print(f"\n[info] report: {path}")
    return path


if __name__ == "__main__":
    main()
