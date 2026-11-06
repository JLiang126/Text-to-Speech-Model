"""Reads what scripts/train_model.py writes, so training losses can sit next to the test metrics.

A run folder is a config's ``output_dir`` (e.g. ``outputs/no_droplayer/``):

  run/training_loss_history.csv     one row per epoch: epoch + LOSS_NAMES (scripts/plots.py)
  run/validation_loss_history.csv   the same losses on the validation split
  run/training_config.json          the settings the run actually used
  splits/train_manifest.csv ...     the split it trained on

Validation losses come from ``evaluate`` in train_model.py: the model runs in training mode
(posterior encoder and monotonic alignment on the real recording), so they measure how well
a run reconstructs held-out speech, not free synthesis from text. Only three are comparable
between runs: ``mel`` (L1 mel distance, unweighted), ``kl`` (prior vs posterior) and ``duration``
(duration predictor). ``gen``, ``disc`` and ``fmaps`` depend on each run's own discriminator,
and ``generator_total`` on the loss weights, so they are left out of comparisons.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]
TRAINING_HISTORY_CSV = "training_loss_history.csv"  # names from scripts/plots.py
VALIDATION_HISTORY_CSV = "validation_loss_history.csv"
COMPARABLE_LOSSES = ("mel", "kl", "duration")
LOSS_LABELS = {"mel": "Validation mel L1", "kl": "Validation KL", "duration": "Validation duration loss"}


def config_output_dir(config: Optional[PathLike]) -> Optional[Path]:
    """``output_dir`` from a training YAML, or None."""
    if not config or not Path(config).is_file():
        return None
    import yaml
    value = (yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}).get("output_dir")
    return Path(value) if value else None


def _history_dir(folder: Path) -> Optional[Path]:
    for sub in (folder / "run", folder):
        if (sub / VALIDATION_HISTORY_CSV).is_file():
            return sub
    return None


def find_run_folder(experiment: str, config: Optional[PathLike] = None,
                    runs_dir: Optional[PathLike] = "outputs") -> Optional[Path]:
    """The folder holding this experiment's loss histories: ``<runs_dir>/<experiment>``, else the
    folder named by the config's output_dir (inside ``runs_dir``, then as written). The experiment's
    own name comes first because a config's output_dir can be reused by a later run with other
    settings (e.g. the embedding-only YAML now points at a 50x-LR run)."""
    output_dir = config_output_dir(config)
    candidates: List[Path] = []
    if runs_dir is not None:
        candidates.append(Path(runs_dir) / experiment)
        if output_dir is not None:
            candidates.append(Path(runs_dir) / output_dir.name)
    if output_dir is not None:
        candidates.append(output_dir)
    for folder in candidates:
        if _history_dir(folder) is not None:
            return folder
    return None


def load_history(folder: PathLike) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
    """(training, validation) loss histories of a run folder."""
    sub = _history_dir(Path(folder))
    if sub is None:
        raise FileNotFoundError(f"No {VALIDATION_HISTORY_CSV} in {folder} or {Path(folder) / 'run'}")
    train_path = sub / TRAINING_HISTORY_CSV
    train = pd.read_csv(train_path) if train_path.is_file() else None
    return train, pd.read_csv(sub / VALIDATION_HISTORY_CSV)


def run_settings(folder: PathLike) -> Dict[str, object]:
    """training_config.json of a run folder ({} if absent)."""
    for sub in (Path(folder) / "run", Path(folder)):
        path = sub / "training_config.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_histories(experiments: Mapping[str, Optional[PathLike]], runs_dir: Optional[PathLike] = "outputs"
                   ) -> Dict[str, Tuple[Optional[pd.DataFrame], pd.DataFrame]]:
    """{experiment: (training, validation)} for every experiment whose logs are found.
    ``experiments`` maps an experiment name to its training config (or None)."""
    histories = {}
    for name, config in experiments.items():
        folder = find_run_folder(name, config, runs_dir)
        if folder is not None:
            histories[name] = load_history(folder)
    return histories


def _slope(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.polyfit(np.arange(len(values)), values, 1)[0]) if len(values) >= 2 else float("nan")


def loss_table(histories: Mapping[str, Tuple[Optional[pd.DataFrame], pd.DataFrame]],
               losses: Sequence[str] = COMPARABLE_LOSSES, tail: int = 3) -> pd.DataFrame:
    """One row per run: final and best validation losses, the epoch of the lowest validation
    mel loss, the slope over the last ``tail`` epochs (negative = still improving) and the
    final train/validation mel gap (a large gap = fitting the training sentences)."""
    rows = []
    for name, (train, val) in histories.items():
        val = val.dropna(subset=[l for l in losses if l in val.columns], how="all").reset_index(drop=True)
        if val.empty:
            continue
        row: Dict[str, object] = {"experiment": name, "epochs": int(val["epoch"].iloc[-1])}
        for loss in losses:
            if loss not in val.columns:
                continue
            series = val[loss].to_numpy(dtype=float)
            row[f"{loss}_final"] = float(series[-1])
            row[f"{loss}_min"] = float(np.nanmin(series))
            row[f"{loss}_best_epoch"] = int(val["epoch"].iloc[int(np.nanargmin(series))])
            row[f"{loss}_slope_last{tail}"] = _slope(series[-tail:])
        if train is not None and "mel" in train.columns and "mel" in val.columns:
            row["mel_train_final"] = float(train["mel"].dropna().iloc[-1])
            row["mel_gap_final"] = row["mel_final"] - row["mel_train_final"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_loss_curves(histories: Mapping[str, Tuple[Optional[pd.DataFrame], pd.DataFrame]], path: PathLike,
                     losses: Sequence[str] = COMPARABLE_LOSSES, title: Optional[str] = None,
                     show_training: bool = True) -> Path:
    """One panel per loss, one colour per run: validation solid, training dashed."""
    import matplotlib.pyplot as plt

    losses = [l for l in losses if any(l in val.columns for _, val in histories.values())]
    fig, axes = plt.subplots(1, len(losses), figsize=(4.2 * len(losses), 3.9), squeeze=False)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, loss in zip(axes[0], losses):
        for i, (name, (train, val)) in enumerate(histories.items()):
            colour = colours[i % len(colours)]
            if loss in val.columns:
                ax.plot(val["epoch"], val[loss], color=colour, lw=1.8, marker="o", ms=3, label=name)
            if show_training and train is not None and loss in train.columns:
                ax.plot(train["epoch"], train[loss], color=colour, lw=1, ls="--", alpha=0.6)
        ax.set_title(LOSS_LABELS.get(loss, loss), fontsize=10, loc="left")
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
    # legend below the panels, so it never covers a curve
    handles, labels = axes[0][0].get_legend_handles_labels()
    ncol = min(len(labels), 5)
    legend_rows = -(-len(labels) // ncol) if labels else 0
    if labels:
        fig.legend(handles, labels, loc="lower center", ncol=ncol, fontsize=8, frameon=False)
    if title:
        fig.suptitle(title + (" (solid = validation, dashed = training)" if show_training else ""),
                     x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.06 * legend_rows + 0.02, 1, 0.97))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def settings_mismatches(settings: Mapping[str, object], expected: Mapping[str, object],
                        keys: Iterable[str]) -> Dict[str, Tuple[object, object]]:
    """Settings that differ between a run's training_config.json and what we think it used."""
    out = {}
    for key in keys:
        if key in settings and key in expected and str(expected[key]).strip() != "":
            a, b = settings[key], expected[key]
            try:
                same = np.isclose(float(a), float(b))
            except (TypeError, ValueError):
                same = str(a).strip().lower() == str(b).strip().lower()
            if not same:
                out[key] = (a, b)
    return out
