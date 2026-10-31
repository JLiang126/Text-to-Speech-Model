import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MATPLOTLIB_FLAG = False
LOSS_NAMES = ("mel", "kl", "duration", "fmaps", "gen", "disc", "generator_total")
HISTORY_COLUMNS = ("epoch", *LOSS_NAMES)
TRAINING_HISTORY_CSV = "training_loss_history.csv"
VALIDATION_HISTORY_CSV = "validation_loss_history.csv"


def plot_spectrogram_to_numpy(spectrogram):
    """
    Plots the spectrogram as a heatmap and returns it as a numpy array.
    """
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    """
    Plots the alignment matrix as a heatmap and returns it as a numpy array.
    """
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.transpose(), aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data



def _normalize_history_frame(history):
    if not isinstance(history, pd.DataFrame):
        raise ValueError("Loss history must be a pandas DataFrame.")
    frame = history.copy()

    if frame.empty:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    if "epoch" not in frame.columns:
        raise ValueError("Loss history is missing the 'epoch' column.")

    return frame.reindex(columns=HISTORY_COLUMNS).reset_index(drop=True)


def save_loss_history_csvs(training_history, validation_history, directory):
    """Save separate training and validation loss DataFrames as CSV files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _normalize_history_frame(training_history).to_csv(directory / TRAINING_HISTORY_CSV, index=False)
    _normalize_history_frame(validation_history).to_csv(directory / VALIDATION_HISTORY_CSV, index=False)


def plot_training_history(training_history, validation_history, output_path):
    """Plot loss columns from separate training and validation DataFrames."""
    training = _normalize_history_frame(training_history)
    validation = _normalize_history_frame(validation_history)
    if training.empty:
        raise ValueError("Cannot plot an empty training history.")

    metrics = [
        name for name in LOSS_NAMES
        if name in training.columns and training[name].notna().any()
    ]
    if not metrics:
        raise ValueError("Training history does not contain any loss values.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = math.ceil(len(metrics) / 2)
    fig, axes = plt.subplots(row_count, 2, figsize=(12, 4 * row_count), squeeze=False)
    for axis, metric in zip(axes.flat, metrics):
        axis.plot(training["epoch"], training[metric], label="Training")
        if not validation.empty and validation[metric].notna().any():
            axis.plot(validation["epoch"], validation[metric], label="Validation")
        axis.set(title=metric, xlabel="Epoch", ylabel="Loss")
        axis.grid(True, alpha=0.3)
        axis.legend()
    for axis in list(axes.flat)[len(metrics):]:
        axis.set_visible(False)
    fig.tight_layout()
    try:
        fig.savefig(output_path, dpi=150)
    finally:
        plt.close(fig)
