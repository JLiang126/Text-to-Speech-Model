"""
Functions for plotting spectrograms, alignments, and loss history.

Also manages the saving of loss history as CSV files.

"""

import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Constants and configuration for plotting and loss history management
MATPLOTLIB_FLAG = False 
LOSS_NAMES = ("mel", "kl", "duration", "fmaps", "gen", "disc", "generator_total")
HISTORY_COLUMNS = ("epoch", *LOSS_NAMES)  # Expected columns for the loss history DataFrame
TRAINING_HISTORY_CSV = "training_loss_history.csv" # File name for saving training loss history
VALIDATION_HISTORY_CSV = "validation_loss_history.csv" # File name for saving validation loss history


def plot_spectrogram_to_numpy(spectrogram):
    """
    Plots the spectrogram as a heatmap and returns it as a numpy array.

    Note: this is from ylacombe/finetune-hf-vits/utils/plot.py.
    Can potentially be removed if not needed.
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

    Note: this is from ylacombe/finetune-hf-vits/utils/plot.py
    Can potentially be removed if not needed.
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



def normalize_history_frame(history):
    """
    Normalize the loss history DataFrame to ensure it contains the expected columns.

    Args:
        history (pd.DataFrame): The loss history DataFrame.

    Returns:
        pd.DataFrame: The normalized loss history DataFrame with the expected columns.
    Raises:
        ValueError: If the input is not a pandas DataFrame or is missing the 'epoch' column.
    """
    # check that it's a pandas DataFrame
    if not isinstance(history, pd.DataFrame):
        raise ValueError("Loss history must be a pandas DataFrame.")
    frame = history.copy()

    # if the DataFrame is empty, return an empty DataFrame with the expected columns
    if frame.empty:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    if "epoch" not in frame.columns:
        raise ValueError("Loss history is missing the 'epoch' column.")

    return frame.reindex(columns=HISTORY_COLUMNS).reset_index(drop=True)


def save_loss_history_csvs(training_history, validation_history, directory):
    """
    Save separate training and validation loss DataFrames as CSV files.

    Args:
        training_history (pd.DataFrame): The training loss history.
        validation_history (pd.DataFrame): The validation loss history.
        directory (str or Path): The directory to save the CSV files.
    """
    # Ensure the directory exists before saving CSV files
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # save training and validation loss history as CSV files
    normalize_history_frame(training_history).to_csv(directory / TRAINING_HISTORY_CSV, index=False)
    normalize_history_frame(validation_history).to_csv(directory / VALIDATION_HISTORY_CSV, index=False)


def plot_training_history(training_history, validation_history, output_path):
    """
    Plot loss columns from separate training and validation DataFrames.

    Args:
        training_history (pd.DataFrame): The training loss history.
        validation_history (pd.DataFrame): The validation loss history.
        output_path (str or Path): The path to save the plot image.
    Raises:
        ValueError: If the training history is empty or contains no loss values.
    """

    # ensure the training and validation history frames are normalized and contain expected columns
    training = normalize_history_frame(training_history)
    validation = normalize_history_frame(validation_history)
    if training.empty:
        raise ValueError("Cannot plot an empty training history.")

    # get the metrics that are present in the training history and have non-NaN values
    metrics = [name for name in LOSS_NAMES if name in training.columns and training[name].notna().any()]
    if not metrics:
        raise ValueError("Training history does not contain any loss values.")

    # ensure the output directory exists before saving the plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # get the number of rows needed for the subplot grid based on the number of metrics
    row_count = math.ceil(len(metrics) / 2)
    # create the subplot grid with the calculated number of rows and 2 columns
    fig, axes = plt.subplots(row_count, 2, figsize=(12, 4 * row_count), squeeze=False)
    # loop through each metric and plot it on the corresponding axis
    for axis, metric in zip(axes.flat, metrics):
        # plot training loss for the current metric
        axis.plot(training["epoch"], training[metric], label="Training")
        # plot validation loss for the current metric if available
        if not validation.empty and validation[metric].notna().any():
            axis.plot(validation["epoch"], validation[metric], label="Validation")
        # add title, labels, grid, and legend for the current subplot
        axis.set(title=metric, xlabel="Epoch", ylabel="Loss")
        axis.grid(True, alpha=0.3)
        axis.legend()
    # hide any unused subplots
    for axis in list(axes.flat)[len(metrics):]:
        axis.set_visible(False)
    fig.tight_layout() # tidy up the layout
    try:
        fig.savefig(output_path, dpi=150)
    finally:
        plt.close(fig)
