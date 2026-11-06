"""Audio I/O and the preprocessing shared by every evaluation metric.

Every metric sees audio that went through the same steps (DC removal, peak
normalisation, silence trimming), so differences between experiments cannot
come from loudness or from leading/trailing silence.

The trimming rule deliberately mirrors ``scripts/AudioProcessor.py`` (RMS energy,
20 ms window, 5 ms hop, threshold relative to the loudest frame), so reference
recordings are treated the same way as the training targets. It is reimplemented
in NumPy so evaluation does not depend on torchaudio.
"""
from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

PathLike = Union[str, Path]

EVAL_PEAK = 0.95      # same peak level as the training preprocessing
EVAL_TRIM_DB = 40.0   # same trim threshold as DataConfig.trim_db


def load_audio(path: PathLike, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Loads a wav file as mono float32 in [-1, 1], optionally resampled to ``target_sr``."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # stereo -> mono (LJSpeech is already mono)
    if target_sr is not None and sr != target_sr:
        audio = resample(audio, sr, target_sr)
        sr = target_sr
    return audio.astype(np.float32), int(sr)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resampling with an exact rational ratio (22050 -> 16000 is 320/441)."""
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    g = gcd(int(orig_sr), int(target_sr))
    return resample_poly(audio, int(target_sr) // g, int(orig_sr) // g).astype(np.float32)


def save_wav(path: PathLike, audio: np.ndarray, sr: int) -> Path:
    """Writes 16-bit PCM wav (clipped to [-1, 1]) and creates parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0), int(sr), subtype="PCM_16")
    return path


def peak_normalise(audio: np.ndarray, peak: float = EVAL_PEAK) -> np.ndarray:
    """Scales so the loudest sample sits at ``peak``. Silent input is returned unchanged."""
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs < 1e-8:
        return audio.astype(np.float32)
    return (audio / max_abs * peak).astype(np.float32)


def trim_silence(audio: np.ndarray, sr: int, top_db: float = EVAL_TRIM_DB,
                 win_ms: float = 20.0, hop_ms: float = 5.0) -> np.ndarray:
    """Removes leading/trailing frames quieter than ``top_db`` below the loudest frame."""
    win = max(1, int(sr * win_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if audio.size < win:
        return audio
    frames = np.lib.stride_tricks.sliding_window_view(audio, win)[::hop]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-10)
    threshold = rms.max() * 10.0 ** (-top_db / 20.0)
    active = np.nonzero(rms > threshold)[0]
    if active.size == 0:
        return audio
    start = int(active[0]) * hop
    end = min(audio.size, int(active[-1]) * hop + win)
    return audio[start:end]


def prepare(audio: np.ndarray, sr: int, trim: bool = True,
            top_db: float = EVAL_TRIM_DB, peak: float = EVAL_PEAK) -> np.ndarray:
    """Standard evaluation preprocessing: DC removal -> peak normalisation -> optional trim."""
    audio = np.asarray(audio, dtype=np.float32)
    audio = audio - float(np.mean(audio)) if audio.size else audio
    audio = peak_normalise(audio, peak)
    if trim:
        audio = trim_silence(audio, sr, top_db)
    return audio
