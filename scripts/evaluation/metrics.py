"""Objective metrics for the TTS experiments, computed per utterance.

| Metric         | What it measures                          | Better | Reference needed        | Model used |
|----------------|-------------------------------------------|--------|-------------------------|------------|
| ``mcd_dtw``    | spectral distance to the real recording   | lower  | the same sentence (test) | none |
| ``wer``/``cer``| intelligibility (ASR transcript vs text)  | lower  | the text only           | Whisper |
| ``secs``       | speaker similarity to the LJSpeech voice  | higher | real LJSpeech clips     | ECAPA-TDNN |
| ``utmos``      | predicted naturalness on a 1-5 MOS scale  | higher | none                    | UTMOS22 strong |

The three model-based metrics are loaded lazily. If one cannot be loaded (no
internet, missing package), the caller records NaN for it and carries on, so one
failed download never kills an evaluation run.

Sources
-------
- MCD: Kubichek (1993) 'Mel-cepstral distance measure for objective speech quality
  assessment'. Here computed on mel-frequency cepstra (DCT of the log-mel
  spectrogram, c0 dropped) and aligned with dynamic time warping.
- Whisper: Radford et al. (2023) 'Robust speech recognition via large-scale weak
  supervision', https://github.com/openai/whisper (MIT).
- ECAPA-TDNN: Desplanques, Thienpondt & Demuynck (2020), via SpeechBrain
  https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb (Apache-2.0).
  Speaker-encoder cosine similarity (SECS) as used in YourTTS (Casanova et al. 2022).
- UTMOS: Saeki et al. (2022) 'UTMOS: UTokyo-SaruLab system for VoiceMOS Challenge
  2022', packaged by https://github.com/tarepan/SpeechMOS (MIT).
"""
from __future__ import annotations

import re
from typing import Callable, Dict, Iterable, Optional

import numpy as np

ASR_SR = 16_000      # Whisper, ECAPA and UTMOS all expect 16 kHz input
MCD_K = 10.0 / np.log(10.0)  # converts natural-log cepstral distance to decibels


# --------------------------------------------------------------------------- MCD
def mel_cepstrum(audio: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 256,
                 win_length: int = 1024, n_mels: int = 80, fmin: float = 0.0,
                 fmax: float = 8000.0, n_coeffs: int = 24) -> np.ndarray:
    """Mel-frequency cepstrum (n_coeffs, frames) with c0 removed.

    STFT/mel settings match ``scripts/DataConfig.py`` (1024/256/1024, 80 Slaney mels,
    0-8 kHz). Dropping c0 removes overall loudness, so the distance compares
    spectral shape only.

    The orthonormal DCT is divided by sqrt(2 * n_mels), which puts the coefficients
    on the usual cepstrum scale (log|S(w)| = c0 + 2 * sum_k c_k cos(k w)). That is
    the convention the standard MCD constant assumes; without it the distances come
    out about sqrt(2 * n_mels) (~12.6x) too large.
    """
    import librosa
    from scipy.fft import dct

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                         win_length=win_length, n_mels=n_mels, fmin=fmin,
                                         fmax=fmax, power=1.0, center=True)
    log_mel = np.log(np.maximum(mel, 1e-5))
    cepstrum = dct(log_mel, type=2, axis=0, norm="ortho") / np.sqrt(2.0 * n_mels)
    return cepstrum[1: n_coeffs + 1]


def mcd_dtw(reference: np.ndarray, synthesized: np.ndarray, sr: int, **cepstrum_kwargs) -> Dict[str, float]:
    """Mel-cepstral distortion (dB) after DTW alignment, plus the duration ratio.

    MCD = (10 / ln 10) * sqrt(2 * sum_d (c_d - c'_d)^2), averaged over aligned frames.
    ``dur_ratio`` = synthesized frames / reference frames (1.0 = same speaking rate).
    Absolute values are not comparable with SPTK-based MCD numbers from papers;
    they are comparable between our experiments because every system is scored
    the same way.
    """
    import librosa

    c_ref = mel_cepstrum(reference, sr, **cepstrum_kwargs)
    c_syn = mel_cepstrum(synthesized, sr, **cepstrum_kwargs)
    _, path = librosa.sequence.dtw(X=c_ref, Y=c_syn, metric="euclidean")
    diff = c_ref[:, path[:, 0]] - c_syn[:, path[:, 1]]
    frame_mcd = MCD_K * np.sqrt(2.0 * np.sum(diff ** 2, axis=0))
    return {"mcd_dtw": float(frame_mcd.mean()), "dur_ratio": float(c_syn.shape[1] / c_ref.shape[1])}


# ----------------------------------------------------------------- text / WER
def _basic_normalise(text: str) -> str:
    """Fallback normaliser: lowercase, keep letters/digits/apostrophes, collapse spaces."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_text_normaliser() -> Callable[[str], str]:
    """Whisper's EnglishTextNormalizer if available (handles numbers, spelling variants), else a basic one."""
    try:
        from whisper.normalizers import EnglishTextNormalizer
        return EnglishTextNormalizer()
    except Exception:  # whisper not installed: still usable, just less thorough
        return _basic_normalise


def word_char_errors(reference: str, hypothesis: str,
                     normaliser: Optional[Callable[[str], str]] = None) -> Dict[str, float]:
    """WER/CER for one utterance plus raw counts so corpus-level rates can be summed later."""
    import jiwer

    normaliser = normaliser or _basic_normalise
    ref = normaliser(reference or "").strip()
    hyp = normaliser(hypothesis or "").strip()
    nan = float("nan")
    if not ref:
        return {"wer": nan, "cer": nan, "word_errors": nan, "ref_words": nan,
                "char_errors": nan, "ref_chars": nan}
    n_words, n_chars = len(ref.split()), len(ref)
    if not hyp:  # nothing recognised: every reference token is a deletion
        return {"wer": 1.0, "cer": 1.0, "word_errors": n_words, "ref_words": n_words,
                "char_errors": n_chars, "ref_chars": n_chars}
    w = jiwer.process_words(ref, hyp)
    c = jiwer.process_characters(ref, hyp)
    word_errors = w.substitutions + w.deletions + w.insertions
    char_errors = c.substitutions + c.deletions + c.insertions
    return {"wer": float(w.wer), "cer": float(c.cer),
            "word_errors": int(word_errors), "ref_words": int(w.hits + w.substitutions + w.deletions),
            "char_errors": int(char_errors), "ref_chars": int(c.hits + c.substitutions + c.deletions)}


class WhisperASR:
    """Transcribes 16 kHz audio with greedy (temperature 0) Whisper decoding."""

    def __init__(self, model_name: str = "small.en", device: str = "cpu"):
        import whisper

        self.model_name = model_name
        self.device = device
        self.model = whisper.load_model(model_name, device=device)

    def transcribe(self, audio_16k: np.ndarray) -> str:
        result = self.model.transcribe(
            np.asarray(audio_16k, dtype=np.float32), language="en", temperature=0.0,
            condition_on_previous_text=False, without_timestamps=True,
            fp16=str(self.device).startswith("cuda"), verbose=None)
        return str(result["text"]).strip()


# ------------------------------------------------------------ speaker similarity
def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    return vector / (np.linalg.norm(vector) + 1e-12)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(a), _unit(b)))


def speaker_centroid(embeddings: Iterable[np.ndarray]) -> np.ndarray:
    """Mean of unit-normalised embeddings, renormalised: the 'average LJSpeech voice'."""
    stacked = np.stack([_unit(e) for e in embeddings])
    return _unit(stacked.mean(axis=0))


class SpeakerEncoder:
    """ECAPA-TDNN speaker embeddings (192-d) from SpeechBrain, trained on VoxCeleb."""

    SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

    def __init__(self, device: str = "cpu", savedir: str = "pretrained_models/spkrec-ecapa-voxceleb"):
        try:
            from speechbrain.inference.speaker import EncoderClassifier  # speechbrain >= 1.0
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier  # older speechbrain
        kwargs = {"source": self.SOURCE, "savedir": savedir, "run_opts": {"device": device}}
        try:  # copy files instead of symlinking: symlinks need admin rights on Windows
            from speechbrain.utils.fetching import LocalStrategy
            self.model = EncoderClassifier.from_hparams(**kwargs, local_strategy=LocalStrategy.COPY)
        except (ImportError, TypeError):
            self.model = EncoderClassifier.from_hparams(**kwargs)
        self.device = device

    def embed(self, audio_16k: np.ndarray) -> np.ndarray:
        import torch

        with torch.inference_mode():
            wav = torch.from_numpy(np.asarray(audio_16k, dtype=np.float32)).unsqueeze(0).to(self.device)
            embedding = self.model.encode_batch(wav).squeeze().detach().cpu().numpy()
        return _unit(embedding)


# ------------------------------------------------------------------ naturalness
class UTMOSPredictor:
    """UTMOS22 strong learner: predicted mean opinion score (about 1-5) from 16 kHz audio."""

    HUB_REPO = "tarepan/SpeechMOS:v1.2.0"

    def __init__(self, device: str = "cpu"):
        import torch

        self.device = device
        self.model = torch.hub.load(self.HUB_REPO, "utmos22_strong", trust_repo=True).to(device).eval()

    def score(self, audio_16k: np.ndarray) -> float:
        import torch

        with torch.inference_mode():
            wav = torch.from_numpy(np.asarray(audio_16k, dtype=np.float32)).unsqueeze(0).to(self.device)
            return float(self.model(wav, ASR_SR).reshape(-1)[0].item())
