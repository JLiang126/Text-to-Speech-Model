"""Speaker similarity to LJSpeech: the reference voice, and the VCTK speaker ranking.

Why a ranking: the VCTK-pretrained VITS has one learned embedding per VCTK speaker, and
fine-tuning on LJSpeech continues from whichever speaker index the LJSpeech clips are
given. The speaker-similarity experiment fine-tunes from the closest and the furthest
VCTK voice (everything else identical), so it needs a ranking that is defensible and
comes with its uncertainty. ``scripts/rank_speakers.py`` runs it; the maths is here.

Method:
1. N LJSpeech *training* sentences (the test split stays unseen).
2. Every speaker index s says every sentence i, with the same per-sentence seed.
3. Real and synthetic audio get the same preprocessing and ECAPA-TDNN embedding.
4. Paired similarity: cos(e[s, i], e[LJ, i]), the same words spoken by both, so the
   comparison is about the voice, not the content. A speaker's score is the mean over i.
5. Uncertainty: a bootstrap over sentences (all speakers resampled together) gives a CI
   for each score and for each rank, plus how often each speaker comes out closest or
   furthest. Split-half reliability (odd vs even sentences) shows whether N is enough.

``vctk_speakers.csv`` maps speaker index -> VCTK speaker ID and the number of training
clips, read from the official VITS training filelist
(jaywalnut310/vits filelists/vctk_audio_sid_text_train_filelist.txt). It lists indices
0-107 only: the official config allocates 109 embeddings (n_speakers=109), but index 108
never occurs in training, so that embedding was never trained and is excluded by default.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from scripts.evaluation.audio import load_audio, prepare, resample
from scripts.evaluation.manifest import load_manifest
from scripts.evaluation.metrics import ASR_SR, speaker_centroid

VCTK_SPEAKERS_CSV = Path(__file__).with_name("vctk_speakers.csv")
VCTK_N_SPEAKERS = 109  # embedding table size of the official VCTK checkpoint (and its HF ports)


def reference_centroid(embed: Callable[[np.ndarray], np.ndarray], manifest_path: Union[str, Path],
                       data_root: Optional[Union[str, Path]] = None, n_ref: int = 50,
                       seed: int = 1234) -> np.ndarray:
    """'The LJSpeech voice': mean embedding of ``n_ref`` real clips sampled (seeded) from a manifest.

    Use the *train* manifest so no test sentence is part of the reference.
    """
    ref = load_manifest(manifest_path, data_root)
    ref = ref.sample(n=min(n_ref, len(ref)), random_state=seed)
    embeddings = []
    for wav_path in ref["wav_path"]:
        audio, sr = load_audio(wav_path)
        embeddings.append(embed(resample(prepare(audio, sr), sr, ASR_SR)))
    return speaker_centroid(embeddings)


def load_vctk_speakers(path: Union[str, Path] = VCTK_SPEAKERS_CSV) -> pd.DataFrame:
    """speaker_index -> vctk_id, n_train_utts (indices that were trained in the official VCTK model)."""
    return pd.read_csv(path).set_index("speaker_index")


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def paired_similarity(synth_emb: np.ndarray, real_emb: np.ndarray) -> np.ndarray:
    """(S, N, D) synthetic and (N, D) real embeddings -> (S, N) cosine per speaker and sentence."""
    return np.einsum("snd,nd->sn", _unit(synth_emb), _unit(real_emb))


def centroid_similarity(synth_emb: np.ndarray, real_emb: np.ndarray) -> np.ndarray:
    """Cosine between each speaker's mean voice and the mean LJSpeech voice -> (S,)."""
    synth_centroids = _unit(_unit(synth_emb).mean(axis=1))
    real_centroid = _unit(_unit(real_emb).mean(axis=0))
    return synth_centroids @ real_centroid


def _ranks(scores: np.ndarray) -> np.ndarray:
    """Rank 1 = highest score, along axis 0 (stable for ties)."""
    return np.argsort(np.argsort(-scores, axis=0, kind="stable"), axis=0, kind="stable") + 1


def rank_speakers(paired: np.ndarray, speaker_ids: Sequence[int], eligible: Optional[Sequence[bool]] = None,
                  n_boot: int = 1000, ci: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """Ranks speakers by mean paired similarity, with bootstrap CIs for scores and ranks.

    ``paired``: (S, N) from paired_similarity. Only ``eligible`` speakers get a rank
    (default: all); the others are still scored and listed last.
    Columns: rank, speaker_index, eligible, similarity, ci_low, ci_high, rank_ci_low,
    rank_ci_high, p_closest, p_furthest (share of bootstrap resamples where the speaker
    is the closest / furthest eligible voice), n_utts.
    """
    paired = np.asarray(paired, dtype=np.float64)
    n_speakers, n_utts = paired.shape
    eligible = np.ones(n_speakers, bool) if eligible is None else np.asarray(eligible, bool)
    keep = np.flatnonzero(eligible)
    mean = paired.mean(axis=1)

    rng = np.random.default_rng(seed)
    boot = np.empty((n_speakers, n_boot))
    for b in range(n_boot):
        boot[:, b] = paired[:, rng.integers(0, n_utts, n_utts)].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    ci_low, ci_high = np.quantile(boot, [alpha, 1.0 - alpha], axis=1)

    rank = np.full(n_speakers, np.nan)
    rank_lo, rank_hi = np.full(n_speakers, np.nan), np.full(n_speakers, np.nan)
    p_closest, p_furthest = np.full(n_speakers, np.nan), np.full(n_speakers, np.nan)
    if keep.size:
        rank[keep] = _ranks(mean[keep])
        boot_ranks = _ranks(boot[keep])
        rank_lo[keep], rank_hi[keep] = np.quantile(boot_ranks, [alpha, 1.0 - alpha], axis=1)
        p_closest[keep] = (boot_ranks == 1).mean(axis=1)
        p_furthest[keep] = (boot_ranks == keep.size).mean(axis=1)

    table = pd.DataFrame({"rank": rank, "speaker_index": np.asarray(speaker_ids, dtype=int), "eligible": eligible,
                          "similarity": mean, "ci_low": ci_low, "ci_high": ci_high,
                          "rank_ci_low": rank_lo, "rank_ci_high": rank_hi,
                          "p_closest": p_closest, "p_furthest": p_furthest, "n_utts": n_utts})
    table = table.sort_values(["eligible", "rank", "similarity"], ascending=[False, True, False],
                              na_position="last").reset_index(drop=True)
    table["rank"] = table["rank"].astype("Int64")
    return table


def _mean_pairwise_cosine(unit_vectors: np.ndarray) -> np.ndarray:
    """Mean cosine over all pairs i < j along axis -2, using sum_{i!=j} u_i.u_j = |sum u|^2 - n."""
    n = unit_vectors.shape[-2]
    if n < 2:
        return np.full(unit_vectors.shape[:-2], np.nan)
    total = unit_vectors.sum(axis=-2)
    return (np.sum(total * total, axis=-1) - n) / (n * (n - 1))


def similarity_baselines(synth_emb: np.ndarray, real_emb: np.ndarray,
                         eligible: Optional[Sequence[bool]] = None) -> Dict[str, float]:
    """Reference points that give the speaker scores a scale.

    real_vs_real                  LJSpeech vs LJSpeech, different sentences: the same-speaker ceiling
    same_synthetic_speaker        one VCTK voice vs itself, different sentences: how consistent a voice is
    different_synthetic_speakers  two VCTK voices, same sentence: the level of "different speakers"
    """
    synth = _unit(synth_emb)
    if eligible is not None:
        synth = synth[np.asarray(eligible, bool)]
    real = _unit(real_emb)
    return {"real_vs_real": float(_mean_pairwise_cosine(real)),
            "same_synthetic_speaker": float(np.mean(_mean_pairwise_cosine(synth))),
            "different_synthetic_speakers": float(np.mean(_mean_pairwise_cosine(np.swapaxes(synth, 0, 1))))}


def pitch_groups(table: pd.DataFrame, threshold_hz: float = 160.0) -> Dict[str, object]:
    """Splits the ranked speakers at ``threshold_hz`` median pitch (adult male voices mostly sit
    below it, female voices above) and gives the closest / furthest speaker within each group.
    The groups are inferred from the synthetic audio, not from VCTK metadata."""
    ranked = table[table["eligible"] & table["f0_median_hz"].notna()]
    groups: Dict[str, object] = {"threshold_hz": float(threshold_hz)}
    for name, members in (("low", ranked[ranked["f0_median_hz"] < threshold_hz]),
                          ("high", ranked[ranked["f0_median_hz"] >= threshold_hz])):
        members = members.sort_values("similarity", ascending=False)
        if members.empty:
            groups[name] = {"n": 0}
            continue
        groups[name] = {"n": int(len(members)), "mean_similarity": float(members["similarity"].mean()),
                        "closest": int(members.iloc[0]["speaker_index"]),
                        "closest_similarity": float(members.iloc[0]["similarity"]),
                        "furthest": int(members.iloc[-1]["speaker_index"]),
                        "furthest_similarity": float(members.iloc[-1]["similarity"])}
    return groups


def split_half_reliability(paired: np.ndarray, eligible: Optional[Sequence[bool]] = None) -> Dict[str, float]:
    """Spearman correlation of speaker scores from odd vs even sentences (+ Spearman-Brown for all N).

    Near 1: the ranking would barely change with different sentences, so N is enough.
    """
    from scipy.stats import spearmanr

    paired = np.asarray(paired, dtype=np.float64)
    if eligible is not None:
        paired = paired[np.asarray(eligible, bool)]
    if paired.shape[1] < 2 or paired.shape[0] < 3:
        return {"split_half_spearman": float("nan"), "spearman_brown": float("nan")}
    r = float(spearmanr(paired[:, 0::2].mean(axis=1), paired[:, 1::2].mean(axis=1)).statistic)
    return {"split_half_spearman": r, "spearman_brown": 2 * r / (1 + r) if r > -1 else float("nan")}


def pick_conditions(table: pd.DataFrame) -> Dict[str, int]:
    """Closest, median and furthest eligible speaker index from a rank_speakers table."""
    ranked = table[table["eligible"]].sort_values("rank")
    if ranked.empty:
        return {}
    indices = ranked["speaker_index"].tolist()
    return {"closest": int(indices[0]), "median": int(indices[(len(indices) - 1) // 2]),
            "furthest": int(indices[-1])}


def f0_median(audio: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 500.0) -> float:
    """Median pitch (Hz) over voiced frames: YIN estimates, kept where the frame is loud
    (within 20 dB of the loudest) and the estimate is not pinned to the search limits.

    A coarse voice descriptor (typical adult male about 85-180 Hz, female about 165-255 Hz),
    used to describe the ranked voices, not as a similarity score.
    """
    import librosa

    audio = np.asarray(audio, dtype=np.float32)
    frame = 2048 if sr > 24000 else 1024
    hop = frame // 4
    if audio.size < frame:
        return float("nan")
    f0 = librosa.yin(audio, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame, hop_length=hop)
    rms = librosa.feature.rms(y=audio, frame_length=frame, hop_length=hop)[0]
    n = min(f0.size, rms.size)
    f0, rms = f0[:n], rms[:n]
    voiced = (rms > 0.1 * rms.max()) & (f0 > fmin * 1.05) & (f0 < fmax * 0.95)
    return float(np.median(f0[voiced])) if voiced.any() else float("nan")
