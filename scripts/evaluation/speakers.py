"""Ranks the speakers of a multi-speaker checkpoint by how close they sound to LJSpeech.

Use: when fine-tuning the VCTK-pretrained VITS on LJSpeech with the multi-speaker
config, every LJSpeech clip needs one speaker ID. Starting from the VCTK voice that
already sounds most like LJSpeech is a defensible, documented choice (the alternative,
an arbitrary ID, starts adaptation from further away). The ranking also gives the
pretrained model's best-case ("floor") speaker similarity before any fine-tuning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from scripts.evaluation.audio import load_audio, prepare, resample
from scripts.evaluation.manifest import load_manifest
from scripts.evaluation.metrics import ASR_SR, cosine_similarity, speaker_centroid


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


def rank_speakers(synthesize: Callable[[str, int], np.ndarray], embed: Callable[[np.ndarray], np.ndarray],
                  target_centroid: np.ndarray, texts: Sequence[str],
                  speaker_ids: Iterable[int]) -> pd.DataFrame:
    """For each speaker: synthesise ``texts``, embed, and compare with the target voice.

    ``synthesize(text, speaker_id)`` must return audio ready for ``embed`` (16 kHz).
    Returns one row per speaker, best first:
      similarity       cosine(mean embedding of that speaker's sentences, target centroid)
      mean_similarity  average per-sentence cosine (shows how consistent the speaker is)
    """
    rows = []
    for sid in speaker_ids:
        embeddings = [embed(synthesize(text, sid)) for text in texts]
        rows.append({"speaker_id": int(sid),
                     "similarity": cosine_similarity(speaker_centroid(embeddings), target_centroid),
                     "mean_similarity": float(np.mean([cosine_similarity(e, target_centroid)
                                                       for e in embeddings]))})
    ranking = pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking
