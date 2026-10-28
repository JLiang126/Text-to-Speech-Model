"""Loads the split manifests written by ``scripts/DataLoader.build_splits``.

Manifest format (CSV with header): ``wav_path, utt_id, raw_text, norm_text, clean_text``.

``build_splits`` stores ``wav_path`` relative to the working directory it was run
from, so the same manifest can break when a script runs from somewhere else.
``resolve_wav_path`` therefore tries a few sensible locations before giving up.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

PathLike = Union[str, Path]

REQUIRED_COLUMNS = ("utt_id", "wav_path")
# Reference text for WER: LJSpeech's normalised transcription (numbers already spelled out).
TEXT_COLUMNS = ("norm_text", "raw_text", "clean_text")


def resolve_wav_path(wav_path: str, utt_id: str, manifest_path: PathLike,
                     data_root: Optional[PathLike] = None) -> Path:
    """Finds the wav for one manifest row, or raises with every location that was tried."""
    raw = Path(str(wav_path))
    candidates: List[Path] = [raw]
    if not raw.is_absolute():
        manifest_dir = Path(manifest_path).resolve().parent
        # dataset/splits/x.csv -> try dataset/splits, dataset, repo root, one level above
        candidates += [base / raw for base in [manifest_dir, *list(manifest_dir.parents)[:3]]]
    if data_root is not None:
        candidates.append(Path(data_root) / "wavs" / f"{utt_id}.wav")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not find audio for {utt_id}. Tried:\n  {tried}\n"
                            f"Pass --data_root pointing at the LJSpeech-1.1 folder.")


def load_manifest(manifest_path: PathLike, data_root: Optional[PathLike] = None,
                  limit: Optional[int] = None) -> pd.DataFrame:
    """Reads a manifest, resolves audio paths and adds a ``ref_text`` column for WER."""
    manifest_path = Path(manifest_path)
    df = pd.read_csv(manifest_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{manifest_path} is missing columns {missing}; found {list(df.columns)}")
    text_col = next((c for c in TEXT_COLUMNS if c in df.columns), None)
    if text_col is None:
        raise ValueError(f"{manifest_path} has no text column (expected one of {TEXT_COLUMNS})")
    if limit is not None:
        df = df.iloc[: int(limit)]
    df = df.reset_index(drop=True).copy()
    df["utt_id"] = df["utt_id"].astype(str)
    df["wav_path"] = [str(resolve_wav_path(p, u, manifest_path, data_root))
                      for p, u in zip(df["wav_path"], df["utt_id"])]
    df["ref_text"] = df[text_col].astype(str)
    return df
