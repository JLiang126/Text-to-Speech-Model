"""Checkpoint-tokenized LJSpeech loading with model-ready VITS batches."""

import os
import wave
import random
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import PreTrainedTokenizerBase

from scripts.DataConfig import DataConfig
from scripts.TextProcessor import clean_normalised_text, configure_espeak
from scripts.AudioProcessor import AudioProcessor


def _configure_espeak_worker(_: int) -> None: configure_espeak()


def set_global_seed(seed: int = 42) -> None:
    """Fixes random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Seed every CUDA device, not just the currently selected GPU.
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# Generated manifests are standard CSV files; only source metadata is pipe-delimited.
def load_manifest_df(manifest_path: Union[str, Path]) -> pd.DataFrame: return pd.read_csv(manifest_path)

def build_splits(cfg: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    Executes splitting pipeline:
    1. Lexicographical sort by utt_id for OS independent ordering.
    2. Filters clips using the configured duration limits.
    3. Shuffles using the configured seed (42 by default).
    4. Selects an optional subset from the shuffled records.
    5. Splits data.
    6. Exports train_manifest.csv, val_manifest.csv, and test_manifest.csv.

    Each manifest contains paths, utterance IDs, and raw, normalized, and simply cleaned text.
    """
    data_root = Path(cfg.data_root)
    out_dir = Path(cfg.output_split_dir)
    # Resolve paths so relative paths and symlinks cannot hide writes into the dataset.
    # relative_to succeeds only when the output is the dataset itself or a descendant.
    try: out_dir.resolve().relative_to(data_root.resolve())
    except ValueError: pass
    else: raise ValueError(f"Split output directory must be outside the dataset directory: {out_dir}")

    metadata_path = data_root / "metadata.csv"
    wavs_dir = data_root / "wavs"
    if not metadata_path.exists(): raise FileNotFoundError(f"LJSpeech metadata.csv not found at {metadata_path.resolve()}")

    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            # LJSpeech stores utterance ID, raw transcript, and normalized transcript.
            parts = line.strip().split("|")
            if len(parts) >= 3:
                utt_id = parts[0].strip()
                raw_text = parts[1].strip()
                norm_text = parts[2].strip()
                wav_path = wavs_dir / f"{utt_id}.wav"

                if wav_path.exists():
                    # WAV headers provide duration without decoding the full recording.
                    with wave.open(str(wav_path), "rb") as wf:
                        sample_rate = wf.getframerate()
                        num_channels = wf.getnchannels()
                        num_frames = wf.getnframes()

                    duration_sec = num_frames / float(sample_rate)
                    # Manifest cleaning preserves Unicode; checkpoint phonemization happens later.
                    clean_text = clean_normalised_text(norm_text, lowercase=cfg.lowercase)
                    # Saved relative paths are interpreted from the working directory at load time.
                    rel_wav_path = os.path.relpath(wav_path, start=Path.cwd())
                    records.append({
                        "utt_id": utt_id,
                        "wav_path": str(rel_wav_path),
                        "raw_text": raw_text,
                        "norm_text": norm_text,
                        "clean_text": clean_text,
                        "char_len": len(clean_text),
                        "word_len": len(clean_text.split()),
                        "sample_rate": sample_rate,
                        "num_channels": num_channels,
                        "num_frames": num_frames,
                        "duration_sec": duration_sec,
                    })

    # Canonical ordering makes the seeded shuffle independent of source metadata order.
    df = pd.DataFrame(records).sort_values("utt_id").reset_index(drop=True)

    # Duration bounds are inclusive and refer to the original, unprocessed recording.
    mask = ((df["duration_sec"] >= cfg.min_duration_sec) & (df["duration_sec"] <= cfg.max_duration_sec))
    df_filtered = df[mask].copy().reset_index(drop=True)

    # A local NumPy generator keeps split selection independent of global RNG use.
    rng = np.random.default_rng(seed=cfg.seed)
    shuffled_idx = rng.permutation(len(df_filtered))
    df_shuffled = df_filtered.iloc[shuffled_idx].reset_index(drop=True)

    # None keeps every record; slicing also safely handles equal or oversized limits.
    df_subset = df_shuffled
    if cfg.subset_size is not None: df_subset = df_shuffled.iloc[: int(cfg.subset_size)].reset_index(drop=True)

    # Integer rounding applies to train and validation; test receives the remainder.
    n_total = len(df_subset)
    n_train = int(n_total * cfg.split_ratios[0])
    n_val = int(n_total * cfg.split_ratios[1])

    splits = {
        "train": df_subset.iloc[:n_train].reset_index(drop=True),
        "val": df_subset.iloc[n_train : n_train + n_val].reset_index(drop=True),
        "test": df_subset.iloc[n_train + n_val :].reset_index(drop=True),
    }

    # Save manifests and split metadata
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dataframes = []
    for split_name, split_df in splits.items():
        manifest = out_dir / f"{split_name}_manifest.csv"
        columns = ['wav_path', 'utt_id', 'raw_text', 'norm_text', 'clean_text']
        split_df[columns].to_csv(manifest, index=False, encoding="utf-8")

        # Return the same columns and CSV-loaded representation used by later runs.
        split_dataframes.append(load_manifest_df(manifest))

    # Hours describe source recordings, before waveform trimming or resampling.
    summary_meta = {
        "config": cfg.to_dict(),
        "raw_count": len(df),
        "filtered_count": len(df_filtered),
        "selected_subset_count": n_total,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "split_hours": {k: round(float(v["duration_sec"].sum() / 3600.0), 3) for k, v in splits.items()},
    }
    with open(out_dir / "split_summary.json", "w", encoding="utf-8") as jf: json.dump(summary_meta, jf, indent=2)

    return split_dataframes[0], split_dataframes[1], split_dataframes[2], summary_meta


def load_wav(wav_path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
    """Load a 16-bit PCM WAV as a normalized float32 tensor with shape (C, T)."""
    with wave.open(str(wav_path), "rb") as wf:
        original_sr = wf.getframerate()
        num_channels = wf.getnchannels()
        num_frames = wf.getnframes()
        raw_bytes = wf.readframes(num_frames)

    # This reader assumes signed 16-bit PCM, as used by the LJSpeech recordings.
    # Dividing by 2**15 maps samples into [-1, 1).
    audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    # PCM channels are interleaved; transpose (time, channels) to (channels, time).
    audio_tensor = torch.from_numpy(audio_np).reshape(-1, num_channels).t().contiguous()
    return audio_tensor, original_sr


class LJSpeechDataset(Dataset):

    """
    Text pre-processing is as follows:
    1. Take the normalised text from the 'norm_text' column of the manifest.
    2. Clean the text as needed (e.g., removing unwanted characters or formatting).
    """

    def __init__(
        self,
        split_source: Union[pd.DataFrame, str, Path],
        cfg: Optional[DataConfig] = None,
        *,
        tokenizer: PreTrainedTokenizerBase,
    ):
        """Use the required checkpoint tokenizer without caching token IDs."""
        if tokenizer is None:
            raise ValueError("A checkpoint tokenizer is required.")
        self.cfg = cfg or DataConfig()
        if isinstance(split_source, (str, Path)): self.df = load_manifest_df(split_source)
        # Dataset indices are positional, regardless of the caller's DataFrame index.
        else: self.df = split_source.reset_index(drop=True)

        # Reuse the supplied checkpoint vocabulary and tokenizer settings across splits.
        self.tokenizer = tokenizer
        self.audio_processor = AudioProcessor(self.cfg)

    def __len__(self) -> int: return len(self.df)

    def _get_token_ids(self, idx: int) -> torch.LongTensor:
        """
        Retrieve token IDs for the given transcript index.

        Raises an error if unknown tokens are present or if the transcript produces no tokens.
        """

        row = self.df.iloc[idx]
        # Tokenize on every access so checkpoint settings remain authoritative.
        ids = self.tokenizer(row["clean_text"])["input_ids"]
        unknown_id = self.tokenizer.unk_token_id
        # Reject vocabulary mismatches before loading audio; include the offending transcript.
        if unknown_id is not None:
            unknown_count = ids.count(unknown_id)
            if unknown_count:
                raise ValueError(
                    f"Unknown tokens ({unknown_count} occurrences of "
                    f"{self.tokenizer.unk_token!r}, ID {unknown_id}) "
                    f"in transcript {row['utt_id']!r}: {row['clean_text']!r}"
                )
        if not ids:
            raise ValueError(f"Transcript produced no tokens: {row['clean_text']!r}")
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        token_ids = self._get_token_ids(idx)

        # Waveform processing and spectrogram generation share the same audio configuration.
        raw_wav, original_sr = load_wav(row["wav_path"])
        clean_wav = self.audio_processor.preprocess_waveform(raw_wav, original_sr=original_sr)
        linear_spec, log_mel = self.audio_processor.compute_spectrograms(clean_wav)

        return {
            "utt_id": row["utt_id"],
            "wav_path": row["wav_path"],
            "raw_text": row["raw_text"],
            "norm_text": row["norm_text"],
            "clean_text": row["clean_text"],
            "token_ids": token_ids,
            "waveform": clean_wav, # (1, T_wav)
            "linear_spec": linear_spec, # (513, T_frames) 
            "log_mel": log_mel, # (80, T_frames)
            "sample_rate": self.cfg.target_sr,
            "duration_sec": clean_wav.size(-1) / float(self.cfg.target_sr),
        }

class TTSCollate:
    """Pad variable-length samples and assemble VITS inputs and loss targets."""

    def __init__(self, speaker_id: Optional[int], segment_frames: int):
        # segment_frames is a spectrogram-frame count, not a waveform-sample count.
        self.speaker_id = speaker_id
        self.segment_frames = segment_frames

    def __call__(self, batch: List[Dict]) -> Dict:
        # Keep metadata and all tensors in the same descending audio-length order.
        batch = sorted(batch, key=lambda x: x["linear_spec"].size(-1), reverse=True)

        b_size = len(batch)
        max_text_len = max(x["token_ids"].size(0) for x in batch)
        max_spec_len = max(x["linear_spec"].size(-1) for x in batch)
        max_wav_len = max(x["waveform"].size(-1) for x in batch)

        n_lin_bins = batch[0]["linear_spec"].size(0)
        n_mels = batch[0]["log_mel"].size(0)

        # Allocate CPU tensors; the training framework handles device transfer later.
        # Text uses (batch, tokens), while spectral features use (batch, bins, frames).
        text_padded = torch.zeros(b_size, max_text_len, dtype=torch.long)
        text_lengths = torch.LongTensor([x["token_ids"].size(0) for x in batch])

        spec_padded = torch.zeros(b_size, n_lin_bins, max_spec_len, dtype=torch.float32)
        mel_padded = torch.zeros(b_size, n_mels, max_spec_len, dtype=torch.float32)
        spec_lengths = torch.LongTensor([x["linear_spec"].size(-1) for x in batch])

        # Monotonic alignment needs at least one audio frame for each text token.
        if torch.any(text_lengths < 1) or torch.any(spec_lengths < text_lengths):
            raise ValueError("Each recording must have at least one audio frame per text token.")
        # Segment sampling must fit within every recording, not just the padded batch.
        if torch.any(spec_lengths < self.segment_frames):
            raise ValueError("A recording is shorter than the model's training segment.")

        # Waveform lengths are in samples and use a separate padding extent from frames.
        wav_padded = torch.zeros(b_size, 1, max_wav_len, dtype=torch.float32)
        wav_lengths = torch.LongTensor([x["waveform"].size(-1) for x in batch])

        # Copy valid prefixes only; the remaining suffixes retain their zero padding.
        for i, item in enumerate(batch):
            t_len = item["token_ids"].size(0)
            s_len = item["linear_spec"].size(-1)
            w_len = item["waveform"].size(-1)

            text_padded[i, :t_len] = item["token_ids"]
            spec_padded[i, :, :s_len] = item["linear_spec"]
            mel_padded[i, :, :s_len] = item["log_mel"]
            wav_padded[i, :, :w_len] = item["waveform"]

        # Zero is a valid VITS blank, so masks must use lengths rather than values.
        text_mask = torch.arange(max_text_len)[None, :] < text_lengths[:, None]
        spec_mask = torch.arange(max_spec_len)[None, :] < spec_lengths[:, None]
        # None disables conditioning for single-speaker models; otherwise share one speaker.
        if self.speaker_id is None:
            speakers = None
        else:
            speakers = torch.full((b_size,), self.speaker_id, dtype=torch.long)

        return {
            "utt_ids": [x["utt_id"] for x in batch],
            "norm_texts": [x["norm_text"] for x in batch],
            # Only this nested dictionary is intended for model.forward keyword arguments.
            "model_inputs": {
                "input_ids": text_padded,
                "attention_mask": text_mask,
                "labels": spec_padded,
                "labels_attention_mask": spec_mask,
                "speaker_id": speakers,
            },
            # Mel and waveform targets stay outside model_inputs for loss computation.
            "mel_padded": mel_padded,
            "wav_padded": wav_padded,
            "wav_lengths": wav_lengths,
        }


def build_dataloaders(
    cfg: DataConfig, batch_size: int = 16, num_workers: int = 0,
    *, tokenizer: PreTrainedTokenizerBase, speaker_id: Optional[int] = None, segment_frames: int,
) -> Dict[str, DataLoader]:
    """
    Builds deterministic Train, Val, and Test DataLoaders from saved manifests
    (or creates the manifests first if they do not exist yet).
    Uses a seeded torch.Generator so expA and expB see identical batch orders.
    Validation uses the configured evaluation batch size and sample limit.
    Pass the checkpoint tokenizer, model's segment length in frames and selected speaker ID.
    """
    if tokenizer is None:
        raise ValueError("A checkpoint tokenizer is required.")
    split_dir = Path(cfg.output_split_dir)
    manifests = [split_dir / f"{name}_manifest.csv" for name in ("train", "val", "test")]
    # Rebuild all splits if any manifest is missing, rather than using a partial set.
    if not all(manifest.exists() for manifest in manifests):
        build_splits(cfg)

    collate_fn = TTSCollate(speaker_id, segment_frames)

    # An independent PyTorch generator makes training batch order repeatable across runs.
    train_gen = torch.Generator()
    train_gen.manual_seed(cfg.seed)

    loaders = {}
    for split_name in ("train", "val", "test"):
        manifest_path = split_dir / f"{split_name}_manifest.csv"
        dataset = LJSpeechDataset(manifest_path, cfg=cfg, tokenizer=tokenizer)
        # Evaluation takes a deterministic prefix; training and test remain unrestricted.
        if split_name == "val" and cfg.do_eval and cfg.max_eval_samples is not None:
            dataset = Subset(dataset, range(min(len(dataset), cfg.max_eval_samples)))
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=cfg.per_device_eval_batch_size if split_name == "val" else batch_size,
            shuffle=(split_name == "train"),
            generator=train_gen if split_name == "train" else None,
            num_workers=num_workers,
            # Persistent workers are valid only when worker subprocesses actually exist.
            persistent_workers=cfg.persistent_workers and num_workers > 0,
            collate_fn=collate_fn,
            worker_init_fn=_configure_espeak_worker,
            # Pinned CPU memory supports faster transfers when training uses CUDA.
            pin_memory=torch.cuda.is_available(),
        )

    return loaders
