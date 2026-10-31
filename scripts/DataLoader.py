"""LJSpeech splitting and checkpoint-aware dataset loading.

Text preprocessing has two stages. During split generation,
clean_normalised_text_simple applies NFKC normalization, replaces brackets with
comma pauses attached to the preceding word, and collapses whitespace. Leading
brackets add no comma and existing adjacent punctuation is not duplicated.
Unicode letters, case, digits, and other punctuation are preserved. The result
is stored as clean_text in the manifest. Token IDs are generated from that field
using the checkpoint tokeniser and then cached in dataset memory.

What the VITS checkpoint tokeniser is doing:

With the current VCTK checkpoint defaults (normalize=True, phonemize=True,
add_blank=True), its preprocessing is:

1. Vocabulary-aware case normalization: recognized vocabulary characters and
   tokens are preserved; other characters are lowercased. The VCTK vocabulary
   contains both uppercase and lowercase letters, so this is not blanket
   lowercasing.
2. Phonemization through Phonemizer's eSpeak backend, using US English (en-us),
   IPA pronunciation with stress markers, punctuation preservation, and
   trimming of surrounding phonemizer whitespace.
3. Whitespace normalization: repeated whitespace in the phoneme output becomes
   a single space.
4. Character-level tokenization: the phoneme string is split into individual
   characters and mapped to vocabulary IDs. Unknown characters map to <unk>.
   Transcripts containing unknown IDs raise an error before caching or use.
5. Blank insertion: token ID 0 is inserted between characters and at both ends.
   Training overrides add_blank with the corresponding DataConfig setting.

The checkpoint tokeniser does not call local TextProcessor functions; transcript
cleaning is a separate stage in this pipeline. The older clean_normalised_text
function remains available, but its ASCII stripping, lowercasing, punctuation
rewriting, and digit/character filtering are no longer used in training.

The local TextProcessor phonemization and token-conversion methods are only
used as a fallback when no checkpoint tokeniser is supplied; unmapped phoneme
characters raise an error rather than being discarded. The local word-mismatch
warning filter is also used, but affects logging, not token IDs.

Summary:
Normalised transcript → simple cleaner → vocabulary-aware case normalization → 
phonemisation → whitespace cleanup → character IDs and blanks → unknown-token guard.

"""

import os
import wave
import random
import json
import csv
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from scripts.DataConfig import DataConfig
from scripts.TextProcessor import TextProcessor, clean_normalised_text_simple, configure_word_mismatch_warnings
from scripts.AudioProcessor import AudioProcessor

def set_global_seed(seed: int = 42) -> None:
    """Fixes random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


"""Loads a saved pipe-delimited split manifest (wav_path|utt_id|norm_text)."""
def load_manifest_df(manifest_path: Union[str, Path]) -> pd.DataFrame: return pd.read_csv(manifest_path)


class _WordMismatchHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("words count mismatch on "):
            self.messages.append(message)
        else:
            logging.getLogger(__name__).handle(record)


def save_word_mismatch_report(
    split_source: Union[pd.DataFrame, str, Path],
    output_path: Union[str, Path],
) -> Path:
    """
    Note: this function has been written by AI.
    
    Save only transcripts that trigger eSpeak word-count mismatch warnings.

    Accepts a split manifest or DataFrame with a clean_text column. Uses the
    English VITS phonemization settings and preserves every source column,
    adding phonemes and mismatch_warning. This is an explicit diagnostic pass;
    it does not change dataset tokenization or run automatically during training.
    """
    from phonemizer.backend import EspeakBackend

    df = load_manifest_df(split_source) if isinstance(split_source, (str, Path)) else split_source
    if "clean_text" not in df.columns:
        raise ValueError("Word mismatch reports require a clean_text column.")
    if not df.columns.is_unique or {"phonemes", "mismatch_warning"} & set(df.columns):
        raise ValueError("Report columns must be unique and not contain phonemes or mismatch_warning.")
    for index, text in df["clean_text"].items():
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Transcript at index {index!r} must be a nonempty string.")

    handler = _WordMismatchHandler()
    report_logger = logging.Logger(f"{__name__}.word_mismatch_report", level=logging.WARNING)
    report_logger.addHandler(handler)
    backend = EspeakBackend(
        language="en-us", preserve_punctuation=True, with_stress=True,
        words_mismatch="warn", logger=report_logger,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*df.columns, "phonemes", "mismatch_warning"])
        writer.writeheader()
        for row in df.to_dict(orient="records"):
            handler.messages.clear()
            phonemes = backend.phonemize([row["clean_text"]], strip=True)[0]
            if handler.messages:
                writer.writerow({
                    **row, "phonemes": phonemes,
                    "mismatch_warning": "; ".join(handler.messages),
                })
                count += 1
    logging.getLogger(__name__).info(
        "Saved %s word-count mismatch transcripts to %s", count, output_path
    )
    return output_path


def build_splits(
    cfg: DataConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    Executes splitting pipeline:
    1. Lexicographical sort by utt_id for OS independent ordering.
    2. Duration filtering clips under a second and over 10 are filtered out.
    3. Seeded shuffle seed=42.
    4. Randle samples a subset.
    5. Splits data.
    5. Exports train_manifest.txt, val_manifest.txt, and test_manifest.txt.

    Each manifest is saved as wav_path | ID | Normalised Text
    """
    data_root = Path(cfg.data_root)
    out_dir = Path(cfg.output_split_dir)
    try:
        out_dir.resolve().relative_to(data_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError(
            f"Split output directory must be outside the dataset directory: {out_dir}"
        )
    metadata_path = data_root / "metadata.csv"
    wavs_dir = data_root / "wavs"

    if not metadata_path.exists(): raise FileNotFoundError(f"LJSpeech metadata.csv not found at {metadata_path.resolve()}")

    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 3:
                utt_id = parts[0].strip()
                raw_text = parts[1].strip()
                norm_text = parts[2].strip()
                wav_path = wavs_dir / f"{utt_id}.wav"
                

                if wav_path.exists():
                    with wave.open(str(wav_path), "rb") as wf:
                        sample_rate = wf.getframerate()
                        num_channels = wf.getnchannels()
                        num_frames = wf.getnframes()
                    duration_sec = num_frames / float(sample_rate)
                    clean_text = clean_normalised_text_simple(norm_text)
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

    # Lexicographical sort by utt_id
    df = pd.DataFrame(records).sort_values("utt_id").reset_index(drop=True)

    # Filter out clips < 1.0s and > 10.0s
    mask = ((df["duration_sec"] >= cfg.min_duration_sec) & (df["duration_sec"] <= cfg.max_duration_sec))
    df_filtered = df[mask].copy().reset_index(drop=True)

    # Seeded shuffle using seed=42
    rng = np.random.default_rng(seed=cfg.seed)
    shuffled_idx = rng.permutation(len(df_filtered))
    df_shuffled = df_filtered.iloc[shuffled_idx].reset_index(drop=True)

    # Pull subset
    df_subset = df_shuffled
    if cfg.subset_size is not None:
        df_subset = df_shuffled.iloc[: int(cfg.subset_size)].reset_index(drop=True)

    # Split the dataset
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

        split_dataframes.append(load_manifest_df(manifest))

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
    """Loads WAV file into a normalisd float32 tensor of shape (C, T) in range of [-1.0, 1.0]."""
    with wave.open(str(wav_path), "rb") as wf:
        original_sr = wf.getframerate()
        num_channels = wf.getnchannels()
        num_frames = wf.getnframes()
        raw_bytes = wf.readframes(num_frames)

    audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    audio_tensor = torch.from_numpy(audio_np).reshape(-1, num_channels).t().contiguous()
    return audio_tensor, original_sr


class LJSpeechDataset(Dataset):
    def __init__(
        self,
        split_source: Union[pd.DataFrame, str, Path],
        cfg: Optional[DataConfig] = None,
        text_processor: Optional[TextProcessor] = None,
        tokenizer=None,
    ):
        """
        Initializes the LJSpeech dataset.

        Note, we'll use the checkpoint's own tokenization and symbol set

        Args:
            split_source: DataFrame or path to the manifest CSV file.
            cfg: Optional data configuration.
            text_processor: Optional TextProcessor instance for converting text to IPA tokens.
            tokenizer: Optional tokenizer for converting text to token IDs.
        """
        self.cfg = cfg or DataConfig()
        if isinstance(split_source, (str, Path)): self.df = load_manifest_df(split_source)
        else: self.df = split_source.reset_index(drop=True)

        if text_processor is not None and tokenizer is not None:
            raise ValueError("Provide either a text processor or a tokenizer, not both.")
        self.tokenizer = tokenizer
        self.text_processor = text_processor or (
            TextProcessor(add_blank=self.cfg.add_blank) if tokenizer is None else None
        )
        self.audio_processor = AudioProcessor(self.cfg)
        self._token_cache: Dict[int, torch.LongTensor] = {}

    def __len__(self) -> int: return len(self.df)

    def _get_token_ids(self, idx: int) -> torch.LongTensor:
        if idx not in self._token_cache:
            row = self.df.iloc[idx]
            # Spawned workers do not inherit the parent's logging filters.
            configure_word_mismatch_warnings(self.cfg.suppress_word_mismatch_warnings)
            if self.tokenizer is None:
                if self.text_processor is None:
                    raise RuntimeError("A text processor is required when no tokenizer is configured.")
                token_ids = self.text_processor.text_to_sequence(row["clean_text"], is_cleaned=True)
            else:
                ids = self.tokenizer(row["clean_text"])["input_ids"]
                unknown_id = self.tokenizer.unk_token_id
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
                token_ids = torch.tensor(ids, dtype=torch.long)
            self._token_cache[idx] = token_ids
        return self._token_cache[idx]

    def preprocess_token_ids(self, indices: Optional[Iterable[int]] = None) -> None:
        """Populate the in-memory token cache without loading or processing audio."""
        for idx in range(len(self)) if indices is None else indices:
            self._get_token_ids(idx)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        token_ids = self._get_token_ids(idx)

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
    """
    Zero pads the IPA token sequences, 513 bin linear spectrograms, 80 band log-mel spectrograms, and waveforms within the batch.
    """
    def __call__(self, batch: List[Dict]) -> Dict:
        batch = sorted(batch, key=lambda x: x["linear_spec"].size(-1), reverse=True)

        b_size = len(batch)
        max_text_len = max(x["token_ids"].size(0) for x in batch)
        max_spec_len = max(x["linear_spec"].size(-1) for x in batch)
        max_wav_len = max(x["waveform"].size(-1) for x in batch)

        n_lin_bins = batch[0]["linear_spec"].size(0)
        n_mels = batch[0]["log_mel"].size(0)

        text_padded = torch.zeros(b_size, max_text_len, dtype=torch.long)
        text_lengths = torch.LongTensor([x["token_ids"].size(0) for x in batch])

        spec_padded = torch.zeros(b_size, n_lin_bins, max_spec_len, dtype=torch.float32)
        mel_padded = torch.zeros(b_size, n_mels, max_spec_len, dtype=torch.float32)
        spec_lengths = torch.LongTensor([x["linear_spec"].size(-1) for x in batch])

        wav_padded = torch.zeros(b_size, 1, max_wav_len, dtype=torch.float32)
        wav_lengths = torch.LongTensor([x["waveform"].size(-1) for x in batch])

        for i, item in enumerate(batch):
            t_len = item["token_ids"].size(0)
            s_len = item["linear_spec"].size(-1)
            w_len = item["waveform"].size(-1)

            text_padded[i, :t_len] = item["token_ids"]
            spec_padded[i, :, :s_len] = item["linear_spec"]
            mel_padded[i, :, :s_len] = item["log_mel"]
            wav_padded[i, :, :w_len] = item["waveform"]

        return {
            "utt_ids": [x["utt_id"] for x in batch],
            "norm_texts": [x["norm_text"] for x in batch],
            "text_padded": text_padded, # Preserved original token counts
            "text_lengths": text_lengths,
            "spec_padded": spec_padded,
            "mel_padded": mel_padded,
            "spec_lengths": spec_lengths,
            "wav_padded": wav_padded,
            "wav_lengths": wav_lengths,
        }


def build_dataloaders(
    cfg: DataConfig,
    batch_size: int = 16,
    num_workers: int = 0,
    tokenizer=None,
) -> Dict[str, DataLoader]:
    """
    Builds deterministic Train, Val, and Test DataLoaders from saved manifests
    (or creates the manifests first if they do not exist yet).
    Uses a seeded torch.Generator so expA and expB see identical batch orders.
    The validation loader uses the configured evaluation batch size and sample limit.
    """
    split_dir = Path(cfg.output_split_dir)
    manifests = [split_dir / f"{name}_manifest.csv" for name in ("train", "val", "test")]
    if not all(manifest.exists() for manifest in manifests):
        build_splits(cfg)

    text_processor = TextProcessor(add_blank=cfg.add_blank) if tokenizer is None else None
    collate_fn = TTSCollate()

    train_gen = torch.Generator()
    train_gen.manual_seed(cfg.seed)

    loaders = {}
    for split_name in ("train", "val", "test"):
        manifest_path = split_dir / f"{split_name}_manifest.csv"
        dataset = LJSpeechDataset(
            manifest_path,
            cfg=cfg,
            text_processor=text_processor,
            tokenizer=tokenizer,
        )
        indices = range(len(dataset))
        if split_name == "val" and cfg.do_eval and cfg.max_eval_samples is not None:
            indices = range(min(len(dataset), cfg.max_eval_samples))
        if cfg.preprocess_token_ids:
            logging.getLogger(__name__).info(
                "Preprocessing token IDs for %s (%s transcripts)", split_name, len(indices)
            )
            dataset.preprocess_token_ids(indices)
        if split_name == "val" and cfg.do_eval and cfg.max_eval_samples is not None:
            dataset = Subset(dataset, indices)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=cfg.per_device_eval_batch_size if split_name == "val" else batch_size,
            shuffle=(split_name == "train"),
            generator=train_gen if split_name == "train" else None,
            num_workers=num_workers,
            persistent_workers=cfg.persistent_workers and num_workers > 0,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders
