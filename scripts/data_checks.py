"""
The feedback was: split sizes and batch shapes were checked, but nothing asserted that the
IDs are unique and the splits disjoint, nothing confirmed that every manifest file exists,
and the value range was tested on the first item only instead of on a whole batch.
These functions close those gaps. Each one raises AssertionError with a clear message.

Asserts that IDs are unique and the splits are disjoint and every manifest file exists.
Also verifies that the value ranges for batches
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd 

from scripts.evaluation.manifest import split_fingerprint 

def check_metadata_coverage(data_root) -> Dict[str, int]:
    """Every metadata.csv entry must have its wav file. build_splits skips missing files
    silently, so an incomplete download would quietly produce a different split."""
    data_root = Path(data_root)
    if not (data_root / "metadata.csv").is_file() and (data_root / "LJSpeech-1.1" / "metadata.csv").is_file():
        data_root = data_root / "LJSpeech-1.1"

    ids = [line.split("|", 1)[0].strip() for line in
           (data_root / "metadata.csv").read_text(encoding="utf-8").splitlines() if line.strip()]
    
    missing = [u for u in ids if not (data_root / "wavs" / f"{u}.wav").is_file()]
    assert not missing, (f"{len(missing)} of {len(ids)} metadata entries have no wav file "
                         f"(e.g. {missing[:3]}): re-extract LJSpeech before building splits")
    
    print(f"- metadata coverage: all {len(ids)} entries have a wav file")
    return {"n_metadata": len(ids), "n_missing": 0}


def check_split_ids(splits: Mapping[str, pd.DataFrame]) -> Dict[str, object]:
    """IDs and wav paths unique within each split, the splits pairwise disjoint, no empty text."""
    for name, df in splits.items():
        assert len(df) > 0, f"{name} split is empty"

        duplicated = df["utt_id"][df["utt_id"].duplicated()].tolist()
        assert not duplicated, f"{name}: repeated utterance IDs {duplicated[:5]}"
        assert not df["wav_path"].duplicated().any(), f"{name}: repeated wav paths"

        text = "clean_text" if "clean_text" in df.columns else "norm_text"
        empty = df["utt_id"][df[text].fillna("").astype(str).str.strip() == ""].tolist()
        assert not empty, f"{name}: empty {text} for {empty[:5]}"

    for (a, df_a), (b, df_b) in combinations(splits.items(), 2):
        shared = set(df_a["utt_id"]) & set(df_b["utt_id"])
        assert not shared, f"{len(shared)} utterances are in both {a} and {b}, e.g. {sorted(shared)[:3]}"

        shared_paths = set(df_a["wav_path"]) & set(df_b["wav_path"])
        assert not shared_paths, f"{len(shared_paths)} wav files are in both {a} and {b}"

    fingerprints = {name: split_fingerprint(df["utt_id"]) for name, df in splits.items()}
    print(f"- IDs: unique within each split, splits pairwise disjoint")
    return {"fingerprints": fingerprints}


def check_split_files(splits: Mapping[str, pd.DataFrame], sample_rate: int = 22050, min_sec: float = 1.0,
                      max_sec: float = 10.0) -> Dict[str, int]:
    """Every manifest file exists and is a mono wav at ``sample_rate`` within the duration limits.
    Reads headers only, so all files are checked, not a sample."""
    import soundfile as sf

    problems, n_files = [], 0
    for name, df in splits.items():
        for utt_id, wav_path in zip(df["utt_id"], df["wav_path"]):
            n_files += 1
            path = Path(wav_path)
            if not path.is_file():
                problems.append(f"{name}/{utt_id}: missing {wav_path}")
                continue

            info = sf.info(str(path))
            duration = info.frames / float(info.samplerate)

            if info.samplerate != sample_rate: problems.append(f"{name}/{utt_id}: {info.samplerate} Hz")
            if info.channels != 1: problems.append(f"{name}/{utt_id}: {info.channels} channels")
            if not min_sec <= duration <= max_sec: problems.append(f"{name}/{utt_id}: {duration:.2f} s outside [{min_sec}, {max_sec}]")

    assert not problems, f"{len(problems)} file problems, e.g. {problems[:5]}"
    print(f"- files: all {n_files} manifest wavs exist, mono, {sample_rate} Hz, {min_sec}-{max_sec} s")
    return {"n_files": n_files}


def check_split_proportions(splits: Mapping[str, pd.DataFrame], ratios: Sequence[float], tolerance: float = 0.01) -> Dict[str, float]:
    """Split sizes against the configured ratios (rounding allows ``tolerance``)."""
    total = sum(len(df) for df in splits.values())
    shares = {name: len(df) / total for name, df in splits.items()}
    for (name, share), ratio in zip(shares.items(), ratios):
        assert abs(share - ratio) <= tolerance, f"{name} is {share:.3f} of the data, configured {ratio}"

    print("- sizes: " + ", ".join(f"{n} {len(df)} ({100 * shares[n]:.1f}%)" for n, df in splits.items()) + f" of {total}")
    return shares


def run_split_checks(splits: Mapping[str, pd.DataFrame], cfg) -> Dict[str, object]:
    """All split-level checks for a DataConfig (``cfg``) and its three split DataFrames."""
    result = {"coverage": check_metadata_coverage(cfg.data_root), "ids": check_split_ids(splits)}
    result["files"] = check_split_files(splits, cfg.target_sr, cfg.min_duration_sec, cfg.max_duration_sec)
    result["shares"] = check_split_proportions(splits, cfg.split_ratios)
    return result


def check_batch(batch: Mapping[str, object], vocab_size: int, hop_length: Optional[int] = None,
                name: str = "batch", verbose: bool = True) -> Dict[str, float]:
    """Checks every item of a TTSCollate batch, not only the first one:
    shapes agree; token IDs inside the vocabulary; every waveform within [-1, 1] and not silent,
    spectrograms finite (linear magnitudes non-negative); zero padding after each item's length
    frames = samples // hop; items sorted by length."""
    import torch

    inputs = batch["model_inputs"]
    ids, text_mask = inputs["input_ids"], inputs["attention_mask"]
    linear, spec_mask = inputs["labels"], inputs["labels_attention_mask"]
    mel, wav, wav_lengths = batch["mel_padded"], batch["wav_padded"], batch["wav_lengths"]
    size = len(batch["utt_ids"])

    assert size == ids.shape[0] == linear.shape[0] == mel.shape[0] == wav.shape[0] == wav_lengths.shape[0], \
        f"{name}: tensors disagree on the batch size"
    assert len(set(batch["utt_ids"])) == size, f"{name}: repeated utterance IDs in one batch"

    assert ids.dtype == torch.long and int(ids.min()) >= 0 and int(ids.max()) < vocab_size, \
        f"{name}: token IDs outside [0, {vocab_size})"
    
    for label, tensor in (("linear spectrogram", linear), ("mel spectrogram", mel), ("waveform", wav)):
        assert bool(torch.isfinite(tensor).all()), f"{name}: non-finite values in the {label}"

    assert float(linear.min()) >= 0.0, f"{name}: negative linear-spectrogram magnitudes"
    assert float(wav.abs().max()) <= 1.0, f"{name}: waveform outside [-1, 1] (max |x| {float(wav.abs().max()):.3f})"

    text_lengths, spec_lengths = text_mask.sum(dim=1), spec_mask.sum(dim=1)
    assert spec_lengths.tolist() == sorted(spec_lengths.tolist(), reverse=True), f"{name}: not sorted by length"

    for i, utt in enumerate(batch["utt_ids"]):
        t_len, s_len, w_len = int(text_lengths[i]), int(spec_lengths[i]), int(wav_lengths[i])
        assert t_len >= 1 and s_len >= t_len, f"{name}/{utt}: fewer frames than tokens"
        assert float(wav[i, :, :w_len].abs().max()) > 0.01, f"{name}/{utt}: silent waveform"
        assert int(ids[i, t_len:].abs().sum()) == 0, f"{name}/{utt}: non-zero token padding"
        
        assert float(linear[i, :, s_len:].abs().sum()) == 0.0, f"{name}/{utt}: non-zero spectrogram padding"
        assert float(mel[i, :, s_len:].abs().sum()) == 0.0, f"{name}/{utt}: non-zero mel padding"
        assert float(wav[i, :, w_len:].abs().sum()) == 0.0, f"{name}/{utt}: non-zero waveform padding"
        if hop_length: assert s_len == w_len // hop_length, f"{name}/{utt}: {s_len} frames for {w_len} samples"

    stats = {"items": size, "wav_min": float(wav.min()), "wav_max": float(wav.max()),
             "mel_min": float(mel.min()), "mel_max": float(mel.max()),
             "token_min": int(ids.min()), "token_max": int(ids.max())}
    
    if verbose:
        print(f"- {name}: {size} items checked, mangitude within [{stats['wav_min']:.3f}, {stats['wav_max']:.3f}], "
              f"tokens withn [{stats['token_min']}, {stats['token_max']}], mel within [{stats['mel_min']:.2f}, {stats['mel_max']:.2f}]")
    return stats


def check_loader(loader, vocab_size: int, hop_length: Optional[int] = None, name: str = "loader") -> Dict[str, float]:
    """check_batch on every batch of a DataLoader: every item of the split is checked."""
    totals = {"items": 0, "batches": 0, "wav_min": 0.0, "wav_max": 0.0, "token_max": 0}
    for i, batch in enumerate(loader):
        stats = check_batch(batch, vocab_size, hop_length, name=f"{name} batch {i}", verbose=False)
        totals["items"] += stats["items"]
        totals["batches"] += 1
        totals["wav_min"] = min(totals["wav_min"], stats["wav_min"])
        totals["wav_max"] = max(totals["wav_max"], stats["wav_max"])
        totals["token_max"] = max(totals["token_max"], stats["token_max"])

    print(f"- {name}: all {totals['items']} items in {totals['batches']} batches passed, waveform "
          f"[{totals['wav_min']:.3f}, {totals['wav_max']:.3f}], largest token ID {totals['token_max']}")
    return totals


def evaluation_overlap(evaluated_ids: Iterable[str], splits: Mapping[str, pd.DataFrame]) -> Dict[str, int]:
    """How many evaluated sentences fall in each split. A held-out evaluation has train = val = 0."""
    evaluated = set(map(str, evaluated_ids))
    return {"evaluated": len(evaluated),
            **{name: len(evaluated & set(df["utt_id"].astype(str))) for name, df in splits.items()}}


# ------------------------------------------------------------------ command line
def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/no_droplayer.yaml", help="training config whose split to check")
    p.add_argument("--split_dir", default="dataset/splits", help="where to write the rebuilt split")
    p.add_argument("--batches", type=int, default=1, help="whole batches to check per split (0 = skip)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--check_results", default=None, help="an evaluate_experiments results folder: count its sentences in each split")
    args = p.parse_args(argv)

    from scripts.DataConfig import DataConfig, load_config
    from scripts.DataLoader import build_splits

    cfg = DataConfig(**load_config(Path(args.config)))
    cfg.output_split_dir = Path(args.split_dir)
    cfg.validate()
    print(f"[data] rebuilding the split of {args.config} into {cfg.output_split_dir}")
    train_df, val_df, test_df, _ = build_splits(cfg)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    result = run_split_checks(splits, cfg)

    if args.batches:
        from scripts.DataLoader import build_dataloaders
        from scripts.TextProcessor import load_vctk_tokenizer
        from scripts.configuration_vits import VitsConfig  # has segment_size, unlike transformers'
        from scripts.evaluation.synthesizer import _use_bundled_espeak

        _use_bundled_espeak()  # phonemizer needs espeak-ng: the pip-installed one if there is no system copy

        tokenizer = load_vctk_tokenizer(cfg.model_name_or_path)
        model_config = VitsConfig.from_pretrained(cfg.model_name_or_path)
        loaders = build_dataloaders(cfg=cfg, batch_size=args.batch_size, num_workers=0, tokenizer=tokenizer,
                                    speaker_id=cfg.speaker_id,
                                    segment_frames=model_config.segment_size // model_config.hop_length)
        
        for split, loader in loaders.items():
            for i, batch in zip(range(args.batches), loader): 
                check_batch(batch, model_config.vocab_size, cfg.hop_length, name=f"{split} batch {i}")

    if args.check_results:
        per_utt = sorted(Path(args.check_results).glob("*/per_utterance.csv"))

        if per_utt:
            overlap = evaluation_overlap(pd.read_csv(per_utt[0])["utt_id"], splits)
            result["evaluation_overlap"] = overlap
            print(f"- evaluation in {args.check_results}: {overlap['evaluated']} sentences, of which "
                  f"{overlap['train']} in train, {overlap['val']} in val, {overlap['test']} in test")
            
            if overlap["train"] or overlap["val"]: print("  -> NOT held out: re-score on this split's test_manifest.csv")

    print("[data] all checks passed")
    return result


if __name__ == "__main__":
    main()
