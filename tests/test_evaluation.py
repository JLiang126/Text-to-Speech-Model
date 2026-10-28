"""Tests for the evaluation package. Run from the repo root: ``python -m pytest tests -q``.

None of these need internet access. The VITS synthesis test runs only when the
official VITS code is available (``VITS_DIR`` env var or ./vits-main).
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.evaluation import audio, manifest, metrics, stats, timing
from scripts.evaluation.prompts import PROMPTS

SR = 22050


def tone(seconds=1.0, freq=220.0, sr=SR, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def speechlike(seconds=1.5, sr=SR, seed=0):
    """Harmonic signal with a moving pitch: rich enough spectrum for cepstral tests."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 120 + 30 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    signal = sum(np.sin(k * phase) / k for k in range(1, 15))
    return (0.3 * signal / np.max(np.abs(signal)) + 0.002 * rng.standard_normal(t.size)).astype(np.float32)


# ------------------------------------------------------------------ audio
def test_resample_length_and_identity():
    x = tone(1.0)
    y = audio.resample(x, SR, 16000)
    assert abs(len(y) - 16000) <= 1
    assert audio.resample(x, SR, SR) is not None and len(audio.resample(x, SR, SR)) == len(x)


def test_peak_normalise_and_silent_input():
    assert np.isclose(np.max(np.abs(audio.peak_normalise(tone(amp=0.1)))), 0.95, atol=1e-4)
    assert np.all(audio.peak_normalise(np.zeros(100, dtype=np.float32)) == 0)


def test_trim_removes_leading_and_trailing_silence():
    x = np.concatenate([np.zeros(SR // 2), tone(1.0), np.zeros(SR // 2)]).astype(np.float32)
    trimmed = audio.trim_silence(x, SR, top_db=40)
    assert abs(len(trimmed) / SR - 1.0) < 0.05


def test_save_and_load_roundtrip(tmp_path):
    x = audio.peak_normalise(tone(0.5))
    path = audio.save_wav(tmp_path / "sub" / "x.wav", x, SR)
    y, sr = audio.load_audio(path)
    assert sr == SR and len(y) == len(x)
    assert np.max(np.abs(x - y)) < 1e-3  # 16-bit quantisation only


# --------------------------------------------------------------- manifest
def _write_manifest(tmp_path, wav_path_value):
    data_root = tmp_path / "LJSpeech-1.1"
    (data_root / "wavs").mkdir(parents=True)
    audio.save_wav(data_root / "wavs" / "LJ001-0001.wav", tone(0.5), SR)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    pd.DataFrame([{"wav_path": wav_path_value, "utt_id": "LJ001-0001", "raw_text": "Hello.",
                   "norm_text": "Hello.", "clean_text": "hello."}]).to_csv(split_dir / "test_manifest.csv", index=False)
    return data_root, split_dir / "test_manifest.csv"


def test_manifest_resolves_paths_relative_to_a_different_cwd(tmp_path):
    data_root, manifest_path = _write_manifest(tmp_path, "LJSpeech-1.1/wavs/LJ001-0001.wav")
    df = manifest.load_manifest(manifest_path)  # path is relative to tmp_path, not to the cwd
    assert Path(df.loc[0, "wav_path"]).is_file()
    assert df.loc[0, "ref_text"] == "Hello."


def test_manifest_falls_back_to_data_root(tmp_path):
    data_root, manifest_path = _write_manifest(tmp_path, "somewhere/else/LJ001-0001.wav")
    with pytest.raises(FileNotFoundError):
        manifest.load_manifest(manifest_path)
    df = manifest.load_manifest(manifest_path, data_root=data_root)
    assert Path(df.loc[0, "wav_path"]).is_file()


# ----------------------------------------------------------------- MCD
def test_mcd_is_zero_for_identical_audio():
    x = speechlike()
    assert metrics.mcd_dtw(x, x, SR)["mcd_dtw"] < 1e-6


def test_mcd_ignores_loudness():
    x = speechlike()
    assert metrics.mcd_dtw(x, 0.5 * x, SR)["mcd_dtw"] < 1e-3  # c0 (energy) is excluded


def test_mcd_grows_with_distortion():
    x = speechlike()
    rng = np.random.default_rng(1)
    small = x + 0.01 * rng.standard_normal(x.size).astype(np.float32)
    large = x + 0.1 * rng.standard_normal(x.size).astype(np.float32)
    assert 0 < metrics.mcd_dtw(x, small, SR)["mcd_dtw"] < metrics.mcd_dtw(x, large, SR)["mcd_dtw"]


def test_mcd_dtw_absorbs_a_timing_offset():
    x = speechlike(seed=2)
    shifted = np.concatenate([np.zeros(2048, dtype=np.float32), x[:-2048]])
    unrelated = speechlike(seed=3)[::-1].copy()
    result = metrics.mcd_dtw(x, shifted, SR)
    assert result["mcd_dtw"] < metrics.mcd_dtw(x, unrelated, SR)["mcd_dtw"]
    assert abs(result["dur_ratio"] - 1.0) < 1e-6


# ------------------------------------------------------------- WER / CER
def test_wer_ignores_case_and_punctuation():
    r = metrics.word_char_errors("Hello, World!", "hello world", metrics.get_text_normaliser())
    assert r["wer"] == 0 and r["cer"] == 0


def test_wer_counts_errors_for_corpus_rate():
    r = metrics.word_char_errors("the cat sat on the mat", "the cat sat on a hat", metrics._basic_normalise)
    assert r["word_errors"] == 2 and r["ref_words"] == 6
    assert np.isclose(r["wer"], 2 / 6)


def test_empty_hypothesis_is_all_deletions_and_empty_reference_is_nan():
    assert metrics.word_char_errors("three words here", "", metrics._basic_normalise)["wer"] == 1.0
    assert np.isnan(metrics.word_char_errors("", "anything", metrics._basic_normalise)["wer"])


def test_whisper_normaliser_matches_spelled_out_numbers():
    normaliser = metrics.get_text_normaliser()
    if normaliser is metrics._basic_normalise:
        pytest.skip("openai-whisper not installed")
    r = metrics.word_char_errors("He paid twenty one dollars.", "He paid $21.", normaliser)
    assert r["wer"] == 0


# --------------------------------------------------- speaker similarity maths
def test_cosine_and_centroid():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert np.isclose(metrics.cosine_similarity(a, a), 1.0)
    assert np.isclose(metrics.cosine_similarity(a, b), 0.0)
    c = metrics.speaker_centroid([a, b])
    assert np.isclose(np.linalg.norm(c), 1.0) and np.isclose(metrics.cosine_similarity(c, a), np.sqrt(0.5))


# ------------------------------------------------------------------ stats
def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    x = np.random.default_rng(0).normal(5.0, 1.0, 200)
    mean, lo, hi = stats.bootstrap_ci(x, seed=3)
    assert lo < mean < hi and np.isclose(mean, x.mean())
    assert stats.bootstrap_ci(x, seed=3) == (mean, lo, hi)


def test_bootstrap_ignores_nan_and_handles_tiny_inputs():
    assert stats.bootstrap_ci([1.0, np.nan, 3.0])[0] == 2.0
    assert np.isnan(stats.bootstrap_ci([np.nan])[0])
    assert stats.bootstrap_ci([4.0]) == (4.0, 4.0, 4.0)


def test_ratio_ci_is_corpus_level():
    point, lo, hi = stats.bootstrap_ratio_ci([1, 0, 3], [10, 10, 10])
    assert np.isclose(point, 4 / 30) and lo <= point <= hi


def test_paired_comparison_detects_a_shift_and_not_noise():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 150)
    shifted = paired_comparison_result = stats.paired_comparison(a, a + 0.5 + rng.normal(0, 0.1, 150))
    assert shifted["ci_low"] > 0 and shifted["wilcoxon_p"] < 0.001
    same = stats.paired_comparison(a, a)
    assert same["mean_diff"] == 0 and same["wilcoxon_p"] == 1.0
    assert paired_comparison_result["n"] == 150


# ----------------------------------------------------------------- timing
def test_latency_protocol_with_a_fake_synthesiser():
    calls = []

    def fake(text):
        calls.append(text)
        time.sleep(0.01)
        return np.zeros(SR // 2, dtype=np.float32), {"frontend_s": 0.001, "model_s": 0.009}

    runs = timing.measure_latency(fake, ["a", "bb"], SR, device="cpu", n_warmup=2, n_repeats=3)
    assert len(calls) == 2 + 2 * 3 and len(runs) == 6
    summary = timing.summarise_latency(runs)
    assert summary["n_texts"] == 2 and summary["n_repeats"] == 3
    assert 0.015 < summary["rtf_mean"] < 0.2  # ~10 ms per 0.5 s of audio
    assert 0 < summary["frontend_share"] < 1


def test_describe_device_reports_hardware():
    info = timing.describe_device("cpu")
    assert info["device"] == "cpu" and info["cpu"] and info["torch"]


# ---------------------------------------------------------------- prompts
def test_at_least_ten_distinct_prompts():
    texts = [t for _, t in PROMPTS]
    assert len(texts) >= 10 and len(set(texts)) == len(texts)


# ------------------------------------------------------- VITS synthesis
def _vits_dir():
    for candidate in (os.environ.get("VITS_DIR"), "vits-main"):
        if candidate and (Path(candidate) / "models.py").is_file():
            return Path(candidate)
    return None


@pytest.mark.skipif(_vits_dir() is None, reason="official VITS code not found (set VITS_DIR)")
def test_vits_synthesis_is_seeded_and_checks_the_vocab(tmp_path):
    import torch
    from scripts.TextProcessor import TextProcessor
    from scripts.evaluation.synthesizer import VITSSynthesizer, import_vits

    vits_dir = _vits_dir()
    models, utils = import_vits(vits_dir)
    config = vits_dir / "configs" / "vctk_base.json"
    hps = utils.get_hparams_from_file(str(config))
    n_vocab = TextProcessor().vocab_size

    def random_checkpoint(vocab, name):
        torch.manual_seed(0)
        net = models.SynthesizerTrn(vocab, hps.data.filter_length // 2 + 1,
                                    hps.train.segment_size // hps.data.hop_length,
                                    n_speakers=hps.data.n_speakers, **hps.model)
        path = tmp_path / name
        torch.save({"model": net.state_dict(), "iteration": 0}, path)
        return path

    ckpt = random_checkpoint(n_vocab, "G_0.pth")
    synth = VITSSynthesizer(config, ckpt, vits_dir=vits_dir, device="cpu", speaker_id=0)
    a1, info = synth.synthesize("Hello world.", seed=7)
    a2, _ = synth.synthesize("Hello world.", seed=7)
    a3, _ = synth.synthesize("Hello world.", seed=8)
    assert a1.ndim == 1 and a1.size > 0 and info["n_tokens"] > 0
    assert np.array_equal(a1, a2) and not np.array_equal(a1, a3)

    with pytest.raises(ValueError, match="speaker"):
        VITSSynthesizer(config, ckpt, vits_dir=vits_dir, device="cpu", speaker_id=None)
    with pytest.raises(ValueError, match="symbols"):
        VITSSynthesizer(config, random_checkpoint(n_vocab + 1, "G_bad.pth"), vits_dir=vits_dir,
                        device="cpu", speaker_id=0)
