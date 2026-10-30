"""Tests for the evaluation package. Run from the repo root: ``python -m pytest tests -q``.

None of these need internet access. The VITS tests run only when the official VITS
code is available (``VITS_DIR`` environment variable, or ./vits-main).
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
from scripts.evaluation import speakers
from scripts.evaluation.synthesizer import checkpoint_step
from scripts.evaluation.text_normalise import normalise_input_text

SR = 22050


def tone(seconds=1.0, freq=220.0, sr=SR, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def speechlike(seconds=1.5, sr=SR, seed=0):
    """Harmonic signal with a moving pitch: a rich enough spectrum for cepstral tests."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 120 + 30 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    signal = sum(np.sin(k * phase) / k for k in range(1, 15))
    return (0.3 * signal / np.max(np.abs(signal)) + 0.002 * rng.standard_normal(t.size)).astype(np.float32)


# ------------------------------------------------------------------ audio
def test_resample_length_and_identity():
    x = tone(1.0)
    assert abs(len(audio.resample(x, SR, 16000)) - 16000) <= 1
    assert len(audio.resample(x, SR, SR)) == len(x)


def test_peak_normalise_and_silent_input():
    assert np.isclose(np.max(np.abs(audio.peak_normalise(tone(amp=0.1)))), 0.95, atol=1e-4)
    assert np.all(audio.peak_normalise(np.zeros(100, dtype=np.float32)) == 0)


def test_trim_removes_leading_and_trailing_silence():
    x = np.concatenate([np.zeros(SR // 2), tone(1.0), np.zeros(SR // 2)]).astype(np.float32)
    assert abs(len(audio.trim_silence(x, SR, top_db=40)) / SR - 1.0) < 0.05


def test_save_and_load_roundtrip(tmp_path):
    x = audio.peak_normalise(tone(0.5))
    y, sr = audio.load_audio(audio.save_wav(tmp_path / "sub" / "x.wav", x, SR))
    assert sr == SR and len(y) == len(x)
    assert np.max(np.abs(x - y)) < 1e-3  # 16-bit quantisation only


# --------------------------------------------------------------- manifest
def _fake_dataset(tmp_path, n=6, wav_path_style="relative"):
    """A tiny LJSpeech-shaped dataset plus train/test manifests in the build_splits format."""
    data_root = tmp_path / "LJSpeech-1.1"
    (data_root / "wavs").mkdir(parents=True)
    rows = []
    for i in range(n):
        utt = f"LJ001-{i:04d}"
        audio.save_wav(data_root / "wavs" / f"{utt}.wav", speechlike(1.2 + 0.1 * i, seed=i), SR)
        wav_path = f"LJSpeech-1.1/wavs/{utt}.wav" if wav_path_style == "relative" else f"elsewhere/{utt}.wav"
        text = f"The clerk read sentence number {i + 1}."
        rows.append({"wav_path": wav_path, "utt_id": utt, "raw_text": text, "norm_text": text,
                     "clean_text": text.lower()})
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    pd.DataFrame(rows[: n // 2]).to_csv(split_dir / "train_manifest.csv", index=False)
    pd.DataFrame(rows[n // 2:]).to_csv(split_dir / "test_manifest.csv", index=False)
    return data_root, split_dir


def test_manifest_resolves_paths_relative_to_a_different_cwd(tmp_path):
    _, split_dir = _fake_dataset(tmp_path)
    df = manifest.load_manifest(split_dir / "test_manifest.csv")  # paths are relative to tmp_path, not the cwd
    assert all(Path(p).is_file() for p in df["wav_path"])
    assert df.loc[0, "ref_text"].startswith("The clerk")


def test_manifest_falls_back_to_data_root(tmp_path):
    data_root, split_dir = _fake_dataset(tmp_path, wav_path_style="broken")
    with pytest.raises(FileNotFoundError):
        manifest.load_manifest(split_dir / "test_manifest.csv")
    df = manifest.load_manifest(split_dir / "test_manifest.csv", data_root=data_root)
    assert all(Path(p).is_file() for p in df["wav_path"])


# ------------------------------------------------------- text normalisation
@pytest.mark.parametrize("text, expected", [
    ("In 1865.", "In eighteen sixty-five."),
    ("It cost $5.", "It cost five dollars."),
    ("Paid 12,500 dollars.", "Paid twelve thousand, five hundred dollars."),
    ("The 3rd time.", "The third time."),
    ("Up 10% & rising.", "Up ten percent and rising."),
    ("Pi is 3.14.", "Pi is three point fourteen."),
    ("No digits here.", "No digits here."),
])
def test_numbers_are_spelled_out(text, expected):
    assert normalise_input_text(text) == expected


def test_no_digits_survive_any_prompt():
    for _, text in PROMPTS:
        assert not any(ch.isdigit() for ch in normalise_input_text(text))


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


def test_cepstrum_uses_the_standard_scale():
    """A log spectrum a*cos(k*w) must give c_k = a/2, the convention the MCD constant assumes."""
    from scipy.fft import dct
    n, k, a = 80, 5, 3.0
    w = np.pi * (np.arange(n) + 0.5) / n
    c = dct(a * np.cos(k * w), type=2, norm="ortho") / np.sqrt(2 * n)
    assert np.isclose(c[k], a / 2)


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
    assert metrics.word_char_errors("He paid twenty one dollars.", "He paid $21.", normaliser)["wer"] == 0


# ------------------------------------------------ speaker similarity
def test_cosine_and_centroid():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert np.isclose(metrics.cosine_similarity(a, a), 1.0)
    assert np.isclose(metrics.cosine_similarity(a, b), 0.0)
    c = metrics.speaker_centroid([a, b])
    assert np.isclose(np.linalg.norm(c), 1.0) and np.isclose(metrics.cosine_similarity(c, a), np.sqrt(0.5))


def test_paired_and_centroid_similarity():
    real = np.array([[1.0, 0.0], [0.0, 1.0]])            # LJ sentence 0 and 1
    synth = np.array([[[2.0, 0.0], [0.0, 3.0]],          # speaker 0 matches both sentences
                      [[0.0, 1.0], [1.0, 0.0]]])         # speaker 1 has them swapped
    assert np.allclose(speakers.paired_similarity(synth, real), [[1, 1], [0, 0]])
    # both speakers have the same average voice, so their centroids tie
    assert np.allclose(speakers.centroid_similarity(synth, real), [1.0, 1.0])


def _paired_scores(means, n_utts=40, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(means)[:, None] + rng.normal(0, noise, (len(means), n_utts))


def test_rank_speakers_orders_with_uncertainty_and_skips_ineligible():
    paired = _paired_scores([0.8, 0.5, 0.3, 0.6, 0.95])
    eligible = [True, True, True, True, False]  # speaker 4 scores highest but was never trained
    table = speakers.rank_speakers(paired, [10, 11, 12, 13, 14], eligible, n_boot=300)
    ranked = table[table["eligible"]]
    assert ranked["speaker_index"].tolist() == [10, 13, 11, 12] and ranked["rank"].tolist() == [1, 2, 3, 4]
    assert table.iloc[-1]["speaker_index"] == 14 and pd.isna(table.iloc[-1]["rank"])
    top = ranked.iloc[0]
    assert top["ci_low"] <= top["similarity"] <= top["ci_high"]
    assert top["p_closest"] > 0.99 and ranked.iloc[-1]["p_furthest"] > 0.99
    assert ranked["rank_ci_low"].iloc[0] == 1
    again = speakers.rank_speakers(paired, [10, 11, 12, 13, 14], eligible, n_boot=300)
    pd.testing.assert_frame_equal(table, again)  # seeded bootstrap
    assert speakers.pick_conditions(table) == {"closest": 10, "median": 13, "furthest": 12}


def test_similarity_baselines_give_the_scale():
    rng = np.random.default_rng(0)
    lj = rng.normal(size=16)
    real = lj + 0.1 * rng.normal(size=(6, 16))                   # one voice, six sentences
    voices = rng.normal(size=(5, 16))
    synth = voices[:, None, :] + 0.1 * rng.normal(size=(5, 6, 16))  # five different voices
    base = speakers.similarity_baselines(synth, real, eligible=[True] * 4 + [False])
    assert base["real_vs_real"] > 0.95 and base["same_synthetic_speaker"] > 0.95
    assert abs(base["different_synthetic_speakers"]) < 0.5
    # the |sum|^2 shortcut equals the explicit mean over pairs
    u = real / np.linalg.norm(real, axis=1, keepdims=True)
    explicit = np.mean([u[i] @ u[j] for i in range(6) for j in range(i + 1, 6)])
    assert np.isclose(base["real_vs_real"], explicit)


def test_pitch_groups_pick_within_each_group():
    table = pd.DataFrame({"speaker_index": [1, 2, 3, 4, 5], "eligible": [True, True, True, True, False],
                          "similarity": [0.3, 0.1, 0.2, -0.1, 0.9],
                          "f0_median_hz": [110.0, 120.0, 210.0, 200.0, 205.0]})
    groups = speakers.pitch_groups(table, threshold_hz=160)
    assert groups["low"]["n"] == 2 and groups["low"]["closest"] == 1 and groups["low"]["furthest"] == 2
    assert groups["high"]["n"] == 2 and groups["high"]["closest"] == 3 and groups["high"]["furthest"] == 4


def test_split_half_reliability_separates_signal_from_noise():
    clear = speakers.split_half_reliability(_paired_scores(np.linspace(0.2, 0.9, 20)))
    noise = speakers.split_half_reliability(_paired_scores(np.full(20, 0.5), noise=0.2))
    assert clear["split_half_spearman"] > 0.95 and clear["spearman_brown"] >= clear["split_half_spearman"]
    assert abs(noise["split_half_spearman"]) < 0.6


def test_f0_median_tracks_pitch():
    low = speechlike(2.0, sr=16000)                         # pitch moves around 120 Hz
    high = audio.resample(speechlike(2.0, sr=16000), 16000, 8800)[:32000]  # same signal played faster
    assert abs(speakers.f0_median(low, 16000) - 120) < 12
    assert abs(speakers.f0_median(high, 16000) - 120 * 16000 / 8800) < 20
    assert np.isnan(speakers.f0_median(np.zeros(100, np.float32), 16000))


def test_vctk_speaker_table_lists_the_trained_indices():
    table = speakers.load_vctk_speakers()
    assert table.index.tolist() == list(range(108))  # 109 embeddings, index 108 never trained
    assert table["vctk_id"].is_unique and table["vctk_id"].str.match(r"^p\d{3}$").all()
    assert (table["n_train_utts"] > 0).all()


def test_checkpoint_step_from_both_formats():
    assert checkpoint_step("checkpoints/expA/G_10000.pth") == 10000
    assert checkpoint_step("runs/expA/checkpoint-2000") == 2000
    assert checkpoint_step("runs/expA/checkpoint-2000/") == 2000
    assert checkpoint_step("runs/expA") is None and checkpoint_step(None) is None


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
    shifted = stats.paired_comparison(a, a + 0.5 + rng.normal(0, 0.1, 150))
    assert shifted["n"] == 150 and shifted["ci_low"] > 0 and shifted["wilcoxon_p"] < 0.001
    assert shifted["cohens_dz"] > 2  # 0.5 shift with 0.1 noise is a huge effect
    same = stats.paired_comparison(a, a)
    assert same["mean_diff"] == 0 and same["wilcoxon_p"] == 1.0 and same["cohens_dz"] == 0


def test_holm_adjustment():
    adjusted = stats.holm_adjust([0.01, 0.04, 0.03, np.nan])
    assert np.allclose(adjusted[:3], [0.03, 0.06, 0.06]) and np.isnan(adjusted[3])


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


def test_device_description_and_tag():
    info = timing.describe_device("cpu")
    assert info["device"] == "cpu" and info["cpu"] and info["torch"]
    assert timing.device_tag("cuda:1") == "cuda" and timing.device_tag("cpu") == "cpu"


# ---------------------------------------------------------------- prompts
def test_at_least_ten_distinct_prompts():
    texts = [t for _, t in PROMPTS]
    assert len(texts) >= 10 and len(set(texts)) == len(texts)


# ------------------------------------------------------------- compare.py
def _fake_result(folder, name, utt_values, rtf=0.05, ground_truth=False, step=None):
    folder.mkdir(parents=True)
    per_utt = pd.DataFrame({"utt_id": [f"u{i}" for i in range(len(utt_values))], "mcd_dtw": utt_values,
                            "dur_ratio": 1.0 + 0.01 * np.arange(len(utt_values))})
    per_utt.to_csv(folder / "per_utterance.csv", index=False)
    summary = {"experiment": name, "ground_truth": ground_truth, "step": step, "n_utterances": len(utt_values),
               "metrics": {"mcd_dtw": dict(zip(["mean", "ci_low", "ci_high"], stats.bootstrap_ci(utt_values)),
                                           n=len(utt_values))}, "corpus": {}}
    (folder / "summary.json").write_text(json.dumps(summary))
    if not ground_truth:
        runs = pd.DataFrame({"text_id": [0, 0, 1, 1], "repeat": [0, 1, 0, 1], "latency_s": [0.1] * 4,
                             "audio_s": [0.1 / rtf] * 4, "rtf": [rtf] * 4})
        runs.to_csv(folder / "rtf_runs_cpu.csv", index=False)
        lat = {**timing.summarise_latency(runs), "device": {"cpu": "Test CPU"}, "experiment": name, "step": step}
        (folder / "latency_cpu.json").write_text(json.dumps(lat))


def test_compare_builds_table_tests_and_figures(tmp_path):
    from scripts import compare

    rng = np.random.default_rng(0)
    base = rng.normal(6.0, 0.5, 40)
    _fake_result(tmp_path / "pretrained", "pretrained", base + 2.0)
    _fake_result(tmp_path / "expA_1000", "expA", base + 0.5, step=1000)
    _fake_result(tmp_path / "expA_2000", "expA", base, step=2000)
    _fake_result(tmp_path / "expB_2000", "expB", base + 0.6 + rng.normal(0, 0.05, 40), step=2000)
    out = tmp_path / "comparison"
    result = compare.main([str(tmp_path / d) for d in ("pretrained", "expA_1000", "expA_2000", "expB_2000")]
                          + ["--pair", "expA@2000", "expB", "--out", str(out)])
    for name in ("comparison_table.csv", "comparison_table.md", "paired_tests.csv", "comparison.png",
                 "learning_curves.png"):
        assert (out / name).is_file(), name
    assert "RTF CPU" in result["table"].columns
    mcd = result["tests"].query("metric == 'mcd_dtw'").iloc[0]
    assert mcd["verdict"] == "expA@2000 better"  # expB has higher (worse) MCD on every sentence
    assert {"wilcoxon_p_holm", "cohens_dz", "better_if"} <= set(result["tests"].columns)


# ------------------------------------------------------- VITS synthesis
def _vits_dir():
    for candidate in (os.environ.get("VITS_DIR"), "vits-main"):
        if candidate and (Path(candidate) / "models.py").is_file():
            return Path(candidate)
    return None


needs_vits = pytest.mark.skipif(_vits_dir() is None, reason="official VITS code not found (set VITS_DIR)")


def _random_checkpoint(tmp_path, name, vocab_offset=0):
    """A random-weight multi-speaker VITS generator saved in the official checkpoint format."""
    import torch
    from scripts.TextProcessor import TextProcessor
    from scripts.evaluation.synthesizer import import_vits

    vits_dir = _vits_dir()
    models, utils = import_vits(vits_dir)
    config = vits_dir / "configs" / "vctk_base.json"
    hps = utils.get_hparams_from_file(str(config))
    torch.manual_seed(0)
    net = models.SynthesizerTrn(TextProcessor().vocab_size + vocab_offset, hps.data.filter_length // 2 + 1,
                                hps.train.segment_size // hps.data.hop_length,
                                n_speakers=hps.data.n_speakers, **hps.model)
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "iteration": 0}, path)
    return config, path


@needs_vits
def test_vits_synthesis_is_seeded_and_validates_inputs(tmp_path):
    from scripts.evaluation.synthesizer import VITSSynthesizer

    config, ckpt = _random_checkpoint(tmp_path, "G_0.pth")
    synth = VITSSynthesizer(config, ckpt, vits_dir=_vits_dir(), device="cpu", speaker_id=0)
    a1, info = synth.synthesize("Hello world.", seed=7)
    a2, _ = synth.synthesize("Hello world.", seed=7)
    a3, _ = synth.synthesize("Hello world.", seed=8)
    assert a1.ndim == 1 and a1.size > 0 and info["n_tokens"] > 0
    assert np.array_equal(a1, a2) and not np.array_equal(a1, a3)
    # digits are spelled out, not dropped
    assert len(synth.text_to_ids("I have 3 cats.")) > len(synth.text_to_ids("I have cats."))

    with pytest.raises(ValueError, match="multi-speaker"):
        VITSSynthesizer(config, ckpt, vits_dir=_vits_dir(), device="cpu", speaker_id=None)
    with pytest.raises(ValueError, match="between 0 and"):
        VITSSynthesizer(config, ckpt, vits_dir=_vits_dir(), device="cpu", speaker_id=500)
    _, bad = _random_checkpoint(tmp_path, "G_bad.pth", vocab_offset=1)
    with pytest.raises(ValueError, match="symbols"):
        VITSSynthesizer(config, bad, vits_dir=_vits_dir(), device="cpu", speaker_id=0)


# ------------------------------------------- Hugging Face format (finetune-hf-vits)
needs_transformers = pytest.mark.skipif(
    __import__("importlib").util.find_spec("transformers") is None, reason="transformers not installed")


def _tiny_hf_model(folder, num_speakers=4):
    """A small random transformers VitsModel + VitsTokenizer saved like a finetune-hf-vits run folder."""
    import torch
    from transformers import VitsConfig, VitsModel, VitsTokenizer
    from scripts.TextProcessor import VITS_SYMBOLS

    folder.mkdir(parents=True, exist_ok=True)
    vocab = folder / "vocab.json"
    vocab.write_text(json.dumps({s: i for i, s in enumerate(VITS_SYMBOLS)}, ensure_ascii=False), encoding="utf-8")
    tokenizer = VitsTokenizer(str(vocab), pad_token="_", unk_token="_", add_blank=True, phonemize=True)
    config = VitsConfig(vocab_size=len(VITS_SYMBOLS), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        ffn_dim=64, flow_size=32, upsample_initial_channel=32, num_speakers=num_speakers,
                        speaker_embedding_size=16, prior_encoder_num_flows=2, prior_encoder_num_wavenet_layers=2,
                        posterior_encoder_num_wavenet_layers=2, duration_predictor_num_flows=2,
                        duration_predictor_filter_channels=32, depth_separable_channels=2,
                        resblock_kernel_sizes=[3], resblock_dilation_sizes=[[1, 3]], sampling_rate=SR)
    torch.manual_seed(0)
    model = VitsModel(config).eval()
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return model


def _training_checkpoint(model, folder):
    """What accelerator.save_state writes mid-training: weight norm still on (both PyTorch
    layouts), generator and discriminator in separate files, no config.json."""
    import copy
    from safetensors.torch import save_file
    from torch.nn.utils import weight_norm
    from torch.nn.utils.parametrizations import weight_norm as parametrized_weight_norm

    net = copy.deepcopy(model)
    for i, layer in enumerate(net.decoder.upsampler):
        (parametrized_weight_norm if i % 2 else weight_norm)(layer)
    for flow in net.flow.flows:
        weight_norm(flow.conv_pre)
        parametrized_weight_norm(flow.conv_post)
    folder.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in net.state_dict().items()}, folder / "model.safetensors")
    import torch
    save_file({"discriminator.layer.weight": torch.zeros(3)}, folder / "model_1.safetensors")


@needs_transformers
def test_hf_synthesis_is_seeded_and_reads_training_checkpoints(tmp_path):
    from scripts.evaluation.synthesizer import HFVITSSynthesizer, load_synthesizer, resolve_hf_model

    model = _tiny_hf_model(tmp_path / "run")
    synth = HFVITSSynthesizer(tmp_path / "run", speaker_id=1)
    a1, info = synth.synthesize("Hello world, this is a test.", seed=7)
    a2, _ = synth.synthesize("Hello world, this is a test.", seed=7)
    a3, _ = synth.synthesize("Hello world, this is a test.", seed=8)
    assert a1.ndim == 1 and a1.size > 0 and info["n_tokens"] > 0
    assert np.array_equal(a1, a2) and not np.array_equal(a1, a3)
    assert synth.speaker_table().shape == (4, 16) and synth.frontend == "hf"
    assert "three hundred" in synth.prepare_text("I have 300 cats.")  # digits spelled out, not dropped

    _training_checkpoint(model, tmp_path / "run" / "checkpoint-1500")
    assert resolve_hf_model(tmp_path / "run" / "checkpoint-1500")[1].name == "model.safetensors"
    from_checkpoint = load_synthesizer(model=tmp_path / "run" / "checkpoint-1500", speaker_id=1)
    b1, _ = from_checkpoint.synthesize("Hello world, this is a test.", seed=7)
    assert b1.shape == a1.shape and np.allclose(a1, b1, atol=1e-3)  # weight norm folded back exactly

    with pytest.raises(ValueError, match="multi-speaker"):
        HFVITSSynthesizer(tmp_path / "run", speaker_id=None)
    with pytest.raises(ValueError, match="between 0 and"):
        HFVITSSynthesizer(tmp_path / "run", speaker_id=4)
    with pytest.raises(ValueError, match="not both"):
        load_synthesizer(model=tmp_path / "run", config="x.json", speaker_id=1)


class _FakeEncoder:  # spectral-shape "embedding": no download needed
    def __init__(self, device):
        pass

    def embed(self, audio_16k):
        spectrum = np.abs(np.fft.rfft(audio_16k, n=512))[:64]
        return spectrum / (np.linalg.norm(spectrum) + 1e-12)


@needs_transformers
def test_rank_speakers_script_end_to_end_and_resume(tmp_path, monkeypatch):
    from scripts import rank_speakers

    data_root, split_dir = _fake_dataset(tmp_path)
    _tiny_hf_model(tmp_path / "vctk", num_speakers=109)  # same table size as the VCTK model
    monkeypatch.setattr(rank_speakers, "SpeakerEncoder", _FakeEncoder)
    out = tmp_path / "ranking"
    argv = ["--model", str(tmp_path / "vctk"), "--speakers", "105-108", "--n_utts", "2", "--n_boot", "50",
            "--manifest", str(split_dir / "train_manifest.csv"), "--data_root", str(data_root),
            "--device", "cpu", "--f0_utts", "1", "--mcd_utts", "2", "--out_dir", str(out)]
    result = rank_speakers.main(argv)
    table, summary = result["table"], result["summary"]
    for name in ("speaker_ranking.csv", "summary.json", "speaker_ranking.png", "embeddings.npz"):
        assert (out / name).is_file(), name
    assert sorted(table["speaker_index"]) == [105, 106, 107, 108]
    assert summary["excluded_untrained"] == [108] and table.iloc[-1]["speaker_index"] == 108
    assert table[table["eligible"]]["rank"].tolist() == [1, 2, 3]
    assert set(summary["conditions"]) == {"closest", "median", "furthest"}
    assert table.loc[table["speaker_index"] == 105, "vctk_id"].item() == "p257"
    wavs = sorted(p.name for p in (out / "wavs").glob("*.wav"))
    assert wavs[-1].startswith("real_") and {"closest", "median", "furthest"} <= {w.split("_spk")[0] for w in wavs}
    assert table.loc[table["eligible"], "mcd_db"].notna().all()  # MCD cross-check on the first 2 sentences
    assert {"baselines", "pitch_groups", "agreement"} <= set(summary)
    assert summary["settings"]["sample_rate"] == SR

    def fail_if_called(*args, **kwargs):  # only finished sentences: --resume must not redo any
        raise AssertionError("--resume processed a finished sentence again")

    monkeypatch.setattr(rank_speakers, "f0_median", fail_if_called)
    resumed = rank_speakers.main(argv + ["--resume"])
    pd.testing.assert_series_equal(resumed["table"]["similarity"], table["similarity"])

    # a run saved before MCD existed still re-analyses (MCD just stays empty)
    saved = dict(np.load(out / "embeddings.npz"))
    del saved["mcd"]
    np.savez(out / "embeddings.npz", **saved)
    old = rank_speakers.main(argv + ["--resume"])
    assert old["table"]["mcd_db"].isna().all() and np.isnan(old["summary"]["agreement"]["ecapa_vs_mcd_spearman"])


@needs_transformers
def test_rank_speakers_extends_a_finished_run(tmp_path, monkeypatch):
    """--resume with a larger --n_utts keeps the finished sentences and only adds new ones."""
    from scripts import rank_speakers

    data_root, split_dir = _fake_dataset(tmp_path, n=8)  # 4 training sentences
    _tiny_hf_model(tmp_path / "model", num_speakers=3)
    calls = []

    class CountingEncoder(_FakeEncoder):
        def embed(self, audio_16k):
            calls.append(1)
            return super().embed(audio_16k)

    monkeypatch.setattr(rank_speakers, "SpeakerEncoder", CountingEncoder)
    argv = ["--model", str(tmp_path / "model"), "--manifest", str(split_dir / "train_manifest.csv"),
            "--data_root", str(data_root), "--device", "cpu", "--n_boot", "20", "--f0_utts", "1",
            "--mcd_utts", "1", "--out_dir", str(tmp_path / "out")]
    rank_speakers.main(argv + ["--n_utts", "2"])
    calls.clear()
    extended = rank_speakers.main(argv + ["--n_utts", "3", "--resume"])
    assert len(calls) == 1 + 3  # only the new sentence: its real clip + 3 speakers
    assert extended["summary"]["n_utts"] == 3 and extended["table"]["n_utts"].eq(3).all()


@needs_transformers
def test_evaluate_runs_a_hf_training_checkpoint(tmp_path):
    from scripts import evaluate

    data_root, split_dir = _fake_dataset(tmp_path)
    model = _tiny_hf_model(tmp_path / "runs" / "expA")
    _training_checkpoint(model, tmp_path / "runs" / "expA" / "checkpoint-1500")
    out = tmp_path / "results" / "expA_1500"
    summary = evaluate.main(["--exp", "expA", "--model", str(tmp_path / "runs" / "expA" / "checkpoint-1500"),
                             "--speaker_id", "2", "--manifest", str(split_dir / "test_manifest.csv"),
                             "--data_root", str(data_root), "--device", "cpu", "--out_dir", str(out),
                             "--metrics", "mcd", "--limit", "2"])
    assert summary["step"] == 1500 and summary["n_utterances"] == 2 and summary["settings"]["frontend"] == "hf"
    assert summary["metrics"]["mcd_dtw"]["n"] == 2


@needs_vits
def test_repo_frontend_gives_the_same_token_ids_as_official_vits():
    """With the corrected symbol table, TextProcessor reproduces the official VITS front-end
    (english_cleaners2 + espeak + blanks) token for token on ordinary sentences, i.e. the
    pretrained checkpoint receives exactly the input format it was trained on."""
    pytest.importorskip("unidecode")  # the official cleaners import it
    import importlib
    from scripts.TextProcessor import TextProcessor
    from scripts.evaluation.synthesizer import _use_bundled_espeak, import_vits

    import_vits(_vits_dir())
    _use_bundled_espeak()
    official_text = importlib.import_module("text")
    commons = importlib.import_module("commons")
    processor = TextProcessor(add_blank=True)
    for sentence in ("The birch canoe slid on the smooth planks.",
                     "It was the appropriate size to contain, in disassembled form,",
                     "Did you really expect the report to be finished before the end of the week?"):
        official = commons.intersperse(official_text.text_to_sequence(sentence, ["english_cleaners2"]), 0)
        assert processor.text_to_sequence(sentence).tolist() == official, sentence


@needs_vits
def test_evaluate_end_to_end_and_timing_only_run(tmp_path):
    from scripts import evaluate

    data_root, split_dir = _fake_dataset(tmp_path)
    config, ckpt = _random_checkpoint(tmp_path, "ckpt/G_1500.pth")
    out = tmp_path / "results" / "expA"
    common = ["--exp", "expA", "--config", str(config), "--checkpoint", str(ckpt), "--speaker_id", "0",
              "--vits_dir", str(_vits_dir()), "--manifest", str(split_dir / "test_manifest.csv"),
              "--data_root", str(data_root), "--device", "cpu", "--out_dir", str(out),
              "--n_warmup", "1", "--n_repeats", "1", "--rtf_texts", "test"]
    summary = evaluate.main(common + ["--metrics", "mcd,rtf"])
    assert summary["step"] == 1500 and summary["n_utterances"] == 3
    assert summary["metrics"]["mcd_dtw"]["n"] == 3 and "cpu" in summary["latency"]
    for name in ("per_utterance.csv", "summary.json", "latency_cpu.json", "rtf_runs_cpu.csv"):
        assert (out / name).is_file(), name
    assert len(list((out / "wavs").glob("*.wav"))) == 3

    before = (out / "summary.json").read_text()
    evaluate.main(common + ["--metrics", "rtf", "--cpu_threads", "1"])  # timing-only must not touch summary.json
    assert (out / "summary.json").read_text() == before
