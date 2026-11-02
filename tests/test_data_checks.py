"""Data checks (A2 feedback), the held-out guard and the training-log readers."""
import argparse
import json

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import data_checks
from scripts.evaluation import audio, training_logs
from scripts.evaluation.manifest import overlap_with, split_fingerprint

SR = 22050


def _clip(seconds, seed):
    rng = np.random.default_rng(seed)
    return (0.3 * np.sin(2 * np.pi * 180 * np.arange(int(seconds * SR)) / SR)
            + 0.01 * rng.standard_normal(int(seconds * SR))).astype(np.float32)


def _dataset(tmp_path, n=10):
    root = tmp_path / "LJSpeech-1.1"
    (root / "wavs").mkdir(parents=True)
    rows, lines = [], []
    for i in range(n):
        utt = f"LJ001-{i:04d}"
        audio.save_wav(root / "wavs" / f"{utt}.wav", _clip(1.2 + 0.1 * i, i), SR)
        text = f"Sentence number {i + 1}."
        lines.append(f"{utt}|{text}|{text}")
        rows.append({"wav_path": str(root / "wavs" / f"{utt}.wav"), "utt_id": utt, "raw_text": text,
                     "norm_text": text, "clean_text": text})
    (root / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    df = pd.DataFrame(rows)
    return root, {"train": df.iloc[:8].reset_index(drop=True), "val": df.iloc[8:9].reset_index(drop=True),
                  "test": df.iloc[9:].reset_index(drop=True)}


def test_split_checks_pass_on_a_clean_split_and_catch_each_problem(tmp_path):
    root, splits = _dataset(tmp_path)
    assert data_checks.check_metadata_coverage(root)["n_metadata"] == 10
    ids = data_checks.check_split_ids(splits)
    assert ids["fingerprints"]["test"] == split_fingerprint(["LJ001-0009"])
    assert data_checks.check_split_files(splits, SR)["n_files"] == 10
    shares = data_checks.check_split_proportions(splits, (0.8, 0.1, 0.1))
    assert shares["train"] == pytest.approx(0.8)

    leaked = dict(splits, test=pd.concat([splits["test"], splits["train"].iloc[:1]], ignore_index=True))
    with pytest.raises(AssertionError, match="in both train and test"):
        data_checks.check_split_ids(leaked)
    repeated = dict(splits, val=pd.concat([splits["val"], splits["val"]], ignore_index=True))
    with pytest.raises(AssertionError, match="repeated utterance IDs"):
        data_checks.check_split_ids(repeated)
    with pytest.raises(AssertionError, match="configured"):
        data_checks.check_split_proportions(splits, (0.9, 0.05, 0.05))
    with pytest.raises(AssertionError, match="Hz"):
        data_checks.check_split_files(splits, 16000)
    (root / "wavs" / "LJ001-0003.wav").unlink()
    with pytest.raises(AssertionError, match="missing"):
        data_checks.check_split_files(splits, SR)
    with pytest.raises(AssertionError, match="no wav file"):
        data_checks.check_metadata_coverage(root)


def _batch(hop=256):
    """A TTSCollate-shaped batch: 3 items sorted by length, zero padding after each length."""
    spec_lengths, text_lengths = [6, 5, 3], [5, 4, 3]
    b, t_spec, t_text = 3, 6, 5
    ids = torch.zeros(b, t_text, dtype=torch.long)
    linear, mel = torch.zeros(b, 513, t_spec), torch.zeros(b, 80, t_spec)
    wav = torch.zeros(b, 1, t_spec * hop)
    for i, (s, t) in enumerate(zip(spec_lengths, text_lengths)):
        ids[i, :t] = torch.arange(1, t + 1)
        linear[i, :, :s], mel[i, :, :s] = 0.5, -4.0
        wav[i, 0, : s * hop] = torch.linspace(-0.9, 0.9, s * hop)
    arange = torch.arange(t_spec)
    return {"utt_ids": ["a", "b", "c"], "norm_texts": ["x"] * 3,
            "model_inputs": {"input_ids": ids, "attention_mask": torch.arange(t_text)[None] < torch.tensor(text_lengths)[:, None],
                             "labels": linear, "labels_attention_mask": arange[None] < torch.tensor(spec_lengths)[:, None],
                             "speaker_id": 0},
            "mel_padded": mel, "wav_padded": wav, "wav_lengths": torch.tensor(spec_lengths) * hop}


def test_batch_check_covers_every_item_not_just_the_first():
    stats = data_checks.check_batch(_batch(), vocab_size=178, hop_length=256)
    assert stats["items"] == 3 and stats["wav_max"] <= 1.0

    bad = _batch()
    bad["wav_padded"][2, 0, -1] = 0.2  # padding of the LAST item, which a first-item check never sees
    with pytest.raises(AssertionError, match="c: non-zero waveform padding"):
        data_checks.check_batch(bad, vocab_size=178)
    bad = _batch()
    bad["wav_padded"][1, 0, 10] = 1.5
    with pytest.raises(AssertionError, match=r"outside \[-1, 1\]"):
        data_checks.check_batch(bad, vocab_size=178)
    bad = _batch()
    bad["model_inputs"]["input_ids"][2, 0] = 178
    with pytest.raises(AssertionError, match="token IDs outside"):
        data_checks.check_batch(bad, vocab_size=178)
    bad = _batch()
    bad["model_inputs"]["labels"][1, 0, 2] = float("nan")
    with pytest.raises(AssertionError, match="non-finite"):
        data_checks.check_batch(bad, vocab_size=178)
    with pytest.raises(AssertionError, match="frames for"):
        data_checks.check_batch(_batch(hop=256), vocab_size=178, hop_length=200)


def test_evaluation_overlap_counts_sentences_per_split(tmp_path):
    _, splits = _dataset(tmp_path)
    overlap = data_checks.evaluation_overlap(["LJ001-0000", "LJ001-0009", "LJ999-0000"], splits)
    assert overlap == {"evaluated": 3, "train": 1, "val": 0, "test": 1}


def test_held_out_guard_refuses_sentences_a_model_trained_on(tmp_path):
    from scripts import evaluate

    _, splits = _dataset(tmp_path)
    folder = tmp_path / "splits"
    folder.mkdir()
    for name, df in splits.items():
        df.to_csv(folder / f"{name}_manifest.csv", index=False)
    test = splits["test"]

    def args(**kw):
        base = {"manifest": str(folder / "test_manifest.csv"), "train_manifest": None, "allow_overlap": False}
        return argparse.Namespace(**{**base, **kw})

    info = evaluate.describe_test_set(args(), test, is_model=True)  # default: the manifests beside --manifest
    assert info["n"] == 1 and info["fingerprint"] == split_fingerprint(test["utt_id"])
    assert sorted(info["overlap"].values()) == [0, 0]

    other_run = tmp_path / "run_train.csv"  # a run trained on a split that contains the test sentence
    pd.concat([splits["train"], test]).to_csv(other_run, index=False)
    assert overlap_with(test["utt_id"], other_run) == 1
    with pytest.raises(SystemExit, match="Not a held-out test: 1 of the 1"):
        evaluate.describe_test_set(args(train_manifest=[str(other_run)]), test, is_model=True)
    allowed = evaluate.describe_test_set(args(train_manifest=[str(other_run)], allow_overlap=True), test, is_model=True)
    assert allowed["overlap"][str(other_run)] == 1
    evaluate.describe_test_set(args(train_manifest=[str(other_run)]), test, is_model=False)  # real recordings: no guard


def _history(folder, mel, offset=0.0):
    (folder / "run").mkdir(parents=True)
    epochs = np.arange(1, len(mel) + 1)
    val = pd.DataFrame({"epoch": epochs, "mel": mel, "kl": np.linspace(2, 1, len(mel)),
                        "duration": np.linspace(1.5, 1.2, len(mel)), "fmaps": 1.0, "gen": 2.0, "disc": 2.5,
                        "generator_total": 50.0})
    train = val.assign(mel=np.asarray(mel) - 0.05 - offset)
    train.to_csv(folder / "run" / training_logs.TRAINING_HISTORY_CSV, index=False)
    val.to_csv(folder / "run" / training_logs.VALIDATION_HISTORY_CSV, index=False)


def test_training_logs_table_curves_and_settings(tmp_path):
    runs = tmp_path / "outputs"
    _history(runs / "slow", [0.60, 0.55, 0.50, 0.46, 0.42])         # still improving at the end
    _history(runs / "renamed_dir", [0.50, 0.45, 0.44, 0.445, 0.45])  # best at epoch 3, then flat/up
    config = tmp_path / "fast.yaml"
    config.write_text("output_dir: outputs/renamed_dir/\nlearning_rate: 0.00004\nspeaker_embedding_lr_ratio: 50.0\n")
    (runs / "renamed_dir" / "run" / "training_config.json").write_text(
        json.dumps({"learning_rate": 4e-5, "speaker_embedding_lr_ratio": 1.0, "freeze_mode": "none"}))

    assert training_logs.find_run_folder("fast", config, runs) == runs / "renamed_dir"  # via the config's output_dir
    assert training_logs.find_run_folder("missing", None, runs) is None
    histories = training_logs.load_histories({"slow": None, "fast": config, "missing": None}, runs)
    assert list(histories) == ["slow", "fast"]
    table = training_logs.loss_table(histories).set_index("experiment")
    assert table.loc["slow", "mel_slope_last3"] < 0 < table.loc["fast", "mel_slope_last3"]
    assert table.loc["fast", "mel_best_epoch"] == 3 and table.loc["slow", "mel_min"] == pytest.approx(0.42)
    assert table.loc["slow", "mel_gap_final"] == pytest.approx(0.05)
    assert training_logs.plot_loss_curves(histories, tmp_path / "curves.png", title="LR").is_file()

    settings = training_logs.run_settings(runs / "renamed_dir")
    claimed = {"learning_rate": "4e-5", "speaker_embedding_lr_ratio": 50.0, "freeze_mode": "none"}
    assert training_logs.settings_mismatches(settings, claimed, ("learning_rate", "speaker_embedding_lr_ratio",
                                                                 "freeze_mode")) == {
        "speaker_embedding_lr_ratio": (1.0, 50.0)}
