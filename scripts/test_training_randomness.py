import random
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.DataConfig import DataConfig
from scripts.DataLoader import TTSCollate
from scripts.AudioProcessor import AudioProcessor
from scripts.configuration_vits import VitsConfig
from scripts.model import VitsModelForPreTraining, monotonic_align_max_path, slice_segments
from scripts.train_model import (
    evaluate,
    generator_losses,
    prepare_batch,
    reconstruction_losses,
    save_sample_sounds_each_epoch,
)
from scripts.plots import LOSS_NAMES


class TrainingRandomnessTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.device = torch.device("cpu")
        self.cfg = DataConfig(
            output_dir=self.directory / "run", target_sr=16000,
            n_fft=16, win_length=16, hop_length=4, n_mels=4,
        )
        config = VitsConfig(
            vocab_size=6, hidden_size=4, flow_size=4,
            num_hidden_layers=1, num_attention_heads=2, ffn_dim=8,
            layerdrop=0.5, num_speakers=3, speaker_embedding_size=4,
            use_stochastic_duration_prediction=False,
            duration_predictor_filter_channels=4,
            upsample_initial_channel=16, upsample_rates=[2, 2],
            upsample_kernel_sizes=[4, 4], resblock_kernel_sizes=[3],
            resblock_dilation_sizes=[[1, 3, 5]],
            prior_encoder_num_flows=1, prior_encoder_num_wavenet_layers=1,
            posterior_encoder_num_wavenet_layers=1, wavenet_kernel_size=3,
            spectrogram_bins=9, segment_size=16, hop_length=4,
            discriminator_scale_channels=None,
            discriminator_period_channels=[1, 2], discriminator_periods=[2],
        )
        self.model = VitsModelForPreTraining(config)
        self.discriminator = self.model.discriminator
        del self.model.discriminator
        self.model.decoder.apply_weight_norm()
        for flow in self.model.flow.flows:
            torch.nn.utils.weight_norm(flow.conv_pre)
            torch.nn.utils.weight_norm(flow.conv_post)
        self.discriminator.apply_weight_norm()
        self.model.requires_grad_(False)
        self.model.embed_speaker.requires_grad_(True)
        self.discriminator.requires_grad_(False)
        self.audio_processor = AudioProcessor(self.cfg)
        self.samples = []
        for index in range(2):
            waveform = torch.sin(torch.arange(64).float() * 0.3 + index).unsqueeze(0)
            spec, mel = self.audio_processor.compute_spectrograms(waveform)
            self.samples.append({
                "token_ids": torch.tensor([1, 2, 3]), "waveform": waveform,
                "linear_spec": spec, "log_mel": mel,
                "utt_id": str(index), "norm_text": "abc",
            })
        self.loader = DataLoader(self.samples, batch_size=1, collate_fn=self.collate)
        self.accelerator = SimpleNamespace(
            device=self.device, gather_for_metrics=lambda values: values,
        )

    def collate(self, samples):
        batch = TTSCollate()(samples)
        return {
            name: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for name, value in batch.items()
        }

    def snapshot_rng(self):
        return (
            random.getstate(), np.random.get_state(),
            torch.random.get_rng_state().clone(),
            torch.cuda.get_rng_state(self.device).clone() if self.device.type == "cuda" else None,
        )

    def assert_rng_unchanged(self, before):
        self.assertEqual(random.getstate(), before[0])
        actual_numpy = np.random.get_state()
        self.assertEqual(actual_numpy[0], before[1][0])
        np.testing.assert_array_equal(actual_numpy[1], before[1][1])
        self.assertEqual(actual_numpy[2:], before[1][2:])
        torch.testing.assert_close(torch.random.get_rng_state(), before[2], rtol=0, atol=0)
        if before[3] is not None:
            torch.testing.assert_close(
                torch.cuda.get_rng_state(self.device), before[3], rtol=0, atol=0,
            )

    def run_evaluation(self):
        return evaluate(
            self.model, self.discriminator, self.loader, monotonic_align_max_path,
            self.audio_processor, 0, self.model.segment_size, self.accelerator, self.cfg,
        )

    def make_sample_saver(self):
        prompts = [
            {"input_ids": [1, 2, 3], "seed": 42 + index}
            for index in range(3)
        ]
        with patch.object(save_sample_sounds_each_epoch, "select_validation_prompts", return_value=prompts):
            saver = save_sample_sounds_each_epoch(None, self.cfg, 0)
        saver.save_folder.mkdir(parents=True, exist_ok=True)
        return saver

    def random_failure(self, **kwargs):
        random.random()
        np.random.random()
        torch.rand(1)
        if self.device.type == "cuda":
            torch.rand(1, device=self.device)
        raise RuntimeError("forward failed")

    def test_both_audio_import_orders_work_in_fresh_processes(self):
        for first, second in (
            ("AudioProcessor", "DataLoader"), ("DataLoader", "AudioProcessor"),
        ):
            with self.subTest(first=first):
                result = subprocess.run(
                    [sys.executable, "-c", f"import scripts.{first}; import scripts.{second}"],
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_evaluation_preserves_rng_metrics_and_prior_modes(self):
        for was_training in (True, False):
            with self.subTest(was_training=was_training):
                self.model.train(was_training)
                self.discriminator.train(not was_training)
                before = self.snapshot_rng()
                metrics = self.run_evaluation()
                self.assert_rng_unchanged(before)
                self.assertEqual(self.model.training, was_training)
                self.assertEqual(self.discriminator.training, not was_training)
                self.assertEqual(set(metrics), set(LOSS_NAMES))
                self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
                self.assertEqual(metrics, self.run_evaluation())
                self.assert_rng_unchanged(before)

    def test_evaluation_failure_restores_rng_and_modes(self):
        self.discriminator.eval()
        before = self.snapshot_rng()
        with patch.object(self.model, "forward", side_effect=self.random_failure):
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                self.run_evaluation()
        self.assert_rng_unchanged(before)
        self.assertTrue(self.model.training)
        self.assertFalse(self.discriminator.training)

    def train_step(self, optimizer):
        self.model.train()
        batch = self.collate(self.samples[:1])
        inputs = prepare_batch(batch, 0, self.model.segment_size)
        outputs = self.model(
            **inputs, monotonic_alignment_function=monotonic_align_max_path, return_dict=True,
        )
        mel, latent = reconstruction_losses(
            outputs, batch, self.audio_processor, self.model.segment_size,
        )
        target_audio = slice_segments(
            batch["wav_padded"], outputs.ids_slice * self.cfg.hop_length,
            self.model.config.segment_size,
        )
        _, real_maps = self.discriminator(target_audio)
        scores, fake_maps = self.discriminator(outputs.waveform)
        loss = generator_losses(
            outputs, real_maps, fake_maps, scores, mel, latent, self.cfg,
        )["generator_total"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

    def test_evaluation_frequency_does_not_change_speaker_training(self):
        initial_state = deepcopy(self.model.state_dict())
        expected_state = None
        expected_rng = None
        for evaluation_count in (0, 1, 3):
            with self.subTest(evaluation_count=evaluation_count):
                self.model.load_state_dict(initial_state, strict=True)
                optimizer = torch.optim.AdamW(self.model.embed_speaker.parameters(), lr=1e-3)
                random.seed(123)
                np.random.seed(123)
                torch.manual_seed(123)
                self.train_step(optimizer)
                for _ in range(evaluation_count):
                    self.run_evaluation()
                self.train_step(optimizer)
                if expected_state is None:
                    expected_state = deepcopy(self.model.state_dict())
                    expected_rng = self.snapshot_rng()
                    self.assertFalse(torch.equal(
                        self.model.embed_speaker.weight, initial_state["embed_speaker.weight"],
                    ))
                else:
                    for name, value in self.model.state_dict().items():
                        torch.testing.assert_close(value, expected_state[name], rtol=0, atol=0)
                    self.assert_rng_unchanged(expected_rng)

    def test_listening_samples_preserve_rng_and_reuse_prompt_seeds(self):
        saver = self.make_sample_saver()
        before = self.snapshot_rng()
        saver.save_epoch(self.model, 1)
        self.assert_rng_unchanged(before)
        self.assertTrue(self.model.training)
        saver.save_epoch(self.model, 2)
        self.assert_rng_unchanged(before)
        for index in range(1, 4):
            self.assertEqual(
                (saver.save_folder / f"epoch-1-sample-{index}.wav").read_bytes(),
                (saver.save_folder / f"epoch-2-sample-{index}.wav").read_bytes(),
            )

    def test_listening_failure_restores_rng_and_mode(self):
        saver = self.make_sample_saver()
        self.model.eval()
        before = self.snapshot_rng()
        with patch.object(self.model, "forward", side_effect=self.random_failure):
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                saver.save_epoch(self.model, 1)
        self.assert_rng_unchanged(before)
        self.assertFalse(self.model.training)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_gpu_evaluation_and_listening_preserve_all_rng_states(self):
        self.device = torch.device("cuda:0")
        self.model.to(self.device)
        self.discriminator.to(self.device)
        self.accelerator.device = torch.device("cuda")
        before = self.snapshot_rng()
        self.run_evaluation()
        self.assert_rng_unchanged(before)
        self.make_sample_saver().save_epoch(self.model, 1)
        self.assert_rng_unchanged(before)


if __name__ == "__main__":
    unittest.main()
