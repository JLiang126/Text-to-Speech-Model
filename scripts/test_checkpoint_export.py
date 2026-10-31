import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoTokenizer, VitsModel as InferenceModel, VitsTokenizer

from scripts.configuration_vits import VitsConfig
from scripts.model import VitsDiscriminator, VitsModelForPreTraining
from scripts.train_model import save_inference_checkpoint


class CheckpointExportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        vocabulary = {"_": 0, "a": 1, "b": 2, "c": 3, " ": 4, "<unk>": 5}
        vocabulary_path = self.directory / "vocab.json"
        vocabulary_path.write_text(json.dumps(vocabulary), encoding="utf-8")
        self.tokenizer = VitsTokenizer(
            vocab_file=str(vocabulary_path), phonemize=False,
            normalize=False, add_blank=False,
        )
        config = VitsConfig(
            vocab_size=len(vocabulary), hidden_size=4, flow_size=4,
            num_hidden_layers=1, num_attention_heads=2, ffn_dim=8,
            hidden_dropout=0.0, attention_dropout=0.0, activation_dropout=0.0,
            layerdrop=0.0, num_speakers=3, speaker_embedding_size=4,
            use_stochastic_duration_prediction=False,
            duration_predictor_filter_channels=4, duration_predictor_dropout=0.0,
            upsample_initial_channel=16, upsample_rates=[2, 2],
            upsample_kernel_sizes=[4, 4], resblock_kernel_sizes=[3],
            resblock_dilation_sizes=[[1, 3, 5]],
            prior_encoder_num_flows=1, prior_encoder_num_wavenet_layers=1,
            posterior_encoder_num_wavenet_layers=1, wavenet_kernel_size=3,
            spectrogram_bins=3, segment_size=16, hop_length=4,
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
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        loss = sum(parameter.square().mean() for parameter in self.model.parameters())
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=False)

    def snapshot(self):
        return {
            "weights": {name: value.clone() for name, value in self.model.state_dict().items()},
            "parameters": {
                name: (id(parameter), parameter.requires_grad)
                for name, parameter in self.model.named_parameters()
            },
            "gradients": {
                name: None if parameter.grad is None else parameter.grad.clone()
                for name, parameter in self.model.named_parameters()
            },
            "optimizer_parameters": [
                id(parameter)
                for group in self.optimizer.param_groups for parameter in group["params"]
            ],
            "optimizer_state": deepcopy(self.optimizer.state_dict()),
            "training": self.model.training,
            "config": self.model.config.to_dict(),
            "rng": torch.random.get_rng_state().clone(),
        }

    def assert_training_unchanged(self, before):
        self.assertEqual(self.model.training, before["training"])
        self.assertEqual(self.model.config.to_dict(), before["config"])
        self.assertEqual(tuple(self.model.state_dict()), tuple(before["weights"]))
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before["weights"][name], rtol=0, atol=0)
        self.assertEqual({
            name: (id(parameter), parameter.requires_grad)
            for name, parameter in self.model.named_parameters()
        }, before["parameters"])
        for name, parameter in self.model.named_parameters():
            if before["gradients"][name] is None:
                self.assertIsNone(parameter.grad)
            else:
                torch.testing.assert_close(
                    parameter.grad, before["gradients"][name], rtol=0, atol=0,
                )
        self.assertEqual([
            id(parameter)
            for group in self.optimizer.param_groups for parameter in group["params"]
        ], before["optimizer_parameters"])
        torch.testing.assert_close(
            self.optimizer.state_dict(), before["optimizer_state"], rtol=0, atol=0,
        )
        torch.testing.assert_close(torch.random.get_rng_state(), before["rng"], rtol=0, atol=0)

    def load_inference_model(self, output_dir, variant):
        restored, info = InferenceModel.from_pretrained(
            output_dir, variant=variant, local_files_only=True, output_loading_info=True,
        )
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            self.assertFalse(info[name], info)
        restored_tokenizer = AutoTokenizer.from_pretrained(output_dir, local_files_only=True)
        self.assertEqual(restored_tokenizer.get_vocab(), self.tokenizer.get_vocab())
        return restored

    def test_epoch_and_final_exports_preserve_training_and_latest_weights(self):
        for variant in ("Epoch_1", "experiment"):
            with self.subTest(variant=variant):
                output_dir = self.directory / variant
                before = self.snapshot()
                save_inference_checkpoint(self.model, self.tokenizer, output_dir, variant)
                self.assert_training_unchanged(before)
                restored = self.load_inference_model(output_dir, variant)
                for name, value in restored.state_dict().items():
                    if name in before["weights"]:
                        torch.testing.assert_close(value, before["weights"][name], rtol=0, atol=0)
                for source, target in zip(self.model.decoder.upsampler, restored.decoder.upsampler):
                    expected = torch._weight_norm(source.weight_v, source.weight_g, 0)
                    torch.testing.assert_close(target.weight, expected, rtol=0, atol=0)
                for source, target in zip(self.model.flow.flows, restored.flow.flows):
                    for layer_name in ("conv_pre", "conv_post"):
                        layer = getattr(source, layer_name)
                        expected = torch._weight_norm(layer.weight_v, layer.weight_g, 0)
                        torch.testing.assert_close(
                            getattr(target, layer_name).weight, expected, rtol=0, atol=0,
                        )
        weight = self.model.embed_speaker.weight.detach().clone()
        self.model.embed_speaker.weight.square().sum().backward()
        self.optimizer.step()
        self.assertFalse(torch.equal(self.model.embed_speaker.weight, weight))

    def test_resized_speaker_exports_match_synthesis(self):
        self.model.resize_speaker_embeddings(4, pad_to_multiple_of=None)
        self.model.requires_grad_(False)
        self.model.embed_speaker.requires_grad_(True)
        before = self.snapshot()
        output_dir = self.directory / "new-speaker"
        save_inference_checkpoint(self.model, self.tokenizer, output_dir, "Epoch_1")
        self.assert_training_unchanged(before)
        restored = self.load_inference_model(output_dir, "Epoch_1")
        self.assertEqual(restored.config.num_speakers, 4)
        torch.testing.assert_close(restored.embed_speaker.weight, self.model.embed_speaker.weight)
        self.model.eval()
        inputs = self.tokenizer("abc", return_tensors="pt")
        with torch.inference_mode():
            torch.manual_seed(42)
            expected = self.model(**inputs, speaker_id=3)
            torch.manual_seed(42)
            actual = restored(**inputs, speaker_id=3)
        torch.testing.assert_close(actual.sequence_lengths, expected.sequence_lengths)
        torch.testing.assert_close(actual.waveform, expected.waveform, rtol=1e-5, atol=1e-6)

    def test_optional_discriminator_exports_without_modifying_live_discriminator(self):
        self.discriminator.requires_grad_(False)
        before = self.snapshot()
        discriminator_weights = {
            name: value.clone() for name, value in self.discriminator.state_dict().items()
        }
        discriminator_parameters = {
            name: (id(parameter), parameter.requires_grad)
            for name, parameter in self.discriminator.named_parameters()
        }
        output_dir = self.directory / "with-discriminator"
        save_inference_checkpoint(
            self.model, self.tokenizer, output_dir, "experiment",
            discriminator=self.discriminator,
        )
        self.assert_training_unchanged(before)
        self.assertFalse(hasattr(self.model, "discriminator"))
        self.assertTrue(self.discriminator.training)
        self.assertEqual({
            name: (id(parameter), parameter.requires_grad)
            for name, parameter in self.discriminator.named_parameters()
        }, discriminator_parameters)
        for name, value in self.discriminator.state_dict().items():
            torch.testing.assert_close(value, discriminator_weights[name], rtol=0, atol=0)
        restored, info = VitsModelForPreTraining.from_pretrained(
            output_dir, variant="experiment", local_files_only=True, output_loading_info=True,
        )
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            self.assertFalse(info[name], info)
        with torch.inference_mode():
            audio = torch.randn(1, 1, 64)
            expected_scores, expected_maps = self.discriminator(audio)
            actual_scores, actual_maps = restored.discriminator(audio)
        torch.testing.assert_close(actual_scores, expected_scores)
        torch.testing.assert_close(actual_maps, expected_maps)

    def test_save_failures_propagate_without_changing_training_state(self):
        before = self.snapshot()
        with patch("scripts.train_model.VitsModel.save_pretrained", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                save_inference_checkpoint(
                    self.model, self.tokenizer, self.directory / "failed", "Epoch_1",
                )
        self.assert_training_unchanged(before)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_gpu_export_preserves_device_and_cuda_rng(self):
        self.model.to("cuda:0")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.optimizer.zero_grad()
        self.model.embed_speaker.weight.square().sum().backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        before = self.snapshot()
        cuda_rng = torch.cuda.get_rng_state(0).clone()
        output_dir = self.directory / "gpu"
        save_inference_checkpoint(self.model, self.tokenizer, output_dir, "Epoch_1")
        self.assert_training_unchanged(before)
        torch.testing.assert_close(torch.cuda.get_rng_state(0), cuda_rng, rtol=0, atol=0)
        self.assertTrue(all(parameter.device == torch.device("cuda:0") for parameter in self.model.parameters()))
        restored = self.load_inference_model(output_dir, "Epoch_1")
        torch.testing.assert_close(
            restored.embed_speaker.weight, self.model.embed_speaker.weight.cpu(), rtol=0, atol=0,
        )


if __name__ == "__main__":
    unittest.main()
