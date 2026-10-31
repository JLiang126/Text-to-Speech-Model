"""Fine-tune a VITS checkpoint on local LJSpeech data.

Run from the repository root:
    python -m scripts.train_model
    accelerate launch scripts/train_model.py

The default settings are loaded from configs/train.yaml. --config can select
another YAML object containing training settings. `--output-mode restart`
starts a fresh run, and `new` requires an empty output root. The default mode
is `restart`. Run files are written under `<output_dir>/run`, while listening
samples and intermediate checkpoints are written beside `run`.
The loader's existing split manifests are reused only when their settings
match. Training progress is reported in the console, with three fixed validation
phrases synthesized per epoch and their text and phonemes saved in listening_samples.json.
"""


import argparse
import json
import logging
import math
import os
import shutil
import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import time

# Support direct execution and the upstream configuration used by the local model.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "finetune-hf-vits-main"))

from scripts.DataConfig import DataConfig
from scripts.DataLoader import build_dataloaders
from scripts.TextProcessor import configure_word_mismatch_warnings
from scripts.AudioProcessor import AudioProcessor
from scripts.monotonic_alignment import MonotonicAlignment
from scripts.model import VitsModelForPreTraining, slice_segments

from scripts.plots import (
    LOSS_NAMES,
    plot_training_history,
    save_loss_history_csvs,
)

logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load training settings from a YAML object and resolve derived defaults."""
    contents = config_path.read_text(encoding="utf-8")
    settings = yaml.safe_load(contents)
    if not isinstance(settings, dict):
        raise ValueError(f"Training config must contain a YAML object: {config_path}")

    # A null split directory lets DataConfig derive it from output_dir.
    if settings.get("output_split_dir") is None:
        settings.pop("output_split_dir", None)
    # Preserve the DataConfig environment-variable default when using the bundled config.
    if config_path.resolve() == DEFAULT_CONFIG_PATH and os.environ.get("LJSPEECH_ROOT"):
        settings["data_root"] = os.environ["LJSPEECH_ROOT"]
    return settings


def parse_args():
    # Training settings live in YAML; keep config and output handling on the CLI.
    parser = argparse.ArgumentParser(description=__doc__)
    # the configuration YAML file has all the settings and inputs for training
    # path to the config file is a keyword argument (--config) on the command line
    # default is the bundled train.yaml file in the configs directory
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML file containing DataConfig fields (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--output-mode",
        choices=("restart", "new"),
        default="restart",
        help="restart the fixed run directory, or require a new output root.",
    )
    parser.add_argument(
        "--test-mode",
        choices=('yes', 'no'),
        default='no',
        help="enable or disable test mode (single epoch, reduced dataset).",
    )
    args = parser.parse_args()

    # Load YAML settings, then resolve paths derived from the configured data root.
    settings = load_config(args.config)
    cfg = DataConfig(**settings)

    # Apply test mode settings if enabled
    if args.test_mode == 'yes':
        cfg.num_train_epochs = 1
        cfg.subset_size = 10 * cfg.per_device_train_batch_size  # Reduce dataset size for quick testing

    # Accept both the dataset folder and its parent as the configured data root.
    cfg.data_root = Path(cfg.data_root)
    if not (cfg.data_root / "metadata.csv").exists() and (cfg.data_root / "LJSpeech-1.1").is_dir():
        cfg.data_root /= "LJSpeech-1.1"
    output_root = Path(cfg.output_dir)
    cfg.output_split_dir = Path(cfg.output_split_dir)
    split_dir_resolved = cfg.output_split_dir.resolve()
    data_root_resolved = cfg.data_root.resolve()
    if (
        split_dir_resolved == data_root_resolved
        or split_dir_resolved in data_root_resolved.parents
        or data_root_resolved in split_dir_resolved.parents
    ):
        parser.error("output_split_dir must not overlap data_root; the dataset directory is read-only.")
    run_dir = output_root / "run"

    # Fail before loading a model or dataset when a setting cannot be used.
    if cfg.num_train_epochs < 1 or cfg.per_device_train_batch_size < 1 or cfg.gradient_accumulation_steps < 1:
        parser.error("Epochs, batch size and gradient accumulation steps must be positive.")
    if cfg.learning_rate <= 0 or cfg.num_workers < 0:
        parser.error("Learning rate must be positive and num_workers must be nonnegative.")
    if cfg.mixed_precision not in ("no", "fp16", "bf16"):
        parser.error("mixed_precision must be one of: no, fp16, bf16.")
    if cfg.subset_size is not None and cfg.subset_size < 1:
        parser.error("subset_size must be positive or null.")
    if cfg.do_eval and (cfg.eval_steps < 1 or cfg.per_device_eval_batch_size < 1):
        parser.error("Evaluation interval and batch size must be positive.")
    if cfg.do_eval and cfg.max_eval_samples is not None and cfg.max_eval_samples < 1:
        parser.error("max_eval_samples must be positive or null.")
    if any(ratio < 0 for ratio in cfg.split_ratios) or not math.isclose(sum(cfg.split_ratios), 1.0):
        parser.error("split_ratios must be nonnegative and sum to one.")
    if len(cfg.split_ratios) != 3:
        parser.error("split_ratios must contain train, validation and test ratios.")
    cfg.output_dir = run_dir
    if args.output_mode == "new" and output_root.exists():
        if not output_root.is_dir() or any(output_root.iterdir()):
            parser.error(f"Output directory ({output_root}) is not empty.")
    return cfg, args


def load_data(cfg, tokenizer, num_workers):
    """
    Load the training and validation data according to the configuration.
    """
    # A manifest is only valid for the split settings that created it. This
    # prevents quietly training on stale splits after configuration changes.
    summary_path = Path(cfg.output_split_dir) / "split_summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())["config"]
        current = json.loads(json.dumps(cfg.to_dict()))
        split_fields = ("seed", "subset_size", "split_ratios", "min_duration_sec",
                        "max_duration_sec", "lowercase")
        changed = [name for name in split_fields if previous[name] != current[name]]
        if Path(previous["data_root"]).resolve() != Path(cfg.data_root).resolve():
            changed.append("data_root")
        if changed:
            raise ValueError(f"Existing split settings differ ({', '.join(changed)}). Set a new output_split_dir.")
    tokenizer.add_blank = cfg.add_blank
    # Pass the checkpoint tokenizer directly so dataset token IDs match the model.
    # The validation loader keeps its configured batch size and sample cap.
    loaders = build_dataloaders(
        cfg,
        batch_size=cfg.per_device_train_batch_size,
        num_workers=num_workers,
        tokenizer=tokenizer,
    )
    if not len(loaders["train"].dataset):
        raise ValueError("The training split is empty. Check duration limits, subset size and split ratios.")

    eval_loader = loaders["val"] if cfg.do_eval else None
    if eval_loader is not None and not len(eval_loader.dataset):
        raise ValueError("The validation split is empty. Increase the subset or validation ratio.")
    return loaders["train"], eval_loader, loaders["val"].dataset


def prepare_batch(batch, speaker_id, segment_frames):
    """Translate TTSCollate's padded tensors and lengths into model inputs."""
    text = batch["text_padded"]
    spectrogram = batch["spec_padded"]
    text_lengths = batch["text_lengths"]
    spec_lengths = batch["spec_lengths"]
    # These checks catch malformed text/audio pairs and avoid sending impossible
    # alignments or too-short recordings into the model.
    if torch.any(text_lengths < 1) or torch.any(spec_lengths < text_lengths):
        raise ValueError("Each recording must have at least one audio frame per text token.")
    if torch.any(spec_lengths < segment_frames):
        raise ValueError("A recording is shorter than the model's training segment.")
    # The collator pads samples to batch maxima; masks tell VITS which positions
    # are real tokens/frames and which positions are padding.
    text_mask = torch.arange(text.size(1), device=text.device)[None, :] < text_lengths[:, None]
    spec_mask = torch.arange(spectrogram.size(2), device=text.device)[None, :] < spec_lengths[:, None]
    speakers = None if speaker_id is None else torch.full(
        (text.size(0),), speaker_id, dtype=torch.long, device=text.device
    )
    return {
        "input_ids": text, "attention_mask": text_mask,
        "labels": spectrogram, "labels_attention_mask": spec_mask,
        "speaker_id": speakers,
    }


def discriminator_loss(real_outputs, generated_outputs):
    # Least-squares GAN objective: real audio should score near 1, generated
    # audio near 0. Sum across the discriminator's multiple output scales.
    return sum(
        ((1 - real.float()) ** 2).mean() + (generated.float() ** 2).mean()
        for real, generated in zip(real_outputs, generated_outputs)
    )


def kl_loss(outputs):
    # Compare posterior and prior latent distributions, excluding padded frames.
    prior_log = outputs.prior_log_variances.float()
    kl = prior_log - outputs.posterior_log_variances.float() - 0.5
    kl += 0.5 * (outputs.prior_latents.float() - outputs.prior_means.float()) ** 2 * torch.exp(-2 * prior_log)
    mask = outputs.labels_padding_mask
    return (kl * mask).sum() / mask.sum()


def reconstruction_losses(outputs, batch, audio_processor, segment_frames):
    # Align target and generated mel features to the model's sampled audio
    # segment, then pair reconstruction loss with the latent KL regularizer.
    target_mel = slice_segments(batch["mel_padded"], outputs.ids_slice, segment_frames)
    # Use the same audio processor for real and generated mel features.
    _, generated_mel = audio_processor.compute_spectrograms(outputs.waveform.float().squeeze(1))
    if generated_mel.ndim == 2:
        generated_mel = generated_mel.unsqueeze(0)
    return F.l1_loss(generated_mel, target_mel.float()), kl_loss(outputs)


def generator_loss(outputs, real_maps, generated_maps, generated_scores, mel_loss, latent_loss, cfg):
    return generator_losses(
        outputs, real_maps, generated_maps, generated_scores, mel_loss, latent_loss, cfg
    )["generator_total"]


def generator_losses(outputs, real_maps, generated_maps, generated_scores, mel_loss, latent_loss, cfg):
    # Combine duration, mel, KL, feature-matching, and adversarial objectives.
    # Config weights make the relative contribution of each term explicit.
    duration_loss = outputs.log_duration.float().sum()
    feature_loss = 2 * sum(
        F.l1_loss(generated.float(), real.detach().float())
        for real_layers, generated_layers in zip(real_maps, generated_maps)
        for real, generated in zip(real_layers, generated_layers)
    )
    adversarial_loss = sum(((1 - score.float()) ** 2).mean() for score in generated_scores)
    total = (
        cfg.weight_duration * duration_loss + cfg.weight_mel * mel_loss
        + cfg.weight_kl * latent_loss + cfg.weight_fmaps * feature_loss
        + cfg.weight_gen * adversarial_loss
    )
    return {
        "mel": mel_loss, "kl": latent_loss, "duration": duration_loss,
        "fmaps": feature_loss, "gen": adversarial_loss, "generator_total": total,
    }


def accumulate_losses(totals, losses, batch_size, accelerator):
    values = torch.stack([losses[name].detach().float() for name in LOSS_NAMES])
    if not torch.isfinite(values).all():
        raise FloatingPointError("Logged losses contain non-finite values.")
    metrics = accelerator.gather_for_metrics(values.repeat(batch_size, 1))
    totals[:-1] += metrics.sum(0)
    totals[-1] += metrics.size(0)


def averaged_losses(totals):
    if totals[-1] == 0:
        raise ValueError("Cannot average losses from an empty loader.")
    return {name: (totals[i] / totals[-1]).item() for i, name in enumerate(LOSS_NAMES)}


@torch.no_grad()
def evaluate(model, discriminator, loader, alignment, audio_processor, speaker_id, segment_frames, accelerator, cfg):
    """
    Do evaluation on the validation set and return averaged metrics.
    """
    # Weight batch objectives by sample count across distributed workers.
    model.eval()
    discriminator.eval()
    total = torch.zeros(len(LOSS_NAMES) + 1, device=accelerator.device)
    for batch in loader:
        inputs = prepare_batch(batch, speaker_id, segment_frames) # prepare the batch for model input
        outputs = model(**inputs, monotonic_alignment_function=alignment, return_dict=True) # forward pass through the model
        mel, latent = reconstruction_losses(outputs, batch, audio_processor, segment_frames) # compute reconstruction losses for mel and latent representations
        target_audio = slice_segments(
            batch["wav_padded"], outputs.ids_slice * cfg.hop_length, cfg.hop_length * segment_frames
        )
        real_scores, real_maps = discriminator(target_audio)
        fake_scores, fake_maps = discriminator(outputs.waveform)
        losses = generator_losses(outputs, real_maps, fake_maps, fake_scores, mel, latent, cfg)
        losses["disc"] = discriminator_loss(real_scores, fake_scores)
        accumulate_losses(total, losses, inputs["input_ids"].size(0), accelerator)
    # Restore training mode for the next optimization step.
    model.train()
    discriminator.train()
    return averaged_losses(total)



class save_sample_sounds_each_epoch:
    """
    Class for saving sample sounds for each epoch during training.
    """

    def __init__(self, dataset, cfg, speaker_id, save_folder='listening_samples'):
        """
        Initialize the save_sample_sounds_each_epoch instance.

        Args:
            dataset (torch.utils.data.Dataset): The dataset containing audio samples.
            cfg (DataConfig): The configuration object with training settings.
            speaker_id (int): The ID of the speaker for which to save samples.
        """
        self.dataset = dataset
        self.output_root = Path(cfg.output_dir).parent
        self.speaker_id = speaker_id
        self.sampling_rate = cfg.target_sr
        self.prompts = self.select_validation_prompts(dataset, cfg.seed)
        self.save_folder = self.output_root / save_folder

    @staticmethod
    def save_waveform(waveform, output_path, sampling_rate):
        """
        Save a waveform as a WAV file at the specified output path.

        Args:
            waveform (torch.Tensor): The audio waveform to save.
            output_path (Path): The path to save the WAV file.
            sampling_rate (int): The sampling rate of the audio.
        """
        
        waveform = torch.as_tensor(waveform).detach().float().cpu().squeeze()
        if waveform.ndim != 1:
            raise ValueError("Listening samples must be mono audio.")
        if waveform.numel() == 0 or not torch.isfinite(waveform).all():
            raise FloatingPointError("Listening sample is empty or non-finite.")
        pcm = (waveform.clamp(-1, 1).numpy() * 32767).astype(np.int16)
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sampling_rate)
            wav.writeframes(pcm.tobytes())

    @staticmethod
    def get_validation_audio(dataset, prompt):
        sample = dataset[prompt["validation_index"]]
        return sample["waveform"], sample["sample_rate"]

    @staticmethod
    def select_validation_prompts(dataset, seed, num_samples=3):
        if len(dataset) < num_samples:
            raise ValueError(f"At least {num_samples} validation examples are required for listening samples.")

        # pick random indices for validation samples
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:num_samples].tolist()
        records = []
        # loop through the selected indices and gather the corresponding records
        for position, index in enumerate(indices):
            source = dataset
            source_index = index
            while isinstance(source, torch.utils.data.Subset):
                source_index = source.indices[source_index]
                source = source.dataset
            row = source.df.iloc[source_index]
            input_ids = source._get_token_ids(source_index).tolist()
            if source.tokenizer is not None:
                phonemes = "".join(source.tokenizer.convert_ids_to_tokens(
                    input_ids, skip_special_tokens=True,
                ))
            else:
                phonemes = source.text_processor.sequence_to_text(input_ids)
            records.append({
                "validation_index": index,
                "utt_id": str(row["utt_id"]),
                "text": str(row["norm_text"]),
                "clean_text": str(row["clean_text"]),
                "phonemes": phonemes,
                "input_ids": input_ids,
                "seed": seed + position,
            })
        return records

    def save_reference_audio(self):
        """
        Save the actual reference audio for the validation prompts for comparison
        """
        directory = self.save_folder
        directory.mkdir(parents=True, exist_ok=True)
        for position, prompt in enumerate(self.prompts, start=1):
            audio, sample_rate = self.get_validation_audio(self.dataset, prompt)
            self.save_waveform(audio, directory / f"original_sample_{position}.wav", sample_rate)

    def save_prompt_metadata(self):
        """
        Save the metadata for the listening samples to a JSON file.
        """
        (self.save_folder / "listening_samples.json").write_text(json.dumps({
            "speaker_id": self.speaker_id,
            "sampling_rate": self.sampling_rate,
            "filename_pattern": "epoch-{epoch}-sample-{sample}.wav",
            "prompts": self.prompts,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    @torch.no_grad()
    def save_epoch(self, model, epoch):
        """
        Save validation samples
        """
        was_training = model.training # Remember the model's training state to restore it later.
        model.eval()
        device = next(model.parameters()).device
        devices = [device.index] if device.type == "cuda" else []
        try:
            for position, prompt in enumerate(self.prompts, start=1):
                tokens = torch.tensor([prompt["input_ids"]], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(tokens, dtype=torch.bool)
                with torch.random.fork_rng(devices=devices):
                    torch.random.default_generator.manual_seed(prompt["seed"])
                    if device.type == "cuda":
                        with torch.cuda.device(device):
                            torch.cuda.manual_seed(prompt["seed"])
                    outputs = model(
                        input_ids=tokens,
                        attention_mask=attention_mask,
                        speaker_id=self.speaker_id,
                    )
                waveform = outputs.waveform[0, :int(outputs.sequence_lengths[0].item())]
                self.save_waveform(
                    waveform,
                    self.save_folder / f"epoch-{epoch}-sample-{position}.wav",
                    self.sampling_rate,
                )
        finally:
            model.train(was_training)


def make_scheduler(optimizer, cfg, total_steps):
    """
    Create a learning rate scheduler for the optimizer based on the configuration.

    The warmup is a linear increase of the learning rate from 0 to the initial value over
    the specified warmup steps.

    If a linear decay schedule is used, the learning rate will increase linearly from 0 to
    the initial value during warmup and then decrease linearly to zero.
    """

    if not cfg.use_linear_lr_schedule:
        return None

    warmup_steps = math.ceil(total_steps * cfg.warmup_ratio)

    def scale(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def main():
    # Configure distributed training and seed Python/NumPy/PyTorch before
    # constructing anything that may use randomness.
    cfg, args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    configure_word_mismatch_warnings(cfg.suppress_word_mismatch_warnings)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision, step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(non_blocking=True)
    )
    set_seed(cfg.seed)
    # VITS uses monotonic alignment to match text tokens to audio frames.
    alignment = MonotonicAlignment().maximum_path

    # -------- 1. Load a pretrained generator and discriminator. ----------
    # Load the model and tokenizer together so dataset token IDs and model
    # embeddings come from the same pretrained checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    model, loading_info = VitsModelForPreTraining.from_pretrained(
        cfg.model_name_or_path, output_loading_info=True
    )


    # Disable layerdrop for both the model and the text encoder to ensure all layers are used during fine-tuning.
    # this was found by debugging with Github Copilot Sol 6.1
    logging.info(f"Setting encoder layerdrop to {cfg.encoder_layerdrop}")
    model.config.layerdrop = cfg.encoder_layerdrop
    model.text_encoder.encoder.layerdrop = cfg.encoder_layerdrop
    if cfg.freeze_mode == "freeze_text_encoder":
        logging.info("Freezing text encoder as per configuration.")
        model.text_encoder.requires_grad_(False)



    # Fine-tuning requires the checkpoint's generator and discriminator weights
    # to load as expected; reject incomplete or incompatible checkpoints.
    if loading_info["missing_keys"] or loading_info["mismatched_keys"]:
        raise ValueError(f"Checkpoint is missing compatible training weights: {loading_info}")
    if (cfg.target_sr != model.config.sampling_rate or cfg.hop_length != model.config.hop_length
            or cfg.n_fft // 2 + 1 != model.config.spectrogram_bins):
        raise ValueError("DataConfig sample rate, hop length and FFT bins must match the checkpoint.")
    if tokenizer.pad_token_id != 0:
        raise ValueError("TTSCollate pads with zero; this checkpoint must use padding token ID zero.")
    if model.config.num_speakers > 1 and not 0 <= cfg.speaker_id < model.config.num_speakers:
        raise ValueError(f"speaker_id must be between 0 and {model.config.num_speakers - 1}.")
    # Single-speaker checkpoints do not use a speaker embedding input.
    speaker_id = cfg.speaker_id if model.config.num_speakers > 1 else None
    # The model slices latent/spectrogram sequences in frames, but audio crops in samples.
    segment_frames = model.segment_size
    segment_samples = model.config.segment_size

    # Split the discriminator from the loaded model so it can have its own optimizer
    # and update schedule; it is reattached when saving the final model.
    discriminator = model.discriminator
    # detach the discriminator
    del model.discriminator


    # Apply weight normalization for adversarial training, since the checkpoint may not have it applied.
    # this is suggested by the VITS / HiFi-GAN training procedure to stabilize adversarial training.
    # it parameterizes the weights by magnitude and direction
    # this is different from say batch normalisation which normalizes activations; weight norm reparameterizes the weights themselves.
    model.decoder.apply_weight_norm() # Apply weight normalization to the decoder
    for flow in model.flow.flows:
        # Apply weight normalization to the flow's pre and post convolutions
        torch.nn.utils.weight_norm(flow.conv_pre)
        torch.nn.utils.weight_norm(flow.conv_post)
    for disc in discriminator.discriminators:
        disc.apply_weight_norm() # Apply weight normalization to the discriminator
    # Recompute selected activations during backward to reduce memory at extra compute cost.
    # disable checkpointing for faster training if memory is not a constraint.
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()


    # WIP code. This freezes everything in the model EXCEPT for the speaker embedding.
    # Could be useful to test whether the generator is already good at generating speech for a given speaker,
    # and we just need to learn an embedding that represents the target speaker.
    # In theory it may produce higher quality since the training is much more focused.
    if cfg.freeze_mode == "unfreeze_speech_embeddings_only":

        if not hasattr(model, "embed_speaker"):
            raise ValueError("This checkpoint has no speaker embedding.")

        model.requires_grad_(False)
        model.embed_speaker.requires_grad_(True)
        logging.info("Only the generator's speaker embedding is trainable.")

        # freeze the discriminator as well, since we only want to train the speaker embedding
        discriminator.requires_grad_(False)

        # possibly unfreeze the duration predictor?
        # model.duration_predictor.requires_grad_(True)

    # -------- 2. Load and prepare the dataset. ----------
    if args.output_mode == "restart" and accelerator.is_main_process:
        for path in (
            cfg.output_dir,
            cfg.output_dir.parent / "listening_samples",
            cfg.output_dir.parent / "intermediate_checkpoints",
            cfg.output_split_dir,
        ):
            if path.exists():
                shutil.rmtree(path)
    accelerator.wait_for_everyone()

    with accelerator.main_process_first():
        train_loader, eval_loader, validation_dataset = load_data(cfg, tokenizer, cfg.num_workers)

    if accelerator.is_main_process:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        # Set up logging to file for the main process.
        file_handler = logging.FileHandler(
            cfg.output_dir / "training.log", mode="w", encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s"
        ))
        logging.getLogger().addHandler(file_handler)
    accelerator.wait_for_everyone()

    # audio processor for loss computation and feature extraction.
    audio_processor = AudioProcessor(cfg)

    sample_sounds = None
    if accelerator.is_main_process:
        sample_sounds = save_sample_sounds_each_epoch(validation_dataset, cfg, speaker_id)
        sample_sounds.save_reference_audio()
        sample_sounds.save_prompt_metadata()
        # save samples prior to training to see how they sound
        sample_sounds.save_epoch(accelerator.unwrap_model(model), 0)
    accelerator.wait_for_everyone()

    # -------- 3. Optimizers and schedulers. ----------
    # Separate optimizers keep generator and discriminator parameter updates independent.
    # note that although the generator and discriminator share the same learning rate, and betas,
    # the losses are weighted differently and so update sizes are likely different
    # the betas control the momentum terms for the AdamW optimizer. Higher beta values give more weight to past gradients.
    # beta1 smoothes past gradients, beta2 smoothes past squared gradients.
    
    # isolate parameters associated with the speaker embeddings
    speaker_params = list(model.embed_speaker.parameters())
    speaker_param_ids = {id(p) for p in speaker_params}

    # todo: set a different learning rate for speaker embedding parameters if needed
    gen_optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in speaker_param_ids],},
            {"params": speaker_params,"lr": cfg.learning_rate * cfg.speaker_embedding_lr_ratio},
        ],
        lr=cfg.learning_rate, betas=(cfg.adam_beta1, cfg.adam_beta2),
    )

    train_discriminator = any(parameter.requires_grad for parameter in discriminator.parameters())
    disc_optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=cfg.learning_rate, betas=(cfg.adam_beta1, cfg.adam_beta2)
    )
    # Move/wrap both networks, optimizers, and the loader for the selected devices,
    # distributed execution, and mixed precision.
    model, discriminator, gen_optimizer, disc_optimizer, train_loader = accelerator.prepare(
        model, discriminator, gen_optimizer, disc_optimizer, train_loader
    )
    # wrap the evaluation loader for distributed execution and mixed precision if it exists
    if eval_loader is not None:
        eval_loader = accelerator.prepare(eval_loader)
    # Scheduler steps count optimizer updates, not raw batches, when gradients
    # are accumulated over multiple batches.
    # gradient accumulation can allow for effectively larger batch sizes without increasing memory usage.
    updates_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    total_steps = cfg.num_train_epochs * updates_per_epoch
    # note: come back to this
    gen_scheduler = make_scheduler(gen_optimizer, cfg, total_steps)
    disc_scheduler = make_scheduler(disc_optimizer, cfg, total_steps) if train_discriminator else None
    if accelerator.is_main_process:
        if gen_scheduler is None:
            logger.info("Using a constant learning rate.")
        elif cfg.warmup_ratio:
            logger.info(
                "Using a linear learning rate schedule with warmup. Warmup ratio: %s",
                cfg.warmup_ratio,
            )
        else:
            logger.info("Using a linear learning rate schedule without warmup.")
        logger.info(
            "Training %s recordings on %s; effective batch size = %s",
            len(train_loader.dataset),
            accelerator.device,
            cfg.per_device_train_batch_size
            * accelerator.num_processes
            * cfg.gradient_accumulation_steps,
        )


    # -------- 4. Start a fresh training run. ----------
    global_step = 0
    history_columns = ["epoch", *LOSS_NAMES]
    training_loss_history = pd.DataFrame(columns=history_columns)
    validation_loss_history = pd.DataFrame(columns=history_columns)
    # Only one process should write shared config/tokenizer files.
    if accelerator.is_main_process:
        # Only the main process should write the training configuration and tokenizer files.
        (cfg.output_dir / "training_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
        tokenizer.save_pretrained(cfg.output_dir) # save the tokenizer files to the output directory

    # 5 ---------- main training loop ----------
    progress = tqdm(total=total_steps, disable=not accelerator.is_local_main_process) # create a progress bar for tracking training steps
    model.train() # set the model to training mode
    discriminator.train() # set the discriminator to training mode
    gen_optimizer.zero_grad() # zero the gradients for the generator optimizer
    disc_optimizer.zero_grad() # zero the gradients for the discriminator optimizer

    # log the training device and CUDA memory stats if applicable
    device = accelerator.device
    using_cuda = device.type == "cuda"
    accelerator.wait_for_everyone()
    if using_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    if accelerator.is_main_process:
        logger.info(
            "Training device: %s (%s)",
            torch.cuda.get_device_name(device) if using_cuda else str(device),
            device,
        )
    # record the start time of training
    # perf_counter is used to get a high-resolution timestamp for measuring training duration.
    training_start = time.perf_counter()


    # Begin the training loop over epochs
    for epoch in range(cfg.num_train_epochs):
        epoch_loss_totals = torch.zeros(len(LOSS_NAMES) + 1, device=accelerator.device)
        # Iterate over batches in the training loader for the current epoch
        for batch in train_loader:
            # Accumulate gradients across batches when configured, synchronizing
            # and updating parameters only at accumulation boundaries.
            with accelerator.accumulate(model, discriminator):

                # handle input preparation and forward pass through the model
                inputs = prepare_batch(batch, speaker_id, segment_frames) # prepare the input batch for the model
                outputs = model(**inputs, monotonic_alignment_function=alignment, return_dict=True) # forward pass through the model
                # The model outputs include the generated waveform and alignment information.


                # slice the target audio to match the generated segment
                # i.e. crop the real waveform to the same length as the generated segment for the discriminator.
                target_audio = slice_segments(batch["wav_padded"], outputs.ids_slice * cfg.hop_length, segment_samples)

                # First update the discriminator: distinguish real audio from
                real_scores, _ = discriminator(target_audio)    # Discriminator scores for real audio
                fake_scores, _ = discriminator(outputs.waveform.detach())  # Discriminator scores for generated audio
                disc_loss = discriminator_loss(real_scores, fake_scores)  # Compute discriminator loss
                if not torch.isfinite(disc_loss):
                    raise FloatingPointError("Discriminator loss is not finite.")
                # Scale the discriminator objective, then clip gradients after
                # unscaling to prevent unstable updates.
                if train_discriminator:
                    accelerator.backward(cfg.weight_disc * disc_loss)
                    if accelerator.sync_gradients: # only unscale and clip when gradients are being synchronized
                        accelerator.unscale_gradients(disc_optimizer)
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0) # clip discriminator gradients to prevent exploding gradients
                    disc_optimizer.step() # update discriminator parameters
                    disc_optimizer.zero_grad() # reset discriminator gradients

                # Freeze discriminator weights while keeping gradients to generated audio.
                # This lets the generator learn from discriminator feedback without
                # modifying discriminator parameters during the generator update.
                discriminator.requires_grad_(False)
                with torch.no_grad():
                    _, real_maps = discriminator(target_audio) # Extract discriminator feature maps for real audio
                fake_scores, fake_maps = discriminator(outputs.waveform) # Discriminator scores and feature maps for generated audio

                # Compute reconstruction losses for generated audio
                # Reconstruction losses include mel-spectrogram loss and latent space loss
                # mel-spectrogram loss measures the difference between the generated and target mel-spectrograms, rather than the raw waveforms.
                # latent loss measures the difference between the generated and target latent representations, i.e. feature maps from the discriminator.
                mel_loss, latent_loss = reconstruction_losses(outputs, batch, audio_processor, segment_frames)
                # Combine reconstruction losses with adversarial loss to form the generator loss
                losses = generator_losses(outputs, real_maps, fake_maps, fake_scores, mel_loss, latent_loss, cfg)
                gen_loss = losses["generator_total"]
                losses["disc"] = disc_loss
                accumulate_losses(epoch_loss_totals, losses, inputs["input_ids"].size(0), accelerator)
                if not torch.isfinite(gen_loss):
                    raise FloatingPointError("Generator loss is not finite.")

                # Backpropagate through the discriminator into the generated
                # waveform, then re-enable discriminator gradients for next batch.
                accelerator.backward(gen_loss) # backpropagate generator loss
                discriminator.requires_grad_(train_discriminator)
                if accelerator.sync_gradients:
                    accelerator.unscale_gradients(gen_optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip generator gradients to prevent exploding gradients
                gen_optimizer.step() # update generator parameters
                gen_optimizer.zero_grad() # reset generator gradients for next iteration
                # Step-based schedules advance only when an optimizer update
                # actually happened (not when mixed precision skips an update).
                if accelerator.sync_gradients:
                    if gen_scheduler is not None and not gen_optimizer.step_was_skipped:
                        gen_scheduler.step() # advance generator learning rate schedule
                    if disc_scheduler is not None and not disc_optimizer.step_was_skipped:
                        disc_scheduler.step()

            if accelerator.sync_gradients: # only update global step and progress when gradients are synchronized
                global_step += 1
                progress.update(1) # increment progress bar by one step
                progress.set_postfix(gen=f"{gen_loss.item():.3f}", disc=f"{disc_loss.item():.3f}") # display current generator and discriminator losses in the progress bar
                if cfg.do_eval and global_step % cfg.eval_steps == 0: # perform evaluation at the specified interval
                    # Evaluate the model on the validation set at the specified interval
                    metrics = evaluate(model, discriminator, eval_loader, alignment, audio_processor,
                                       speaker_id, segment_frames, accelerator, cfg)
                    if accelerator.is_main_process:
                        logger.info("Step %s: validation %s", global_step, metrics)

        # -------- Validate and save epoch outputs. ----------
        train_metrics = averaged_losses(epoch_loss_totals)
        validation_metrics = None
        if cfg.do_eval:
            validation_metrics = evaluate(
                model, discriminator, eval_loader, alignment, audio_processor,
                speaker_id, segment_frames, accelerator, cfg,
            )
        if accelerator.is_main_process:
            logger.info(
                "Epoch %s: training %s, validation %s",
                epoch + 1,
                train_metrics,
                validation_metrics,
            )
        # update the training and validation loss history
        training_loss_history.loc[len(training_loss_history)] = {"epoch": epoch + 1, **train_metrics,}
        validation_loss_history.loc[len(validation_loss_history)] = {
            "epoch": epoch + 1,**({name: None for name in LOSS_NAMES} if validation_metrics is None else validation_metrics)}
        # save the current epoch's model checkpoint
        # note that this doesn't save the discriminator, so it's not a resumable checkpoint
        # it's more for assessing intermediate quality if needed
        if accelerator.is_main_process:
            model_checkpoint_dir = (
                cfg.output_dir.parent / "intermediate_checkpoints" 
            )
            accelerator.unwrap_model(model).save_pretrained(model_checkpoint_dir, variant=f"Epoch_{epoch + 1}")
            logger.info("Saved model checkpoint to %s", model_checkpoint_dir)
        
        # Save the three fixed listening samples for this epoch.
        if accelerator.is_main_process:
            sample_sounds.save_epoch(accelerator.unwrap_model(model), epoch + 1)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # Save the training and validation loss history as CSV files and plot the loss curves.
            save_loss_history_csvs( training_loss_history, validation_loss_history, cfg.output_dir)
            plot_training_history(training_loss_history, validation_loss_history, cfg.output_dir / "loss_curves.png")
        accelerator.wait_for_everyone()


    

    # 6. Export the combined model without training-only weight normalization.
    # Only the main process writes the final model; all processes synchronize
    # before shutting down the distributed training session.
    progress.close() # Close the progress bar at the end of training

    # Synchronize all processes and measure the total training time.
    accelerator.wait_for_everyone()
    if using_cuda:
        torch.cuda.synchronize(device)
    training_time = time.perf_counter() - training_start

    # Log the total training time and peak CUDA memory usage if applicable.
    if accelerator.is_main_process:
        logger.info(
            "Total training time: %.2f seconds (%.2f hours)",
            training_time,
            training_time / 3600,
        )
        if using_cuda:
            logger.info(
                "Peak CUDA memory: %.2f GiB allocated; %.2f GiB reserved",
                torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                torch.cuda.max_memory_reserved(device) / (1024 ** 3),
            )

    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model) # Unwrap the model from the accelerator for final saving
        discriminator = accelerator.unwrap_model(discriminator) # Unwrap the discriminator from the accelerator for final saving
        if cfg.save_final_discriminator:
            logger.info("Including the discriminator in the final checkpoint.")
            model.discriminator = discriminator # re-attach the discriminator to the model for final saving
        model.decoder.remove_weight_norm() # Remove weight normalization from the decoder
        for flow in model.flow.flows:
            torch.nn.utils.remove_weight_norm(flow.conv_pre) # Remove weight normalization from the flow's pre-conv layer
            torch.nn.utils.remove_weight_norm(flow.conv_post) # Remove weight normalization from the flow's post-conv layer
        for disc in discriminator.discriminators:
            disc.remove_weight_norm() # Remove weight normalization from each discriminator
        model.save_pretrained(cfg.output_dir, variant=args.config.stem) # Save the final model with the config name as its variant
        logger.info("Saved fine-tuned model to %s", cfg.output_dir)
    accelerator.wait_for_everyone() # Ensure all processes have finished saving before ending training
    accelerator.end_training() # End the training session and clean up resources. Huzzah!!

    # Final cleanup and logging
    if accelerator.is_main_process:
        logger.info("Training completed successfully.")
        logger.info("Final model and checkpoints are saved in %s", cfg.output_dir)




if __name__ == "__main__":
    main()
