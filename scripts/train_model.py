"""
Fine-tune a VITS checkpoint on local LJSpeech data.

Run from the repository root:
    python -m scripts.train_model
    accelerate launch scripts/train_model.py

The default settings are loaded from configs/best_speaker_embedding.yaml. 
"--config" 
    can select another YAML object containing training settings. 
"--output-mode" 
    "restart" starts a fresh run, and 
    "new" requires an empty output root, otherwise an error is raised.
The default mode is `restart`. 

Run files are written under `<output_dir>/run`, including configs, logs, and the final checkpoint.
Intermediate checkpoints are saved in the "intermediate_checkpoints" directory.
Data split manifests are written in the "splits" directory under the output root.
samples and intermediate checkpoints are written beside `run`.

Evaluation of both the training and validation sets are done once per epoch.
An intermediate checkpoint is saved at the end of each epoch.

This script has been adapted from ylacombe/finetune-hf-vits/run_vits_finetuning.py,
predominantly for its loss calculation functions and general training loop structure.

The main changes are:
- Deleted data loading and preprocessing logic. This has been re-implemented elsewhere in the project.
- Greatly simplified the training loop to eliminate fluff that's not essential for fine-tuning on 
    a small number of epochs, such as resuming from earlier checkpoints, complex learning rate schedules, 
    evaluation at arbitrary intervals, the ability to train across multiple GPUs etc
- Explicity enabled asynchronous data loading (pinned memory=True and non-blocking transfers) to improve
 GPU utilization.
- Added an option to enable test mode, which runs a single epoch on a reduced dataset for quick 
validation of the training setup.
- Added an option to adjust the layerdrop rate in the text encoder (default 0.1 in the original training recipe)
- Added an option to freeze certain layers of the model during fine-tuning to reduce computational load 
    and prevent overfitting.
- Added an option to vary the learning rate of the learned speaker embedding relative to the rest of the model.
"""


import argparse
import json
import logging
import math
import random
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase
import time

# Support direct execution and the upstream configuration used by the local model.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "finetune-hf-vits-main"))

from scripts.DataConfig import DEFAULT_CONFIG_PATH, DataConfig, load_config
from scripts.DataLoader import build_dataloaders
from scripts.TextProcessor import configure_word_mismatch_warnings, load_vctk_tokenizer
from scripts.AudioProcessor import AudioProcessor
from scripts.monotonic_alignment import MonotonicAlignment
from scripts.model import VitsDiscriminator, VitsModel, VitsModelForPreTraining, slice_segments

from scripts.plots import (
    LOSS_NAMES,
    plot_training_history,
    save_loss_history_csvs,
)

logger = logging.getLogger(__name__)


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

    # DataConfig owns YAML loading and configuration validation.
    settings = load_config(args.config)
    cfg = DataConfig(**settings)

    # Apply test mode settings if enabled
    if args.test_mode == 'yes':
        cfg.num_train_epochs = 1
        cfg.subset_size = 10 * cfg.per_device_train_batch_size  # Reduce dataset size for quick testing

    try:
        cfg.validate(output_mode=args.output_mode)
    except ValueError as error:
        parser.error(str(error))
    cfg.output_dir /= "run"
    return cfg, args


def load_data(cfg, tokenizer, num_workers, speaker_id, segment_frames):
    """
    Load the training and validation data according to the configuration.
    """
    tokenizer.add_blank = cfg.add_blank
    # Pass the checkpoint tokenizer directly so dataset token IDs match the model.
    # The validation loader keeps its configured batch size and sample cap.
    loaders = build_dataloaders(
        cfg,
        batch_size=cfg.per_device_train_batch_size,
        num_workers=num_workers,
        tokenizer=tokenizer,
        speaker_id=speaker_id,
        segment_frames=segment_frames,
    )
    # check the training loader for emptiness
    if not len(loaders["train"].dataset):
        raise ValueError("The training split is empty. Check duration limits, subset size and split ratios.")

    # check the evaluation loader for emptiness
    eval_loader = loaders["val"] if cfg.do_eval else None
    if eval_loader is not None and not len(eval_loader.dataset):
        raise ValueError("The validation split is empty. Increase the subset or validation ratio.")
    return loaders["train"], eval_loader


def discriminator_loss(real_outputs, generated_outputs):
    """
    Compute the discriminator loss for real and generated outputs using a least-squares GAN objective.

    This works by encouraging the discriminator to output values close to 1 for real audio and 
    close to 0 for generated audio, using a least-squares loss. I.e. it helps the discriminator 
    learn to distinguish real from generated audio effectively, which indirectly improves the generator's 
    output quality over repeated training iterations.

    This loss ONLY applies to the discriminator and does not affect the generator directly.
    The generator is frozen with respect to this loss and is not updated based on it.

    Args:
        real_outputs (list of torch.Tensor): Discriminator outputs for real audio.
        generated_outputs (list of torch.Tensor): Discriminator outputs for generated audio.

    Returns:
        torch.Tensor: The computed discriminator loss.
    
    Function taken from ylacombe/finetune-hf-vits/run_vits_finetuning.py with comments added for clarity.
    """
    # Least-squares GAN objective: real audio should score near 1, generated
    # audio near 0. Sum across the discriminator's multiple output scales.
    return sum(
        ((1 - real.float()) ** 2).mean() + (generated.float() ** 2).mean()
        for real, generated in zip(real_outputs, generated_outputs)
    )


def kl_loss(outputs):
    """
    Compute the KL divergence loss between the posterior and prior latent distributions.

    This loss encourages the posterior latent distribution (derived from speech), after transformation
    through the normalizing flow, to match the text-conditioned prior.
    This regularises the latent space and helps ensure that latents generated from text at inference are
    consistent with those used to reconstruct speech during training.

    Computes a sample-based KL objective between the flow-transformed speech posterior and the 
    text-conditioned Gaussian prior.

    I.e instead of comparing two distributions, we draw a sample from the posterior and evaluate its 
    likelihood under the prior.
    This is done because the normalizing flow makes the posterior distribution complex and generally
    non-Gaussian, so we can't just compute a closed-form KL divergence 

    The equation for this is: 
    KL(posterior || prior) = sum(prior_log_variances - posterior_log_variances - 0.5 + 0.5 * (prior_latents - prior_means)^2 / exp(2 * prior_log_variances))
    - prior_log_variances: Log standard deviation (log σ_p) of the text-conditioned Gaussian prior.
    - posterior_log_variances: Log standard deviation (log σ_q) of the speech-conditioned Gaussian posterior.
    - prior_latents: Flow-transformed posterior samples z_p = f(z), where z is sampled from the speech-conditioned posterior.
    - prior_means: Mean (μ_p) of the text-conditioned Gaussian prior.
    
    Note there is no log-jacobian term, since a volume-preserving flow is assumed.

    Args:
        outputs: Model outputs containing posterior and prior latent distributions and padding mask.

    Returns:
        torch.Tensor: The computed KL divergence loss.


    Function taken from ylacombe/finetune-hf-vits/run_vits_finetuning.py with comments added for clarity.
    """
    # Compare posterior and prior latent distributions, excluding padded frames.
    prior_log = outputs.prior_log_variances.float() 
    # Compute the element-wise KL divergence between the posterior and prior distributions.
    kl = prior_log - outputs.posterior_log_variances.float() - 0.5
    kl += 0.5 * (outputs.prior_latents.float() - outputs.prior_means.float()) ** 2 * torch.exp(-2 * prior_log)
    # Apply the padding mask to exclude padded frames from the KL loss computation.
    mask = outputs.labels_padding_mask
    return (kl * mask).sum() / mask.sum()


def reconstruction_losses(outputs, batch, audio_processor, segment_frames):
    """
    Compute the reconstruction losses for the generated audio. This is done on the 
    log mel spectrograms of the audio segments.
    
    I.e. it computes the L1 norm between the generated and target log mel spectrograms.
    These are 2D tensors with shape (batch_size, num_mel_bins, num_frames).

    Note that VITS generates a fixed-length audio segment from a randomly selected slice
    of the latent sequence. 

    Args:
        outputs: Model outputs containing generated waveform and latent distributions.
        batch: Input batch containing target mel spectrograms.
        audio_processor: Audio processor used to compute spectrograms.
        segment_frames: Number of frames in the audio segment.

    Returns:
        tuple: L1 mel reconstruction loss and KL divergence loss.

    Function taken from ylacombe/finetune-hf-vits/run_vits_finetuning.py with comments added for clarity.
    """
    # Align target and generated mel features to the model's sampled audio
    # segment, then pair reconstruction loss with the latent KL regularizer.
    target_mel = slice_segments(batch["mel_padded"], outputs.ids_slice, segment_frames)
    # Use the same audio processor for real and generated mel features.
    _, generated_mel = audio_processor.compute_spectrograms(outputs.waveform.float().squeeze(1))
    if generated_mel.ndim == 2:
        generated_mel = generated_mel.unsqueeze(0)
    return F.l1_loss(generated_mel, target_mel.float()), kl_loss(outputs)



def generator_losses(outputs, real_maps, generated_maps, generated_scores, mel_loss, latent_loss, cfg):
    """
    Compute the generator loss combining duration, mel, KL, feature-matching, and adversarial objectives.

    Duration loss: stochastic duration negative log-likelihood loss (variational upper bound)
    The duration predictor predicts probability distributions of plausible durations for each token.
    The loss calculates how well the predicted duration distributions match the target durations provided by
    the monotonic alignment between text and audio.
    Has a jacobian term that accounts for the change of variables in the duration prediction (uses a 
    normalizing flow). 
    Also has a posterior correction term that accounts for the difference between the predicted and actual durations.

    Adversarial loss: simply how well the generator fools the discriminator into classifying 
    generated audio as real. I.e. root mean square error between the discriminator's output 
    and the target label of 1.0 for real audio.

    Feature-matching loss: measures the discrepancy between the discriminator's intermediate 
    feature maps for real and generated audio. 
    Uses L1 loss to measure the difference between the feature maps of real and generated audio,
    i.e. the L1 norm of the difference between corresponding feature maps (|real - generated|).
    It's multiplied by a weight factor (usually 2) to balance its contribution relative to other loss terms.
    It gives a richer training signal to the generator by encouraging it to match the internal representations of real audio, not just the final discriminator output.

    Args:
        outputs: Model outputs containing log duration and other relevant information.
        real_maps: List of discriminator feature maps for real audio.
        generated_maps: List of discriminator feature maps for generated audio.
        generated_scores: List of discriminator scores for generated audio.
        mel_loss: L1 mel reconstruction loss.
        latent_loss: KL divergence loss.
        cfg: Configuration object containing loss weights.

    Returns:
        dict: Dictionary containing individual loss components and the total generator loss.

    Function taken from ylacombe/finetune-hf-vits/run_vits_finetuning.py with comments added for clarity.
    """
    # Combine duration, mel, KL, feature-matching, and adversarial objectives.
    # Config weights make the relative contribution of each term explicit.

    # duration loss is calculated as the sum of the log duration predictions from the model.
    duration_loss = outputs.log_duration.float().sum()
    # feature matching loss
    # outer loop iterates over the layers of the discriminator's feature maps for real and generated audio
    # inner loop iterates over the individual feature maps within each layer
    # each element in the inner loop corresponds to a feature map from the real and generated audio
    feature_loss = 2 * sum(
        F.l1_loss(generated.float(), real.detach().float())
        for real_layers, generated_layers in zip(real_maps, generated_maps)
        for real, generated in zip(real_layers, generated_layers)
    )
    # calculate the adversarial loss for the generator based on the discriminator's scores for generated audio
    adversarial_loss = sum(((1 - score.float()) ** 2).mean() for score in generated_scores)
    # weight the loss components according to the configuration before summing them to get the total loss
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
    """
    Accumulate losses over multiple batches for logging and averaging.

    Args:
        totals: Tensor storing accumulated losses and count.
        losses: Dictionary of individual loss components for the current batch.
        batch_size: Number of samples in the current batch.
        accelerator: Accelerator object for distributed training.

    """
    values = torch.stack([losses[name].detach().float() for name in LOSS_NAMES])
    if not torch.isfinite(values).all():
        raise FloatingPointError("Logged losses contain non-finite values.")
    metrics = accelerator.gather_for_metrics(values.repeat(batch_size, 1))
    totals[:-1] += metrics.sum(0)
    totals[-1] += metrics.size(0)


def averaged_losses(totals):
    """
    Compute the averaged losses from the accumulated totals.

    Args:
        totals: Tensor storing accumulated losses and count.

    Returns:
        dict: Dictionary containing averaged loss values for each loss component.

    """
    if totals[-1] == 0:
        raise ValueError("Cannot average losses from an empty loader.")
    return {name: (totals[i] / totals[-1]).item() for i, name in enumerate(LOSS_NAMES)}


@torch.no_grad()
def evaluate(model, discriminator, loader, alignment, audio_processor, speaker_id, segment_frames, accelerator, cfg):
    """
    Do evaluation on the validation set and return averaged metrics.

    Note: evaluation uses random number generation, so changing the number of evaluations
    will change the randomness in training.
    This is not an issue for our assignment if the number of evaluations is kept constant
    across experiments.

    """

    model.eval()
    discriminator.eval()
    total = torch.zeros(len(LOSS_NAMES) + 1, device=accelerator.device)
    for batch in loader:
        inputs = batch["model_inputs"]
        outputs = model(**inputs, monotonic_alignment_function=alignment, return_dict=True)
        mel, latent = reconstruction_losses(outputs, batch, audio_processor, segment_frames)
        target_audio = slice_segments(
            batch["wav_padded"], outputs.ids_slice * cfg.hop_length, cfg.hop_length * segment_frames
        )
        real_scores, real_maps = discriminator(target_audio)
        fake_scores, fake_maps = discriminator(outputs.waveform)
        losses = generator_losses(outputs, real_maps, fake_maps, fake_scores, mel, latent, cfg)
        losses["disc"] = discriminator_loss(real_scores, fake_scores)
        accumulate_losses(total, losses, inputs["input_ids"].size(0), accelerator)
    return averaged_losses(total)



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



@torch.no_grad()
def save_inference_checkpoint(model: VitsModelForPreTraining, tokenizer: PreTrainedTokenizerBase,
    output_dir: Path, variant: str, discriminator: Optional[VitsDiscriminator] = None,
) -> None:
    """
    Save the model and tokenizer for inference.

    Remove training-added normalization while retaining the native WaveNet
    parametrizations expected by Transformers inference models.

    Only the generator is exported for inference; the discriminator is optional.
    """

    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        export_model = VitsModel(deepcopy(model.config)).to(dtype=model.dtype)
        export_model.decoder.apply_weight_norm()
        for flow in export_model.flow.flows:
            torch.nn.utils.weight_norm(flow.conv_pre)
            torch.nn.utils.weight_norm(flow.conv_post)
        export_model.load_state_dict(model.state_dict(), strict=True)

        export_model.decoder.remove_weight_norm()
        for flow in export_model.flow.flows:
            torch.nn.utils.remove_weight_norm(flow.conv_pre)
            torch.nn.utils.remove_weight_norm(flow.conv_post)

        if discriminator is not None:
            export_discriminator = VitsDiscriminator(deepcopy(model.config)).to(
                dtype=next(discriminator.parameters()).dtype,
            )
            export_discriminator.apply_weight_norm()
            export_discriminator.load_state_dict(discriminator.state_dict(), strict=True)
            export_discriminator.remove_weight_norm()
            export_model.discriminator = export_discriminator

        export_model.eval()
        export_model.save_pretrained(output_dir, variant=variant)
        tokenizer.save_pretrained(output_dir)


def main():
    # Configure distributed training and seed Python/NumPy/PyTorch before
    # constructing anything that may use randomness.
    cfg, args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # suppress word mismatch warnings if configured
    configure_word_mismatch_warnings(cfg.suppress_word_mismatch_warnings)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision, step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(non_blocking=True) # asynchronous data loading
    )
    set_seed(cfg.seed)
    # VITS uses monotonic alignment to match text tokens to audio frames.
    alignment = MonotonicAlignment().maximum_path

    # -------- 1. Load a pretrained generator and discriminator. ----------
    # Load the model and tokenizer together so dataset token IDs and model
    # embeddings come from the same pretrained checkpoint.
    # tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)

    # use this version to fix a token collision issue with the "unknown" token
    tokenizer = load_vctk_tokenizer(cfg.model_name_or_path)

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

    speaker_id = cfg.speaker_id if model.config.num_speakers > 1 else None
    if speaker_id is not None and not 0 <= speaker_id < model.config.num_speakers:
        raise ValueError(
            f"speaker_id must be between 0 and {model.config.num_speakers - 1}; got {speaker_id}."
        )
    # The model slices latent/spectrogram sequences in frames, but audio crops in samples.
    segment_frames = model.segment_size # Number of frames in a segment for the model's latent/spectrogram sequences
    segment_samples = model.config.segment_size # Number of audio samples in a segment

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


    # This freezes everything in the model EXCEPT for the speaker embedding.
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
        train_loader, eval_loader = load_data(
            cfg, tokenizer, cfg.num_workers, speaker_id, segment_frames,
        )

    if accelerator.is_main_process:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        # Set up logging to file for the main process.
        file_handler = logging.FileHandler(cfg.output_dir / "training.log", mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logging.getLogger().addHandler(file_handler)
    accelerator.wait_for_everyone()

    # audio processor for loss computation and feature extraction.
    audio_processor = AudioProcessor(cfg)



    # -------- 3. Optimizers and schedulers. ----------
    # Separate optimizers keep generator and discriminator parameter updates independent.
    # note that although the generator and discriminator share the same learning rate, and betas,
    # the losses are weighted differently and so update sizes are likely different
    # the betas control the momentum terms for the AdamW optimizer. Higher beta values give more weight to past gradients.
    # beta1 smoothes past gradients, beta2 smoothes past squared gradients.
    
    # isolate parameters associated with the speaker embeddings
    speaker_params = list(model.embed_speaker.parameters())
    speaker_param_ids = {id(p) for p in speaker_params}

    # set a different learning rate for speaker embedding parameters if needed
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
    # set up the learning rate schedulers for the generator and discriminator
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
                inputs = batch["model_inputs"]
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
                # latent loss measures the difference between the generated and target latent representations
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
            save_inference_checkpoint(
                accelerator.unwrap_model(model), tokenizer, model_checkpoint_dir,
                variant=f"Epoch_{epoch + 1}",
            )
            logger.info("Saved model checkpoint to %s", model_checkpoint_dir)
        
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
        if cfg.save_final_discriminator:
            logger.info("Including the discriminator in the final checkpoint.")
        save_inference_checkpoint(
            accelerator.unwrap_model(model), tokenizer, cfg.output_dir,
            variant=args.config.stem,
            discriminator=accelerator.unwrap_model(discriminator) if cfg.save_final_discriminator else None,
        )
        logger.info("Saved fine-tuned model to %s", cfg.output_dir)
    accelerator.wait_for_everyone() # Ensure all processes have finished saving before ending training
    accelerator.end_training() # End the training session and clean up resources. Huzzah!!

    # Final cleanup and logging
    if accelerator.is_main_process:
        logger.info("Training completed successfully.")
        logger.info("Final model and checkpoints are saved in %s", cfg.output_dir)




if __name__ == "__main__":
    main()
