# coding=utf-8
# Copyright 2023 The Kakao Enterprise Authors and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch VITS model."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.masking_utils import bidirectional_mask_function, eager_mask
from transformers.modeling_outputs import (
    BaseModelOutput,
    ModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from configuration_vits import VitsConfig



logger = logging.get_logger(__name__)


# General docstring
_CONFIG_FOR_DOC = "VitsConfig"


VITS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/mms-tts-eng",
    # See all VITS models at https://huggingface.co/models?filter=vits
    # and all MMS models at https://huggingface.co/models?sort=trending&search=facebook%2Fmms-tts
]


@dataclass
class VitsModelOutput(ModelOutput):
    """
    Describes the outputs for the VITS model, with potential hidden states and attentions.

    Args:
        waveform (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            The final audio waveform predicted by the model.
        sequence_lengths  (`torch.FloatTensor` of shape `(batch_size,)`):
            The length in samples of each element in the `waveform` batch.
        spectrogram (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_bins)`):
            The log-mel spectrogram predicted at the output of the flow model. This spectrogram is passed to the Hi-Fi
            GAN decoder model to obtain the final audio waveform.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attention weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    waveform: torch.FloatTensor = None
    sequence_lengths: torch.FloatTensor = None
    spectrogram: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class VitsTextEncoderOutput(ModelOutput):
    """
    Describes the outputs for the VITS text encoder model, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        prior_means (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The predicted mean values of the prior distribution for the latent text variables.
        prior_log_variances (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The predicted log-variance values of the prior distribution for the latent text variables.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attention weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    prior_means: torch.FloatTensor = None
    prior_log_variances: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class VitsTrainingOutput(ModelOutput):
    """
    Describes the training outputs for the VITS model, used to compute losses when training.

    Args:
        waveform (`torch.FloatTensor` of shape `(batch_size, 1, segment_length)`):
            A sliced-segment of the audio waveform predicted by the model.
        log_duration (`torch.FloatTensor` of shape `(batch_size,)`):
            Log duration of each element of the latent tensor.
        attn (`torch.FloatTensor` of shape `(batch_size, 1, audio_length, latent_length)`):
            Attention mask. Corresponds to the most likely alignment between text and speech?
        ids_slice (`torch.FloatTensor` of shape `(batch_size,)`):
            A 1D tensor containing the starting indices for each segment of the batch.
        input_padding_mask (`torch.FloatTensor` of shape `(batch_size, 1, latent_length)`):
            Input padding mask.
        labels_padding_mask (`torch.FloatTensor` of shape `(batch_size, 1, audio_length)`):
            Labels padding mask.
        latents (`torch.FloatTensor` of shape `(batch_size, hidden_size, audio_length)`):
            Latent variable.
        prior_latents (`torch.FloatTensor` of shape `(batch_size, hidden_size, audio_length)`):
            Prior latent variable.
        prior_means (`torch.FloatTensor` of shape `(batch_size, hidden_size, audio_length)`):
            Prior means.
        prior_log_variances (`torch.FloatTensor` of shape `(batch_size, hidden_size, audio_length)`):
            Prior log variance.
        posterior_means (`torch.FloatTensor` of shape `(batch_size, hidden_size, audio_length)`):
            Posterior means.
        posterior_log_variances (`torch.FloatTensor` of shape `(batch_size, hidden_size, audio_length)`):
            Posterior log variance.
    """

    waveform: torch.FloatTensor = None
    log_duration: torch.FloatTensor = None
    attn: torch.FloatTensor = None
    ids_slice: torch.FloatTensor = None
    input_padding_mask: torch.FloatTensor = None
    labels_padding_mask: torch.FloatTensor = None
    latents: torch.FloatTensor = None
    prior_latents: torch.FloatTensor = None
    prior_means: torch.FloatTensor = None
    prior_log_variances: torch.FloatTensor = None
    posterior_means: torch.FloatTensor = None
    posterior_log_variances: torch.FloatTensor = None


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, num_channels):
    """Apply WaveNet's gated tanh-sigmoid activation to combined input and conditioning.

    The tanh and sigmoid halves form a gate that controls which transformed features
    pass through each WaveNet layer. Combining the addition and activations in one
    scripted operation keeps this frequently used layer computation compact.

    The purpose of the gate is to control which features are allowed to pass through each layer, 
    enabling the network to model complex dependencies and interactions in the input sequence of 
    """
    in_act = input_a + input_b # Combine the input and conditioning for the gated activation.
    t_act = torch.tanh(in_act[:, :num_channels, :]) # Apply the tanh activation to the first half of the channels.
    s_act = torch.sigmoid(in_act[:, num_channels:, :]) # Apply the sigmoid activation to the second half of the channels.
    acts = t_act * s_act # Combine the tanh and sigmoid activations to form the gated output.
    return acts


def slice_segments(hidden_states, ids_str, segment_size=4):
    """
    This method slices segments of the same size from the input tensor.

    Args:
        hidden_states (`torch.Tensor` of shape `(batch_size, channels, seq_len)`):
            The input tensor to be sliced alongside the `seq_len` dimension.
        ids_str (`torch.Tensor` of shape `(batch_size,)`):
            A 1D tensor containing the starting indices for each segment.
        segment_size (`int`, defaults to 4):
            The size of each segment to slice
    Returns:
        output (`torch.Tensor` of shape `(batch_size, channels, segment_size)`): A tensor containing the sliced
        segments.
    """

    batch_size, channels, _ = hidden_states.shape
    # 1d tensor containing the indices to keep
    indices = torch.arange(segment_size).to(ids_str.device)
    # extend the indices to match the shape of hidden_states
    indices = indices.view(1, 1, -1).expand(batch_size, channels, -1)
    # offset indices with ids_str
    indices = indices + ids_str.view(-1, 1, 1)

    # gather indices
    output = torch.gather(hidden_states, dim=2, index=indices)

    return output


def rand_slice_segments(hidden_states, sample_lengths=None, segment_size=4):
    """
    This method randomly slices segments of the same size from the input tensor.

    Args:
        hidden_states (`torch.Tensor` of shape `(batch_size, channels, seq_len)`):
            The input tensor to be sliced alongside the `seq_len` dimension.
        sample_lengths (`torch.Tensor` of shape `(batch_size,)`, *optional*):
            A 1D tensor containing the sequence length of each sample of the batch. Will defaults to `seq_len` if not
            provided.
        segment_size (`int`, defaults to 4):
            The size of each segment to slice
    Returns:
        output (`torch.Tensor` of shape `(batch_size, channels, segment_size)`): A tensor containing the sliced
        segments.
    """
    batch_size, _, seq_len = hidden_states.size()
    if sample_lengths is None:
        sample_lengths = seq_len
    ids_str_max = sample_lengths - segment_size + 1
    ids_str = (torch.rand([batch_size]).to(device=hidden_states.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(hidden_states, ids_str, segment_size)
    return ret, ids_str


def monotonic_align_max_path(log_likelihoods, mask):
    """
    Returns the most likely alignment between text and speech for the given log-likelihood matrix.
    Note: a faster version of this is used in practice for training.

    Args:
        log_likelihoods (`torch.Tensor` of shape `(batch_size, text_length, latent_variable_length)`):
            The log-likelihood matrix for each sample of the batch.
        mask (`torch.Tensor` of shape `(batch_size, text_length, latent_variable_length)`, *optional*):
            The padding mask.
    Returns:
        path (`torch.Tensor` of shape `(batch_size, text_length, latent_variable_length)`): The most likely alignement.
    """
    # used for training - awfully slow
    # an alternative is proposed in examples/pytorch/text-to-speech/run_vits_finetuning.py
    path = torch.zeros_like(log_likelihoods)

    text_length_maxs = mask.sum(1)[:, 0]
    latent_length_maxs = mask.sum(2)[:, 0]

    indexes = latent_length_maxs - 1

    max_neg_val = -1e9

    for batch_id in range(len(path)):
        index = int(indexes[batch_id].item())
        text_length_max = int(text_length_maxs[batch_id].item())
        latent_length_max = int(latent_length_maxs[batch_id].item())

        for y in range(text_length_max):
            for x in range(max(0, latent_length_max + y - text_length_max), min(latent_length_max, y + 1)):
                if x == y:
                    v_cur = max_neg_val
                else:
                    v_cur = log_likelihoods[batch_id, y - 1, x]
                if x == 0:
                    if y == 0:
                        v_prev = 0.0
                    else:
                        v_prev = max_neg_val
                else:
                    v_prev = log_likelihoods[batch_id, y - 1, x - 1]
                log_likelihoods[batch_id, y, x] += max(v_prev, v_cur)

        for y in range(text_length_max - 1, -1, -1):
            path[batch_id, y, index] = 1
            if index != 0 and (
                index == y or log_likelihoods[batch_id, y - 1, index] < log_likelihoods[batch_id, y - 1, index - 1]
            ):
                index = index - 1
    return path


def _unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    reverse=False,
    tail_bound=5.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
):
    """
    This transformation represents a monotonically increasing piecewise rational quadratic function. The predicted
    widths, heights, and derivatives let the stochastic duration predictor learn a flexible, invertible mapping for
    its duration-related latent variables. In the forward direction, its log-determinant provides the change-of-
    variables correction needed to compute likelihoods; in reverse, the same mapping transforms sampled noise back
    into duration predictions. Outside of the `tail_bound`, the transform behaves as an identity function.

    Args:
        inputs (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Second half of the hidden-states input to the Vits convolutional flow module.
        unnormalized_widths (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            First `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        unnormalized_heights (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            Second `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        unnormalized_derivatives (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            Third `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        reverse (`bool`, *optional*, defaults to `False`):
            Whether the model is being run in reverse mode.
        tail_bound (`float`, *optional* defaults to 5):
            Upper and lower limit bound for the rational quadratic function. Outside of this `tail_bound`, the
            transform behaves as an identity function.
        min_bin_width (`float`, *optional*, defaults to 1e-3):
            Minimum bin value across the width dimension for the piecewise rational quadratic function.
        min_bin_height (`float`, *optional*, defaults to 1e-3):
            Minimum bin value across the height dimension for the piecewise rational quadratic function.
        min_derivative (`float`, *optional*, defaults to 1e-3):
            Minimum bin value across the derivatives for the piecewise rational quadratic function.
    Returns:
        outputs (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Hidden-states as transformed by the piecewise rational quadratic function with the `tail_bound` limits
            applied.
        log_abs_det (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Logarithm of the absolute value of the determinants corresponding to the `outputs` with the `tail_bound`
            limits applied.
    """
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    log_abs_det = torch.zeros_like(inputs)
    constant = np.log(np.exp(1 - min_derivative) - 1)

    unnormalized_derivatives = nn.functional.pad(unnormalized_derivatives, pad=(1, 1))
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    log_abs_det[outside_interval_mask] = 0.0

    outputs[inside_interval_mask], log_abs_det[inside_interval_mask] = _rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        reverse=reverse,
        tail_bound=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    return outputs, log_abs_det


def _rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    reverse,
    tail_bound,
    min_bin_width,
    min_bin_height,
    min_derivative,
):
    """
    This transformation represents a monotonically increasing piecewise rational quadratic function. Unlike the
    function `_unconstrained_rational_quadratic_spline`, the function behaves the same across the `tail_bound`.

    Args:
        inputs (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Second half of the hidden-states input to the Vits convolutional flow module.
        unnormalized_widths (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            First `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        unnormalized_heights (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            Second `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        unnormalized_derivatives (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            Third `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        reverse (`bool`):
            Whether the model is being run in reverse mode.
        tail_bound (`float`):
            Upper and lower limit bound for the rational quadratic function. Outside of this `tail_bound`, the
            transform behaves as an identity function.
        min_bin_width (`float`):
            Minimum bin value across the width dimension for the piecewise rational quadratic function.
        min_bin_height (`float`):
            Minimum bin value across the height dimension for the piecewise rational quadratic function.
        min_derivative (`float`):
            Minimum bin value across the derivatives for the piecewise rational quadratic function.
    Returns:
        outputs (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Hidden-states as transformed by the piecewise rational quadratic function.
        log_abs_det (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Logarithm of the absolute value of the determinants corresponding to the `outputs`.
    """
    upper_bound = tail_bound
    lower_bound = -tail_bound
    if torch.min(inputs) < lower_bound or torch.max(inputs) > upper_bound:
        raise ValueError("Input to a transform is not within its domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError(f"Minimal bin width {min_bin_width} too large for the number of bins {num_bins}")
    if min_bin_height * num_bins > 1.0:
        raise ValueError(f"Minimal bin height {min_bin_height} too large for the number of bins {num_bins}")

    widths = nn.functional.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = nn.functional.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (upper_bound - lower_bound) * cumwidths + lower_bound
    cumwidths[..., 0] = lower_bound
    cumwidths[..., -1] = upper_bound
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + nn.functional.softplus(unnormalized_derivatives)

    heights = nn.functional.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = nn.functional.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (upper_bound - lower_bound) * cumheights + lower_bound
    cumheights[..., 0] = lower_bound
    cumheights[..., -1] = upper_bound
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    bin_locations = cumheights if reverse else cumwidths
    bin_locations[..., -1] += 1e-6
    bin_idx = torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1
    bin_idx = bin_idx[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    intermediate1 = input_derivatives + input_derivatives_plus_one - 2 * input_delta
    if not reverse:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + intermediate1 * theta_one_minus_theta
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        log_abs_det = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, log_abs_det
    else:
        # find the roots of a quadratic equation
        intermediate2 = inputs - input_cumheights
        intermediate3 = intermediate2 * intermediate1
        a = input_heights * (input_delta - input_derivatives) + intermediate3
        b = input_heights * input_derivatives - intermediate3
        c = -input_delta * intermediate2

        discriminant = b.pow(2) - 4 * a * c
        if not (discriminant >= 0).all():
            raise RuntimeError(f"invalid discriminant {discriminant}")

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + intermediate1 * theta_one_minus_theta
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        log_abs_det = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -log_abs_det


class VitsWaveNet(torch.nn.Module):
    """
    Model temporal latent dependencies with gated, dilated convolutions.

    Dilations expand the receptive field without
    requiring a separate recurrent step for every audio frame.

    Specifically, this stack is used in:
        - The posterior encoder of the VITS model.
        - The latent coupling layers of the VITS model.

    """

    def __init__(self, config: VitsConfig, num_layers: int):
        """Build the dilated convolution stack and optional speaker-conditioning layers.

        Each layer produces gated features and residual/skip signals; the optional
        projection lets one speaker embedding condition every layer.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = num_layers
        self.speaker_embedding_size = config.speaker_embedding_size

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.dropout = nn.Dropout(config.wavenet_dropout)

        # Use weight normalization if available, otherwise fall back to the standard implementation.
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm
        else:
            weight_norm = nn.utils.weight_norm

        # If speaker conditioning is used, create a 1x1 convolution to project the speaker embedding
        if config.speaker_embedding_size != 0:
            cond_layer = torch.nn.Conv1d(config.speaker_embedding_size, 2 * config.hidden_size * num_layers, 1)
            self.cond_layer = weight_norm(cond_layer, name="weight")

        # Build the stack of dilated convolution layers with residual and skip connections.
        for i in range(num_layers):
            dilation = config.wavenet_dilation_rate**i
            padding = (config.wavenet_kernel_size * dilation - dilation) // 2
            in_layer = torch.nn.Conv1d(
                in_channels=config.hidden_size,
                out_channels=2 * config.hidden_size,
                kernel_size=config.wavenet_kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < num_layers - 1:
                res_skip_channels = 2 * config.hidden_size
            else:
                res_skip_channels = config.hidden_size

            res_skip_layer = torch.nn.Conv1d(config.hidden_size, res_skip_channels, 1)
            res_skip_layer = weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, inputs, padding_mask, global_conditioning=None):
        """Transform latent features while masking padding and applying optional speaker conditioning.

        Residual updates preserve the input stream while skip outputs accumulate
        information across layers, which is needed to parameterize the latent
        distribution without allowing padded frames to affect valid ones.

        # The forward pass applies the dilated convolutions with residual and skip connections,
        # incorporating speaker conditioning if provided.

        Inputs:
            inputs (torch.Tensor): The input latent features of shape (batch_size, hidden_size, sequence_length).
            padding_mask (torch.Tensor): A mask indicating valid frames, of shape (batch_size, 1, sequence_length).
            global_conditioning (torch.Tensor, optional): Speaker embedding for conditioning, of shape (batch_size, speaker_embedding_size, 1).
        """
        # Initialize the output tensor and a tensor to hold the number of channels for the fused operation.
        outputs = torch.zeros_like(inputs)
        num_channels_tensor = torch.IntTensor([self.hidden_size])

        if global_conditioning is not None:
             # Project the speaker embedding to match the hidden size for conditioning
            global_conditioning = self.cond_layer(global_conditioning)

        # Iterate over each layer in the stack, applying the dilated convolutions with residual and skip connections.
        for i in range(self.num_layers):
            hidden_states = self.in_layers[i](inputs)

            if global_conditioning is not None:
                # Slice the global conditioning to match the current layer's expected size.
                cond_offset = i * 2 * self.hidden_size
                global_states = global_conditioning[:, cond_offset : cond_offset + 2 * self.hidden_size, :]
            else:
                global_states = torch.zeros_like(hidden_states)

            # Apply the fused add-tanh-sigmoid-multiply operation with the hidden states and global conditioning.
            acts = fused_add_tanh_sigmoid_multiply(hidden_states, global_states, num_channels_tensor[0])
            acts = self.dropout(acts)   # Apply dropout for regularization after the fused operation.

            # Split the resulting activations into residual and skip components for the current layer.
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.num_layers - 1:
                # For all but the last layer, split the activations into residual and skip parts.
                res_acts = res_skip_acts[:, : self.hidden_size, :]
                inputs = (inputs + res_acts) * padding_mask
                outputs = outputs + res_skip_acts[:, self.hidden_size :, :] # Add the skip activations to the output for this layer.
            else:
                outputs = outputs + res_skip_acts # Add the skip activations from the last layer to the output.

        return outputs * padding_mask

    def remove_weight_norm(self):
        """Remove weight normalization from the convolution layers for export or inference.

        Training applies this reparameterization for stable optimization; removing it
        restores ordinary convolution weights for inference and checkpoint export.
        """
        if self.speaker_embedding_size != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(layer)

    def apply_weight_norm(self):
        """Apply weight normalization to the convolution layers used by the WaveNet stack.

        This keeps weight magnitudes and directions separately parameterized during
        adversarial training, matching the VITS checkpoint's training structure.
        """
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm
        else:
            weight_norm = nn.utils.weight_norm

        if self.speaker_embedding_size != 0:
            weight_norm(self.cond_layer)
        for layer in self.in_layers:
            weight_norm(layer)
        for layer in self.res_skip_layers:
            weight_norm(layer)


class VitsPosteriorEncoder(nn.Module):
    """Infer a Gaussian latent representation from an observed spectrogram during training.

    The posterior supplies audio-conditioned latents for training the flow and
    reconstruction objectives; inference instead predicts from text alone.

    It uses a wavenet block sandwiched between initial and final convolutions to process the 
    spectrogram and produce the posterior distribution parameters.

    """

    def __init__(self, config: VitsConfig):
        """Create the spectrogram projection, WaveNet encoder, and distribution head.

        The final projection must emit both location and scale parameters so training
        can sample posterior latents and compare them with the text-conditioned prior.
        """
        super().__init__()
        self.out_channels = config.flow_size

        self.conv_pre = nn.Conv1d(config.spectrogram_bins, config.hidden_size, 1)
        self.wavenet = VitsWaveNet(config, num_layers=config.posterior_encoder_num_wavenet_layers)
        self.conv_proj = nn.Conv1d(config.hidden_size, self.out_channels * 2, 1)

    def forward(self, inputs, padding_mask, global_conditioning=None):
        """Sample masked posterior latents and return their mean and log standard deviation.

        Sampling represents uncertainty in the observed-audio latent distribution;
        the returned statistics are also required by the KL divergence loss.

        Args:
            inputs: The input spectrogram tensor of shape (batch, channels, time).
            padding_mask: A mask tensor indicating valid positions in the input.
            global_conditioning: Optional global conditioning tensor for the WaveNet encoder.
        Returns:
            sampled: Sampled latent tensor from the posterior distribution.
            mean: Mean of the posterior distribution.
            log_stddev: Log standard deviation of the posterior distribution.
        """
        inputs = self.conv_pre(inputs) * padding_mask # Apply the initial convolution and mask padded positions.
        inputs = self.wavenet(inputs, padding_mask, global_conditioning) # Pass through the WaveNet encoder with masking and optional global conditioning.
        stats = self.conv_proj(inputs) * padding_mask # Project to mean and log standard deviation and mask padded positions.
        # Split the projected statistics into mean and log standard deviation components.
        # The first half of the channels represents the mean, and the second half represents the log standard deviation.
        mean, log_stddev = torch.split(stats, self.out_channels, dim=1)
        # Sample from the posterior distribution using the reparameterization trick.
        sampled = (mean + torch.randn_like(mean) * torch.exp(log_stddev)) * padding_mask
        return sampled, mean, log_stddev

    def apply_weight_norm(self):
        """Apply weight normalization to the posterior encoder's WaveNet layers.

        This keeps the posterior network's parameterization consistent with the
        generator's other WaveNet flows during training.
        """
        self.wavenet.apply_weight_norm()

    def remove_weight_norm(self):
        """Remove weight normalization from the posterior encoder's WaveNet layers.

        Export and inference use the equivalent ordinary convolution weights rather
        than the training-time weight-normalized parameterization.
        """
        self.wavenet.remove_weight_norm()

    def resize_speaker_embeddings(self, speaker_embedding_size: Optional[int] = None):
        """Replace WaveNet conditioning to support a different speaker-embedding width.

        The projection's input channels must match the new embedding size or
        multi-speaker conditioning cannot be applied to posterior features.
        """
        self.wavenet.speaker_embedding_size = speaker_embedding_size
        hidden_size = self.wavenet.hidden_size
        num_layers = self.wavenet.num_layers

        cond_layer = torch.nn.Conv1d(speaker_embedding_size, 2 * hidden_size * num_layers, 1)
        self.wavenet.cond_layer = nn.utils.weight_norm(cond_layer, name="weight")
        nn.init.kaiming_normal_(self.wavenet.cond_layer.weight)
        if self.wavenet.cond_layer.bias is not None:
            k = math.sqrt(
                self.wavenet.cond_layer.groups
                / (self.wavenet.cond_layer.in_channels * self.wavenet.cond_layer.kernel_size[0])
            )
            nn.init.uniform_(self.wavenet.cond_layer.bias, a=-k, b=k)


class VitsHifiGanDiscriminatorScaleResidualBlock(nn.Module):
    """
    Component of the discriminator that evaluates waveform features at different scales.

    This branch judges broad and local waveform features without reshaping time,
    complementing the period-based branches in the adversarial objective.
    """

    def __init__(self, discriminator_scale_channels, leaky_relu_slope=0.1):
        """Build the grouped-convolution stack for the scale discriminator.

        Progressively wider, strided layers reduce the waveform to a score while
        retaining intermediate maps for generator feature matching.
        """
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        in_channels, out_channels = discriminator_scale_channels[:2]
        # the first convolution layer processes the input waveform with a fixed kernel size and stride
        self.convs = nn.ModuleList([nn.Conv1d(in_channels, out_channels, 15, 1, padding=7)])

        # the groups are progressively increased to capture wider context in the waveform
        groups = 4
        for in_channels, out_channels in zip(discriminator_scale_channels[1:-1], discriminator_scale_channels[2:]):
            # subsequent layers use larger kernels and strides with increasing groups to capture wider context
            self.convs.append(nn.Conv1d(in_channels, out_channels, 41, 4, groups=groups, padding=20))
            groups = groups * 4

        channel_size = discriminator_scale_channels[-1]
        # next layer continues to process the waveform with the current group size and kernel configuration
        self.convs.append(nn.Conv1d(channel_size, channel_size, 41, 4, groups=groups, padding=20))
        # the next layer uses a smaller kernel and stride to refine the waveform features before the final convolution
        self.convs.append(nn.Conv1d(channel_size, channel_size, 5, 1, padding=2))
        # the final convolution reduces the channel dimension to 1 for the discriminator score
        self.final_conv = nn.Conv1d(channel_size, 1, 3, 1, padding=1)

    def apply_weight_norm(self):
        """Apply weight normalization to the scale discriminator's convolutions.

        It stabilizes the discriminator updates that compete with the generator.
        """
        for layer in self.convs:
            nn.utils.weight_norm(layer)
        nn.utils.weight_norm(self.final_conv)

    def remove_weight_norm(self):
        """Remove weight normalization from the scale discriminator's convolutions.

        The exported discriminator uses ordinary convolution parameters.
        """
        for layer in self.convs:
            nn.utils.remove_weight_norm(layer)
        nn.utils.remove_weight_norm(self.final_conv)

    def forward(self, hidden_states):
        """Return waveform scores and intermediate feature maps for adversarial losses.

        The scalar-like scores train real/fake discrimination; saved feature maps
        let the generator match internal real-audio representations as well.
        """
        fmap = []

        # pass the input waveform through each convolution layer, applying leaky ReLU and saving intermediate feature maps
        for conv in self.convs:
            hidden_states = conv(hidden_states)
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            fmap.append(hidden_states)

        # apply the final convolution to produce the discriminator score and save the feature map
        hidden_states = self.final_conv(hidden_states)
        # the output of the final convolution represents the discriminator's assessment of the waveform
        fmap.append(hidden_states)
        # flatten the final feature map to produce a scalar-like score for the discriminator
        hidden_states = torch.flatten(hidden_states, 1, -1)

        return hidden_states, fmap


class VitsHifiGanDiscriminatorPeriodResidualBlock(nn.Module):
    """
    Discriminate periodic waveform patterns after reshaping audio into 2D periods.

    Separate period branches make harmonic and repeating structures visible to the
    adversarial loss, which a single-scale waveform view can miss.

    Essentially this captures repeating structures in the waveform that correspond to the 
    specified period, making it easier for the discriminator to detect harmonic patterns.
    """

    def __init__(self, discriminator_period_channels, period, kernel_size=5, stride=3, leaky_relu_slope=0.1):
        """Build the convolution stack for one waveform period.

        The period determines the time-to-2D reshape; convolutions then score
        patterns across cycles while preserving the period axis.
        """
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope
        self.period = period

        self.convs = nn.ModuleList()
        # loop over channel pairs to create the convolutional layers for the period discriminator
        # each convolution operates on the 2D period representation, capturing patterns along the time axis while preserving the period axis
        for in_channels, out_channels in zip(discriminator_period_channels[:-1], discriminator_period_channels[1:]):
            self.convs.append(
                nn.Conv2d(in_channels, out_channels,
                    (kernel_size, 1), (stride, 1),
                    padding=(self.get_padding(kernel_size, 1), 0),
                )
            )

        channel_size = discriminator_period_channels[-1]
        # add a final convolution to further process the last feature map before the output
        self.convs.append(
            nn.Conv2d(channel_size, channel_size, (kernel_size, 1), 1, padding=(self.get_padding(kernel_size, 1), 0))
        )
        # the final convolution produces a single-channel score map for the period discriminator
        self.final_conv = nn.Conv2d(channel_size, 1, (3, 1), 1, padding=(1, 0))

    def get_padding(self, kernel_size, dilation=1):
        """
        Compute symmetric padding that preserves length for the given kernel and dilation.

        Preserving the temporal dimension keeps successive discriminator feature maps
        aligned with the period layout.

        """
        return (kernel_size * dilation - dilation) // 2

    def apply_weight_norm(self):
        """Apply weight normalization to the period discriminator's convolutions.

        This supports stable adversarial updates across the discriminator branches.
        """
        for layer in self.convs:
            nn.utils.weight_norm(layer)
        nn.utils.weight_norm(self.final_conv)

    def remove_weight_norm(self):
        """Remove weight normalization from the period discriminator's convolutions.

        This converts the training parameterization back to ordinary convolutions.
        """
        for layer in self.convs:
            nn.utils.remove_weight_norm(layer)
        nn.utils.remove_weight_norm(self.final_conv)

    def forward(self, hidden_states):
        """Return period-based waveform scores and intermediate feature maps.

        Padding makes the waveform divisible by the period; scores and maps then
        provide both adversarial and feature-matching signals.
        """
        fmap = []

        # from 1D to 2D
        batch_size, channels, length = hidden_states.shape
        if length % self.period != 0:
            # pad first
            n_pad = self.period - (length % self.period)
            # pad the waveform so its length is divisible by the period
            hidden_states = nn.functional.pad(hidden_states, (0, n_pad), "reflect")
            # update the length after padding
            length = length + n_pad
        # reshape the waveform into 2D with the period as one dimension
        hidden_states = hidden_states.view(batch_size, channels, length // self.period, self.period)

        # apply each convolution in the period discriminator, capturing features at multiple scales
        for conv in self.convs:
            hidden_states = conv(hidden_states)
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            fmap.append(hidden_states)

        # apply the final convolution to produce the last feature map before flattening
        hidden_states = self.final_conv(hidden_states)
        fmap.append(hidden_states)
        # flatten the final feature map to produce a scalar-like score for the discriminator
        hidden_states = hidden_states.flatten(1, -1)
        hidden_states = torch.flatten(hidden_states, 1, -1)

        return hidden_states, fmap


# Copied from transformers.models.speecht5.modeling_speecht5.HifiGanResidualBlock
class HifiGanResidualBlock(nn.Module):
    """
    Refine HiFi-GAN decoder features through stacked dilated residual convolutions.

    Different dilation rates capture context at multiple time scales after each
    upsampling stage, improving waveform detail without changing sequence length.

    The residual connections ensure that the original sequence information is preserved 
    while allowing the convolutions to refine and expand the feature representations.
    These residual connections are within-block, meaning they connect the input of the block 
    to its output after the convolutions, ensuring that the original signal is preserved 
    while the convolutions refine the features.

    This is useful for improving gradient flow and stability during training, as well as 
    enhancing the quality of the generated waveform by providing additional model capacity 
    for capturing complex temporal dependencies.
    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        """
        Create paired dilated and non-dilated convolutions for each residual stage.

        The paired layers process multiple receptive fields while residual additions
        keep the block trainable at the decoder's high temporal resolution.

        The two convolution stacks allow the block to capture both local and broader temporal 
        dependencies while maintaining the original sequence length through residual connections.
        """
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        # first set of convolutions with varying dilation rates
        # these are used to capture context at multiple time scales through different dilation rates
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(channels, channels, kernel_size, stride=1,
                    dilation=dilation[i],
                    padding=self.get_padding(kernel_size, dilation[i]),
                )
                for i in range(len(dilation))
            ]
        )
        # second set of convolutions with fixed dilation of 1
        # these are used to refine the features captured by the first set without changing the temporal resolution
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels, channels,  kernel_size, stride=1, dilation=1,
                    padding=self.get_padding(kernel_size, 1),
                )
                for _ in range(len(dilation))
            ]
        )

    def get_padding(self, kernel_size, dilation=1):
        """Compute symmetric padding that preserves length for the given kernel and dilation.

        The residual path can be added only when both convolution paths have matching
        time dimensions.
        """
        return (kernel_size * dilation - dilation) // 2

    def apply_weight_norm(self):
        """Apply weight normalization to both convolution stacks.

        This is the decoder's training-time parameterization for stable GAN updates.
        """
        for layer in self.convs1:
            nn.utils.weight_norm(layer)
        for layer in self.convs2:
            nn.utils.weight_norm(layer)

    def remove_weight_norm(self):
        """Remove weight normalization from both convolution stacks.

        The conversion is needed before saving or using the decoder without training
        reparameterizations.
        """
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)

    def forward(self, hidden_states):
        """
        Apply each convolution pair with a residual connection.

        Residual additions preserve the upsampled signal while the dilated filters
        refine local waveform structure.
        """

        # loop through the convolution pairs and apply them with residual connections
        for conv1, conv2 in zip(self.convs1, self.convs2):
             # save the input for the residual connection
            residual = hidden_states
            # apply the first convolution with leaky ReLU activation
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv1(hidden_states)
            # apply the second convolution with leaky ReLU activation and add the residual connection
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv2(hidden_states)
            # add the residual connection to preserve the original input information
            hidden_states = hidden_states + residual
        return hidden_states


class VitsHifiGan(nn.Module):
    """
    Convert latent spectrogram features into waveform audio with a HiFi-GAN decoder.

    VITS predicts latent spectrogram frames; this module is the required vocoder stage
    that turns those frames into the time-domain waveform used for listening and GAN loss.
    """

    def __init__(self, config: VitsConfig):
        """Build the upsampling stages and multi-kernel residual blocks from the VITS config.

        Transposed convolutions restore the audio sample rate, and parallel residual
        kernels reduce artifacts across different waveform patterns.
        """
        super().__init__()
        self.config = config
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = nn.Conv1d( config.flow_size, config.upsample_initial_channel,  kernel_size=7, stride=1,  padding=3)

        self.upsampler = nn.ModuleList()
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.upsampler.append(
                nn.ConvTranspose1d(
                    config.upsample_initial_channel // (2**i),
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=kernel_size,
                    stride=upsample_rate,
                    padding=(kernel_size - upsample_rate) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))

        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3, bias=False)

        if config.speaker_embedding_size != 0:
            self.cond = nn.Conv1d(config.speaker_embedding_size, config.upsample_initial_channel, 1)

    def resize_speaker_embedding(self, speaker_embedding_size):
        """Replace the decoder's speaker-conditioning projection for a new embedding width.

        The conditioning layer must accept the resized speaker vector so the decoder
        can continue producing speaker-specific audio.
        """
        self.config.speaker_embedding_size = speaker_embedding_size
        self.cond = nn.Conv1d(speaker_embedding_size, self.config.upsample_initial_channel, 1)
        nn.init.kaiming_normal_(self.cond.weight)
        if self.cond.bias is not None:
            k = math.sqrt(self.cond.groups / (self.cond.in_channels * self.cond.kernel_size[0]))
            nn.init.uniform_(self.cond.bias, a=-k, b=k)

    def apply_weight_norm(self):
        """Apply weight normalization to the upsampler and residual blocks for training.

        This matches the decoder's GAN-training parameterization and helps control
        optimization of its transposed convolutions.
        """
        for layer in self.upsampler:
            nn.utils.weight_norm(layer)
        for layer in self.resblocks:
            layer.apply_weight_norm()

    def remove_weight_norm(self):
        """Remove weight normalization from the upsampler and residual blocks for export.

        This leaves the same function represented by ordinary weights for deployment.
        """
        for layer in self.upsampler:
            nn.utils.remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()

    def forward(
        self, spectrogram: torch.FloatTensor, global_conditioning: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        r"""
        Converts a spectrogram into a speech waveform.

        Args:
            spectrogram (`torch.FloatTensor` of shape `(batch_size, config.spectrogram_bins, sequence_length)`):
                Tensor containing the spectrograms.
            global_conditioning (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_size, 1)`, *optional*):
                Tensor containing speaker embeddings, for multispeaker models.

        Returns:
            `torch.FloatTensor`: Tensor of shape shape `(batch_size, 1, num_frames)` containing the speech waveform.
        """
        hidden_states = self.conv_pre(spectrogram)

        if global_conditioning is not None:
            hidden_states = hidden_states + self.cond(global_conditioning)

        for i in range(self.num_upsamples):
            hidden_states = nn.functional.leaky_relu(hidden_states, self.config.leaky_relu_slope)
            hidden_states = self.upsampler[i](hidden_states)

            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            hidden_states = res_state / self.num_kernels

        hidden_states = nn.functional.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        waveform = torch.tanh(hidden_states)
        return waveform


class VitsResidualCouplingLayer(nn.Module):
    """Transform half of a latent tensor conditioned on the other half.

    Coupling layers let the prior flow reshape a simple latent distribution into the
    learned speech distribution while retaining an inexpensive, exactly invertible map.
    """

    def __init__(self, config: VitsConfig):
        """Create the conditioning network used by the affine coupling transform.

        The network predicts the shift applied to one channel half from the unchanged
        half, which makes the transformation reversible without solving an equation.
        """
        super().__init__()
        self.half_channels = config.flow_size // 2

        self.conv_pre = nn.Conv1d(self.half_channels, config.hidden_size, 1)
        self.wavenet = VitsWaveNet(config, num_layers=config.prior_encoder_num_wavenet_layers)
        self.conv_post = nn.Conv1d(config.hidden_size, self.half_channels, 1)

    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        """Apply the coupling transform or its inverse, preserving masked positions.

        The forward direction maps latents through the prior flow; reverse mode
        synthesizes latents from that prior. The mask prevents padded frames from
        changing either result.
        """
        first_half, second_half = torch.split(inputs, [self.half_channels] * 2, dim=1)
        hidden_states = self.conv_pre(first_half) * padding_mask
        hidden_states = self.wavenet(hidden_states, padding_mask, global_conditioning)
        mean = self.conv_post(hidden_states) * padding_mask
        log_stddev = torch.zeros_like(mean)

        if not reverse:
            second_half = mean + second_half * torch.exp(log_stddev) * padding_mask
            outputs = torch.cat([first_half, second_half], dim=1)
            log_determinant = torch.sum(log_stddev, [1, 2])
            return outputs, log_determinant
        else:
            second_half = (second_half - mean) * torch.exp(-log_stddev) * padding_mask
            outputs = torch.cat([first_half, second_half], dim=1)
            return outputs, None

    def apply_weight_norm(self):
        """Apply weight normalization to the coupling layer's convolutional networks.

        The same parameterization is used by the pretrained flow during training.
        """
        nn.utils.weight_norm(self.conv_pre)
        self.wavenet.apply_weight_norm()
        nn.utils.weight_norm(self.conv_post)

    def remove_weight_norm(self):
        """Remove weight normalization from the coupling layer's convolutional networks.

        Removing it folds the reparameterization into ordinary weights for export.
        """
        nn.utils.remove_weight_norm(self.conv_pre)
        self.wavenet.remove_weight_norm()
        nn.utils.remove_weight_norm(self.conv_post)


class VitsResidualCouplingBlock(nn.Module):
    """Stack coupling layers and channel flips to form the prior latent flow.

    A single coupling layer only transforms one half conditioned on the other;
    alternating channel order lets every latent channel influence later transforms.
    """

    def __init__(self, config: VitsConfig):
        """Create the configured sequence of residual coupling layers.

        The number of stages controls the flow's capacity to match the prior and
        posterior latent distributions.
        """
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(config.prior_encoder_num_flows):
            self.flows.append(VitsResidualCouplingLayer(config))

    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        """Apply all coupling layers in forward or reverse order.

        Reversing both layer order and channel flips is essential for an exact inverse
        when decoding prior latents back into the posterior latent space.
        """
        if not reverse:
            for flow in self.flows:
                inputs, _ = flow(inputs, padding_mask, global_conditioning)
                inputs = torch.flip(inputs, [1])
        else:
            for flow in reversed(self.flows):
                inputs = torch.flip(inputs, [1])
                inputs, _ = flow(inputs, padding_mask, global_conditioning, reverse=True)
        return inputs

    def apply_weight_norm(self):
        """Apply weight normalization throughout the latent flow.

        This keeps every coupling transform in the same training-time parameterization.
        """
        for flow in self.flows:
            flow.apply_weight_norm()

    def remove_weight_norm(self):
        """Remove weight normalization throughout the latent flow.

        All nested layers must be converted so the exported flow can run consistently.
        """
        for flow in self.flows:
            flow.remove_weight_norm()

    def resize_speaker_embeddings(self, speaker_embedding_size: Optional[int] = None):
        """Replace each coupling layer's speaker-conditioning projection.

        Updating every layer is necessary because each flow stage consumes the speaker
        vector when conditioning its latent transformation.
        """
        for flow in self.flows:
            flow.wavenet.speaker_embedding_size = speaker_embedding_size
            hidden_size = flow.wavenet.hidden_size
            num_layers = flow.wavenet.num_layers

            cond_layer = torch.nn.Conv1d(speaker_embedding_size, 2 * hidden_size * num_layers, 1)
            flow.wavenet.cond_layer = nn.utils.weight_norm(cond_layer, name="weight")


class VitsDilatedDepthSeparableConv(nn.Module):
    """Extract contextual features with masked dilated depthwise-separable convolutions.

    These blocks expand the duration predictor's temporal context at lower cost than
    full convolutions, while residual connections preserve information across layers.
    """

    def __init__(self, config: VitsConfig, dropout_rate=0.0):
        """Build the dilated depthwise, pointwise, normalization, and dropout layers.

        The depthwise layers model local temporal patterns; pointwise layers mix
        channels so the predictor can learn interactions between those patterns.
        """
        super().__init__()
        kernel_size = config.duration_predictor_kernel_size
        channels = config.hidden_size
        self.num_layers = config.depth_separable_num_layers

        self.dropout = nn.Dropout(dropout_rate)
        self.convs_dilated = nn.ModuleList()
        self.convs_pointwise = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(self.num_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_dilated.append(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.convs_pointwise.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(nn.LayerNorm(channels))
            self.norms_2.append(nn.LayerNorm(channels))

    def forward(self, inputs, padding_mask, global_conditioning=None):
        """Apply residual convolution blocks while zeroing padded positions.

        Optional speaker conditioning is added before the stack, and masking prevents
        padding from leaking through convolutions or residual updates.
        """
        if global_conditioning is not None:
            inputs = inputs + global_conditioning

        for i in range(self.num_layers):
            hidden_states = self.convs_dilated[i](inputs * padding_mask)
            hidden_states = self.norms_1[i](hidden_states.transpose(1, -1)).transpose(1, -1)
            hidden_states = nn.functional.gelu(hidden_states)
            hidden_states = self.convs_pointwise[i](hidden_states)
            hidden_states = self.norms_2[i](hidden_states.transpose(1, -1)).transpose(1, -1)
            hidden_states = nn.functional.gelu(hidden_states)
            hidden_states = self.dropout(hidden_states)
            inputs = inputs + hidden_states

        return inputs * padding_mask


class VitsConvFlow(nn.Module):
    """Apply a conditional spline coupling transform in the duration predictor's flow.

    The conditional network predicts spline parameters, allowing the stochastic
    duration model to represent non-Gaussian distributions while retaining an exact
    inverse and tractable likelihood.
    """

    def __init__(self, config: VitsConfig):
        """Create the network that predicts spline parameters for the coupling transform.

        Its output is divided into bin widths, heights, and derivatives that define
        the data-dependent rational-quadratic transformation.
        """
        super().__init__()
        self.filter_channels = config.hidden_size
        self.half_channels = config.depth_separable_channels // 2
        self.num_bins = config.duration_predictor_flow_bins
        self.tail_bound = config.duration_predictor_tail_bound

        self.conv_pre = nn.Conv1d(self.half_channels, self.filter_channels, 1)
        self.conv_dds = VitsDilatedDepthSeparableConv(config)
        self.conv_proj = nn.Conv1d(self.filter_channels, self.half_channels * (self.num_bins * 3 - 1), 1)

    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        """Transform one channel half and return its log-determinant for likelihood computation.

        Conditioning comes from text and, when present, the speaker. The Jacobian term
        is needed for correct change-of-variables probabilities during flow training.
        """
        first_half, second_half = torch.split(inputs, [self.half_channels] * 2, dim=1)

        hidden_states = self.conv_pre(first_half)
        hidden_states = self.conv_dds(hidden_states, padding_mask, global_conditioning)
        hidden_states = self.conv_proj(hidden_states) * padding_mask

        batch_size, channels, length = first_half.shape
        hidden_states = hidden_states.reshape(batch_size, channels, -1, length).permute(0, 1, 3, 2)

        unnormalized_widths = hidden_states[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = hidden_states[..., self.num_bins : 2 * self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_derivatives = hidden_states[..., 2 * self.num_bins :]

        second_half, log_abs_det = _unconstrained_rational_quadratic_spline(
            second_half,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            reverse=reverse,
            tail_bound=self.tail_bound,
        )

        outputs = torch.cat([first_half, second_half], dim=1) * padding_mask
        if not reverse:
            log_determinant = torch.sum(log_abs_det * padding_mask, [1, 2])
            return outputs, log_determinant
        else:
            return outputs, None


class VitsElementwiseAffine(nn.Module):
    """Apply a learned per-channel affine transform within a normalizing flow.

    This simple invertible layer provides trainable translation and scale before or
    between more expressive spline transforms.
    """

    def __init__(self, config: VitsConfig):
        """Initialize trainable channel-wise translation and log-scale parameters.

        Log-scale parameters make the multiplicative factor positive and yield a
        simple analytic inverse and log-determinant.
        """
        super().__init__()
        self.channels = config.depth_separable_channels
        self.translate = nn.Parameter(torch.zeros(self.channels, 1))
        self.log_scale = nn.Parameter(torch.zeros(self.channels, 1))

    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        """Apply the affine transform or its inverse and return the forward log-determinant.

        The determinant contribution is required to evaluate flow likelihoods; reverse
        mode maps sampled latents back without needing that training correction.
        """
        if not reverse:
            outputs = self.translate + torch.exp(self.log_scale) * inputs
            outputs = outputs * padding_mask
            log_determinant = torch.sum(self.log_scale * padding_mask, [1, 2])
            return outputs, log_determinant
        else:
            outputs = (inputs - self.translate) * torch.exp(-self.log_scale) * padding_mask
            return outputs, None


class VitsStochasticDurationPredictor(nn.Module):
    """Model token durations with conditional normalizing flows and posterior inference.

    Training estimates duration likelihood using a learned posterior flow, while
    inference samples the prior flow. This captures duration uncertainty that a
    point-estimate predictor cannot represent.
    """

    def __init__(self, config):
        """Build the duration networks and paired prior/posterior flow stacks.

        The posterior stack is used to infer latent variables from observed durations;
        the prior stack converts noise into duration predictions conditioned on text.
        """
        super().__init__()
        embed_dim = config.speaker_embedding_size
        filter_channels = config.hidden_size

        self.conv_pre = nn.Conv1d(filter_channels, filter_channels, 1)
        self.conv_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.conv_dds = VitsDilatedDepthSeparableConv(
            config,
            dropout_rate=config.duration_predictor_dropout,
        )

        if embed_dim != 0:
            self.cond = nn.Conv1d(embed_dim, filter_channels, 1)

        self.flows = nn.ModuleList()
        self.flows.append(VitsElementwiseAffine(config))
        for _ in range(config.duration_predictor_num_flows):
            self.flows.append(VitsConvFlow(config))

        self.post_conv_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_conv_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_conv_dds = VitsDilatedDepthSeparableConv(
            config,
            dropout_rate=config.duration_predictor_dropout,
        )

        self.post_flows = nn.ModuleList()
        self.post_flows.append(VitsElementwiseAffine(config))
        for _ in range(config.duration_predictor_num_flows):
            self.post_flows.append(VitsConvFlow(config))

        self.filter_channels = filter_channels

    def resize_speaker_embeddings(self, speaker_embedding_size):
        """Replace the predictor's speaker-conditioning projection.

        Its input width must match the new speaker embedding for duration estimates to
        remain speaker-conditioned.
        """
        self.cond = nn.Conv1d(speaker_embedding_size, self.filter_channels, 1)

    def forward(self, inputs, padding_mask, global_conditioning=None, durations=None, reverse=False, noise_scale=1.0):
        """Compute duration negative log-likelihood or sample log durations from noise.

        Training mode combines prior and posterior flow densities into a likelihood
        objective; reverse mode samples the latent flow and returns per-token
        log-duration values for synthesis.
        """
        inputs = torch.detach(inputs)
        inputs = self.conv_pre(inputs)

        if global_conditioning is not None:
            global_conditioning = torch.detach(global_conditioning)
            inputs = inputs + self.cond(global_conditioning)

        inputs = self.conv_dds(inputs, padding_mask)
        inputs = self.conv_proj(inputs) * padding_mask

        if not reverse:
            hidden_states = self.post_conv_pre(durations)
            hidden_states = self.post_conv_dds(hidden_states, padding_mask)
            hidden_states = self.post_conv_proj(hidden_states) * padding_mask

            random_posterior = (
                torch.randn(durations.size(0), 2, durations.size(2)).to(device=inputs.device, dtype=inputs.dtype)
                * padding_mask
            )
            latents_posterior = random_posterior

            latents_posterior, log_determinant = self.post_flows[0](
                latents_posterior, padding_mask, global_conditioning=inputs + hidden_states
            )
            log_determinant_posterior_sum = log_determinant

            for flow in self.post_flows[1:]:
                latents_posterior, log_determinant = flow(
                    latents_posterior, padding_mask, global_conditioning=inputs + hidden_states
                )
                latents_posterior = torch.flip(latents_posterior, [1])
                log_determinant_posterior_sum += log_determinant

            first_half, second_half = torch.split(latents_posterior, [1, 1], dim=1)

            log_determinant_posterior_sum += torch.sum(
                (nn.functional.logsigmoid(first_half) + nn.functional.logsigmoid(-first_half)) * padding_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (random_posterior**2)) * padding_mask, [1, 2])
                - log_determinant_posterior_sum
            )

            first_half = (durations - torch.sigmoid(first_half)) * padding_mask
            first_half = torch.log(torch.clamp_min(first_half, 1e-5)) * padding_mask
            log_determinant_sum = torch.sum(-first_half, [1, 2])

            latents = torch.cat([first_half, second_half], dim=1)
            latents, log_determinant = self.flows[0](latents, padding_mask, global_conditioning=inputs)

            log_determinant_sum += log_determinant
            for flow in self.flows[1:]:
                latents, log_determinant = flow(latents, padding_mask, global_conditioning=inputs)
                latents = torch.flip(latents, [1])
                log_determinant_sum += log_determinant

            nll = torch.sum(0.5 * (math.log(2 * math.pi) + (latents**2)) * padding_mask, [1, 2]) - log_determinant_sum
            return nll + logq
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow

            latents = (
                torch.randn(inputs.size(0), 2, inputs.size(2)).to(device=inputs.device, dtype=inputs.dtype)
                * noise_scale
            )
            for flow in flows:
                latents = torch.flip(latents, [1])
                latents, _ = flow(latents, padding_mask, global_conditioning=inputs, reverse=True)

            log_duration, _ = torch.split(latents, [1, 1], dim=1)
            return log_duration


class VitsDurationPredictor(nn.Module):
    """Predict log durations deterministically from text-encoder features.

    This simpler alternative is used when stochastic duration prediction is disabled;
    it provides token lengths needed to expand text states into acoustic frames.
    """

    def __init__(self, config):
        """Create the convolutional duration network and optional speaker conditioning.

        The projections map contextual text features to one duration value per token.
        """
        super().__init__()
        kernel_size = config.duration_predictor_kernel_size
        filter_channels = config.duration_predictor_filter_channels

        self.dropout = nn.Dropout(config.duration_predictor_dropout)
        self.conv_1 = nn.Conv1d(config.hidden_size, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = nn.LayerNorm(filter_channels, eps=config.layer_norm_eps)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = nn.LayerNorm(filter_channels, eps=config.layer_norm_eps)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if config.speaker_embedding_size != 0:
            self.cond = nn.Conv1d(config.speaker_embedding_size, config.hidden_size, 1)

        self.hidden_size = config.hidden_size

    def resize_speaker_embeddings(self, speaker_embedding_size):
        """Replace the predictor's speaker-conditioning projection.

        The projection must accept the configured speaker embedding width in
        multi-speaker synthesis.
        """
        self.cond = nn.Conv1d(speaker_embedding_size, self.hidden_size, 1)

    def forward(self, inputs, padding_mask, global_conditioning=None):
        """Return masked per-token log-duration predictions.

        The model exponentiates these values to obtain token durations, so padding is
        masked here to prevent nonexistent tokens from extending the generated audio.
        """
        inputs = torch.detach(inputs)

        if global_conditioning is not None:
            global_conditioning = torch.detach(global_conditioning)
            inputs = inputs + self.cond(global_conditioning)

        inputs = self.conv_1(inputs * padding_mask)
        inputs = torch.relu(inputs)
        inputs = self.norm_1(inputs.transpose(1, -1)).transpose(1, -1)
        inputs = self.dropout(inputs)

        inputs = self.conv_2(inputs * padding_mask)
        inputs = torch.relu(inputs)
        inputs = self.norm_2(inputs.transpose(1, -1)).transpose(1, -1)
        inputs = self.dropout(inputs)

        inputs = self.proj(inputs * padding_mask)
        return inputs * padding_mask


class VitsAttention(nn.Module):
    """Multi-headed self-attention with learned relative positional representations.

    Relative positions help the text encoder model phone order and local dependencies
    without requiring a fixed absolute position embedding for each sequence length.
    """

    def __init__(self, config: VitsConfig):
        """Create attention projections and learned relative-position embeddings.

        Head dimensions and projection sizes come from the checkpoint config so
        pretrained attention weights have the expected shapes.
        """
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.window_size = config.window_size

        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (got `hidden_size`: {self.embed_dim}"
                f" and `num_attention_heads`: {self.num_heads})."
            )

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)

        if self.window_size:
            self.emb_rel_k = nn.Parameter(torch.randn(1, self.window_size * 2 + 1, self.head_dim) * self.scaling)
            self.emb_rel_v = nn.Parameter(torch.randn(1, self.window_size * 2 + 1, self.head_dim) * self.scaling)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """Reshape projected features into batch, head, sequence, and head dimensions.

        This layout lets batched matrix multiplication compute each head independently
        while keeping the batch dimension explicit for later reconstruction.
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute multi-head attention with relative position terms and optional masks.

        Attention lets each text position use surrounding token context; masks exclude
        padding, while optional weights support diagnostics and model outputs.
        """

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling

        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if self.window_size is not None:
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, src_len)
            relative_logits = torch.matmul(query_states, key_relative_embeddings.transpose(-2, -1))
            rel_pos_bias = self._relative_position_to_absolute_position(relative_logits)
            attn_weights += rel_pos_bias

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        if self.window_size is not None:
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, src_len)
            relative_weights = self._absolute_position_to_relative_position(attn_probs)
            rel_pos_bias = torch.matmul(relative_weights, value_relative_embeddings)
            attn_output += rel_pos_bias

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped

    def _get_relative_embeddings(self, relative_embeddings, length):
        """Pad and slice relative-position embeddings to cover the current sequence length.

        The centered window must yield exactly the relative offsets used by this
        sequence, including when a sequence is longer than the configured window.
        """
        pad_length = max(length - (self.window_size + 1), 0)
        if pad_length > 0:
            relative_embeddings = nn.functional.pad(relative_embeddings, [0, 0, pad_length, pad_length, 0, 0])

        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        return relative_embeddings[:, slice_start_position:slice_end_position]

    def _relative_position_to_absolute_position(self, x):
        """Convert relative-position attention logits into absolute key-position indexing.

        The conversion aligns each relative offset with the key column it represents
        so the logits can be added to ordinary query-key scores.
        """
        batch_heads, length, _ = x.size()

        # Concat columns of pad to shift from relative to absolute indexing.
        x = nn.functional.pad(x, [0, 1, 0, 0, 0, 0])

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch_heads, length * 2 * length])
        x_flat = nn.functional.pad(x_flat, [0, length - 1, 0, 0])

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch_heads, length + 1, 2 * length - 1])
        x_final = x_final[:, :length, length - 1 :]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """Reindex absolute attention probabilities for relative-position value embeddings.

        This inverse layout supplies the relative weights needed to add position-aware
        value contributions to the attended features.
        """
        batch_heads, length, _ = x.size()

        # Pad along column
        x = nn.functional.pad(x, [0, length - 1, 0, 0, 0, 0])
        x_flat = x.view([batch_heads, length**2 + length * (length - 1)])

        # Add 0's in the beginning that will skew the elements after reshape
        x_flat = nn.functional.pad(x_flat, [length, 0, 0, 0])
        x_final = x_flat.view([batch_heads, length, 2 * length])[:, :, 1:]
        return x_final


class VitsFeedForward(nn.Module):
    """Apply the encoder's position-wise convolutional feed-forward network.

    The convolutional layers mix nearby token features after attention, adding
    nonlinear capacity needed to form text-conditioned latent statistics.
    """

    def __init__(self, config):
        """Build the two convolution layers, activation, and dropout from the config.

        Configured padding keeps the sequence length stable, allowing residual addition
        in the enclosing encoder layer.
        """
        super().__init__()
        self.conv_1 = nn.Conv1d(config.hidden_size, config.ffn_dim, config.ffn_kernel_size)
        self.conv_2 = nn.Conv1d(config.ffn_dim, config.hidden_size, config.ffn_kernel_size)
        self.dropout = nn.Dropout(config.activation_dropout)

        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

        if config.ffn_kernel_size > 1:
            pad_left = (config.ffn_kernel_size - 1) // 2
            pad_right = config.ffn_kernel_size // 2
            self.padding = [pad_left, pad_right, 0, 0, 0, 0]
        else:
            self.padding = None

    def forward(self, hidden_states, padding_mask):
        """Transform each sequence position while masking padding around convolutions.

        Padding masks are reapplied after convolutions so their receptive fields cannot
        propagate padded values into valid token representations.
        """
        hidden_states = hidden_states.permute(0, 2, 1)
        padding_mask = padding_mask.permute(0, 2, 1)

        hidden_states = hidden_states * padding_mask
        if self.padding is not None:
            hidden_states = nn.functional.pad(hidden_states, self.padding)

        hidden_states = self.conv_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states * padding_mask
        if self.padding is not None:
            hidden_states = nn.functional.pad(hidden_states, self.padding)

        hidden_states = self.conv_2(hidden_states)
        hidden_states = hidden_states * padding_mask

        hidden_states = hidden_states.permute(0, 2, 1)
        return hidden_states


class VitsEncoderLayer(nn.Module):
    """Combine relative self-attention and a convolutional feed-forward residual block.

    Both sublayers use residual connections and normalization so the stack can learn
    contextual token representations while remaining stable at depth.
    """

    def __init__(self, config: VitsConfig):
        """Create the attention, feed-forward, dropout, and layer-normalization modules.

        These components match the VITS text encoder's two-stage transformer block.
        """
        super().__init__()
        self.attention = VitsAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = VitsFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        """Apply attention and feed-forward residual updates, optionally returning attention weights.

        The output retains the input sequence shape for stacking; optional attention
        maps are included only when requested to avoid unnecessary storage.
        """
        residual = hidden_states
        hidden_states, attn_weights = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(residual + hidden_states)

        residual = hidden_states
        hidden_states = self.feed_forward(hidden_states, padding_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.final_layer_norm(residual + hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class VitsEncoder(nn.Module):
    """Run a stack of VITS encoder layers with padding masks and optional layer outputs.

    It centralizes layerdrop, gradient checkpointing, and output collection so the text
    encoder can train deep contextual representations efficiently.
    """

    def __init__(self, config: VitsConfig):
        """Build the configured encoder layers and initialize training-time controls.

        Layer count and layerdrop are checkpoint settings that affect both model shape
        and regularization behavior.
        """
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([VitsEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        self.layerdrop = config.layerdrop

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        padding_mask: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """Encode token features and optionally collect hidden states and attention maps.

        Expanded attention masks and padding masks keep padded positions excluded;
        optional outputs support diagnostics without burdening ordinary training.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = eager_mask(
                batch_size=hidden_states.shape[0],
                q_length=hidden_states.shape[1],
                kv_length=hidden_states.shape[1],
                mask_function=bidirectional_mask_function,
                attention_mask=attention_mask.to(torch.bool),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        hidden_states = hidden_states * padding_mask

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = self.training and (dropout_probability < self.layerdrop)
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        padding_mask,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        padding_mask=padding_mask,
                        output_attentions=output_attentions,
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = hidden_states * padding_mask

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class VitsTextEncoder(nn.Module):
    """Turn token IDs into contextual features and text-conditioned prior statistics.

    The prior mean and log variance are later aligned with audio frames and transformed
    by the latent flow, making this encoder the text-to-latent entry point for VITS.
    """

    def __init__(self, config: VitsConfig):
        """Create token embeddings, the transformer encoder, and prior-distribution projection.

        The projection emits both Gaussian statistics for each token, which the
        alignment and flow stages need to synthesize speech.
        """
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.encoder = VitsEncoder(config)
        self.project = nn.Conv1d(config.hidden_size, config.flow_size * 2, kernel_size=1)

    def get_input_embeddings(self):
        """Return the token embedding table used by the text encoder.

        Transformers utilities use this accessor to inspect or resize the vocabulary
        embeddings without depending on the encoder's internal attribute name.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """Replace the token embedding table, for example after resizing the vocabulary.

        Keeping this setter paired with the getter allows pretrained-model utilities
        to install resized embeddings consistently.
        """
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], VitsTextEncoderOutput]:
        """Encode token IDs and project states into prior means and log variances.

        Those Gaussian parameters define the text-conditioned prior that is expanded
        over acoustic frames using the monotonic alignment during training.
        """
        hidden_states = self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)

        encoder_outputs = self.encoder(
            hidden_states=hidden_states,
            padding_mask=padding_mask,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state

        stats = self.project(last_hidden_state.transpose(1, 2)).transpose(1, 2) * padding_mask
        prior_means, prior_log_variances = torch.split(stats, self.config.flow_size, dim=2)

        if not return_dict:
            outputs = (last_hidden_state, prior_means, prior_log_variances) + encoder_outputs[1:]
            return outputs

        return VitsTextEncoderOutput(
            last_hidden_state=last_hidden_state,
            prior_means=prior_means,
            prior_log_variances=prior_log_variances,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class VitsPreTrainedModel(PreTrainedModel):
    """
    Base class for VITS models with checkpoint-aware configuration and weight handling.

    It connects the architecture to Transformers' `from_pretrained` and save APIs,
    while providing the initialization rules used when layers are newly created.
    """

    config_class = VitsConfig
    base_model_prefix = "vits"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize a module using the parameter rules expected by VITS.

        Applying the same initialization to newly created or resized layers keeps
        their scale compatible with the pretrained network.
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


VITS_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`VitsConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


VITS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        speaker_id (`int` or `Tuple[int,..]` or `List[int]` or `np.ndarray`, *optional*):
            Which speaker embedding to use. Only used for multispeaker models. Only used for multi-speaker models. You
            can use a single `speaker_id` or a list/tuple/numpy array, provided it is as long as the batch size.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The complete VITS model, for text-to-speech synthesis.",
    VITS_START_DOCSTRING,
)
class VitsModel(VitsPreTrainedModel):
    """Generate speech from token IDs using the VITS text, duration, flow, and decoder components.

    This inference-facing model predicts durations and latent features from text,
    expands them into a spectrogram, and decodes that spectrogram into waveform audio.
    """

    def __init__(self, config: VitsConfig):
        """Assemble the text encoder, latent flow, duration predictor, and waveform decoder.

        The checkpoint config selects deterministic versus stochastic durations and
        defines the dimensions required to load pretrained weights.
        """
        super().__init__(config)
        self.config = config
        self.text_encoder = VitsTextEncoder(config)
        self.flow = VitsResidualCouplingBlock(config)
        self.decoder = VitsHifiGan(config)

        if config.use_stochastic_duration_prediction:
            self.duration_predictor = VitsStochasticDurationPredictor(config)
        else:
            self.duration_predictor = VitsDurationPredictor(config)

        if config.num_speakers > 1:
            self.embed_speaker = nn.Embedding(config.num_speakers, config.speaker_embedding_size)

        # This is used only for training.
        self.posterior_encoder = VitsPosteriorEncoder(config)

        # These parameters control the synthesised speech properties
        self.speaking_rate = config.speaking_rate
        self.noise_scale = config.noise_scale
        self.noise_scale_duration = config.noise_scale_duration

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        """Return the text encoder for model integrations that expect an encoder accessor.

        Exposing it through the standard model API lets generation and Transformers
        utilities locate the source of token representations.
        """
        return self.text_encoder

    @add_start_docstrings_to_model_forward(VITS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=VitsModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        speaker_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[Any], VitsModelOutput]:
        r"""
        labels (`torch.FloatTensor` of shape `(batch_size, config.spectrogram_bins, sequence_length)`, *optional*):
            Float values of target spectrogram. Timesteps set to `-100.0` are ignored (masked) for the loss
            computation.

        Returns:

        Example:

        ```python
        >>> from transformers import VitsTokenizer, VitsModel, set_seed
        >>> import torch

        >>> tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
        >>> model = VitsModel.from_pretrained("facebook/mms-tts-eng")

        >>> inputs = tokenizer(text="Hello - my dog is cute", return_tensors="pt")

        >>> set_seed(555)  # make deterministic

        >>> with torch.no_grad():
        ...     outputs = model(inputs["input_ids"])
        >>> outputs.waveform.shape
        torch.Size([1, 45824])
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None:
            input_padding_mask = attention_mask.unsqueeze(-1).float()
        else:
            input_padding_mask = torch.ones_like(input_ids).unsqueeze(-1).float()

        if self.config.num_speakers > 1 and speaker_id is not None:
            if isinstance(speaker_id, int):
                speaker_id = torch.full(size=(1,), fill_value=speaker_id, device=self.device)
            elif isinstance(speaker_id, (list, tuple, np.ndarray)):
                speaker_id = torch.tensor(speaker_id, device=self.device)

            if not ((0 <= speaker_id).all() and (speaker_id < self.config.num_speakers).all()).item():
                raise ValueError(f"Set `speaker_id` in the range 0-{self.config.num_speakers - 1}.")
            if not (len(speaker_id) == 1 or len(speaker_id == len(input_ids))):
                raise ValueError(
                    f"You passed {len(speaker_id)} `speaker_id` but you should either pass one speaker id or `batch_size` `speaker_id`."
                )

            speaker_embeddings = self.embed_speaker(speaker_id).unsqueeze(-1)
        else:
            speaker_embeddings = None

        if labels is not None:
            raise NotImplementedError(
                "Training of VITS is not supported by this model. See `VitsModelForPreTraining`."
            )

        text_encoder_output = self.text_encoder(
            input_ids=input_ids,
            padding_mask=input_padding_mask,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = text_encoder_output[0] if not return_dict else text_encoder_output.last_hidden_state
        hidden_states = hidden_states.transpose(1, 2)
        input_padding_mask = input_padding_mask.transpose(1, 2)
        prior_means = text_encoder_output[1] if not return_dict else text_encoder_output.prior_means
        prior_log_variances = text_encoder_output[2] if not return_dict else text_encoder_output.prior_log_variances

        if self.config.use_stochastic_duration_prediction:
            log_duration = self.duration_predictor(
                hidden_states,
                input_padding_mask,
                speaker_embeddings,
                reverse=True,
                noise_scale=self.noise_scale_duration,
            )
        else:
            log_duration = self.duration_predictor(hidden_states, input_padding_mask, speaker_embeddings)

        length_scale = 1.0 / self.speaking_rate
        duration = torch.ceil(torch.exp(log_duration) * input_padding_mask * length_scale)
        predicted_lengths = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()

        # Create a padding mask for the output lengths of shape (batch, 1, max_output_length)
        indices = torch.arange(predicted_lengths.max(), dtype=predicted_lengths.dtype, device=predicted_lengths.device)
        output_padding_mask = indices.unsqueeze(0) < predicted_lengths.unsqueeze(1)
        output_padding_mask = output_padding_mask.unsqueeze(1).to(input_padding_mask.dtype)

        # Reconstruct an attention tensor of shape (batch, 1, out_length, in_length)
        attn_mask = torch.unsqueeze(input_padding_mask, 2) * torch.unsqueeze(output_padding_mask, -1)
        batch_size, _, output_length, input_length = attn_mask.shape
        cum_duration = torch.cumsum(duration, -1).view(batch_size * input_length, 1)
        indices = torch.arange(output_length, dtype=duration.dtype, device=duration.device)
        valid_indices = indices.unsqueeze(0) < cum_duration
        valid_indices = valid_indices.to(attn_mask.dtype).view(batch_size, input_length, output_length)
        padded_indices = valid_indices - nn.functional.pad(valid_indices, [0, 0, 1, 0, 0, 0])[:, :-1]
        attn = padded_indices.unsqueeze(1).transpose(2, 3) * attn_mask

        # Expand prior distribution
        prior_means = torch.matmul(attn.squeeze(1), prior_means).transpose(1, 2)
        prior_log_variances = torch.matmul(attn.squeeze(1), prior_log_variances).transpose(1, 2)

        prior_latents = prior_means + torch.randn_like(prior_means) * torch.exp(prior_log_variances) * self.noise_scale
        latents = self.flow(prior_latents, output_padding_mask, speaker_embeddings, reverse=True)

        spectrogram = latents * output_padding_mask
        waveform = self.decoder(spectrogram, speaker_embeddings)
        waveform = waveform.squeeze(1)
        sequence_lengths = predicted_lengths * np.prod(self.config.upsample_rates)

        if not return_dict:
            outputs = (waveform, sequence_lengths, spectrogram) + text_encoder_output[3:]
            return outputs

        return VitsModelOutput(
            waveform=waveform,
            sequence_lengths=sequence_lengths,
            spectrogram=spectrogram,
            hidden_states=text_encoder_output.hidden_states,
            attentions=text_encoder_output.attentions,
        )


class VitsDiscriminator(VitsPreTrainedModel):
    """Score generated and reference waveforms at multiple scales or temporal periods.

    Complementary branches detect waveform artifacts and supply feature maps for
    adversarial and feature-matching objectives; without them the generator loses
    the training signal that encourages realistic audio.
    """

    def __init__(self, config):
        """Build the configured collection of scale and period discriminators.

        Each period branch specializes in a different repeating pattern, while the
        optional scale branch evaluates the waveform in its original time layout.
        """
        super().__init__(config)

        if config.discriminator_scale_channels is not None:
            self.discriminators = nn.ModuleList(
                [VitsHifiGanDiscriminatorScaleResidualBlock(config.discriminator_scale_channels, config.leaky_relu_slope)]
            )
        else:
            self.discriminators = nn.ModuleList([])
        
        self.discriminators.extend(
            [
                VitsHifiGanDiscriminatorPeriodResidualBlock(
                    config.discriminator_period_channels,
                    period,
                    config.discriminator_kernel_size,
                    config.discriminator_stride,
                    config.leaky_relu_slope,
                )
                for period in config.discriminator_periods
            ]
        )

    def forward(self, hidden_states):
        """Return discriminator scores and intermediate feature maps for each branch.

        Scores drive the real-versus-generated loss; feature maps let the generator
        match discriminator activations on real audio.
        """
        fmaps = []
        discriminated_hidden_states_list = []

        for discriminator in self.discriminators:
            discriminated_hidden_states, fmap = discriminator(hidden_states)
            fmaps.append(fmap)
            discriminated_hidden_states_list.append(discriminated_hidden_states)

        return discriminated_hidden_states_list, fmaps

    def apply_weight_norm(self):
        """Apply weight normalization to every discriminator branch for adversarial training.

        This matches the discriminator parameterization used by the pretrained VITS
        adversarial training setup.
        """
        for disc in self.discriminators:
            disc.apply_weight_norm()

    def remove_weight_norm(self):
        """Remove weight normalization from every discriminator branch before export.

        The discriminator is stored alongside the fine-tuned generator, so it must be
        converted consistently before saving the combined checkpoint.
        """
        for disc in self.discriminators:
            disc.remove_weight_norm()


@add_start_docstrings(
    "VITS model with a discriminator, to train VITS.",
    VITS_START_DOCSTRING,
)
class VitsModelForPreTraining(VitsPreTrainedModel):
    """VITS generator and discriminator with the outputs required for adversarial training.

    Unlike the inference-only model, this class includes the posterior encoder and
    discriminator and exposes aligned latent statistics used by VITS reconstruction,
    KL, duration, and adversarial losses.
    """

    def __init__(self, config: VitsConfig):
        """Assemble inference modules together with posterior and discriminator training modules.

        These additional modules are needed to compare observed audio with the
        text-conditioned prior and to train the generator against a discriminator.
        """
        super().__init__(config)
        self.config = config
        self.text_encoder = VitsTextEncoder(config)
        self.flow = VitsResidualCouplingBlock(config)
        self.decoder = VitsHifiGan(config)

        if config.use_stochastic_duration_prediction:
            self.duration_predictor = VitsStochasticDurationPredictor(config)
        else:
            self.duration_predictor = VitsDurationPredictor(config)

        if config.num_speakers > 1:
            self.embed_speaker = nn.Embedding(config.num_speakers, config.speaker_embedding_size)

        # This is used only for training.
        self.posterior_encoder = VitsPosteriorEncoder(config)
        self.discriminator = VitsDiscriminator(config)

        # These parameters control the synthesised speech properties
        self.speaking_rate = config.speaking_rate
        self.noise_scale = config.noise_scale
        self.noise_scale_duration = config.noise_scale_duration
        self.segment_size = self.config.segment_size // self.config.hop_length

        # Initialize weights and apply final processing
        self.post_init()

    def resize_speaker_embeddings(
        self,
        new_num_speakers: int,
        speaker_embedding_size: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = 2,
    ):
        """Resize speaker embeddings and update every component conditioned on their width.

        All conditional projections must change together; otherwise a resized
        embedding would fail at the posterior, flow, duration, or decoder input.
        """
        if pad_to_multiple_of is not None:
            new_num_speakers = ((new_num_speakers + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        # first, take care of embed_speaker
        if self.config.num_speakers <= 1:
            if speaker_embedding_size is None:
                raise ValueError(
                    "The current model had no previous speaker embedding, but `speaker_embedding_size` is not specified. Pass `speaker_embedding_size` to this method."
                )
            # create new embedding layer
            new_embeddings = nn.Embedding(
                new_num_speakers,
                speaker_embedding_size,
                device=self.device,
            )
            # initialize all new embeddings
            self._init_weights(new_embeddings)
        else:
            new_embeddings = self._get_resized_embeddings(self.embed_speaker, new_num_speakers)

        self.embed_speaker = new_embeddings

        # then take care of sub-models
        self.flow.resize_speaker_embeddings(speaker_embedding_size)
        for flow in self.flow.flows:
            self._init_weights(flow.wavenet.cond_layer)

        self.decoder.resize_speaker_embedding(speaker_embedding_size)
        self._init_weights(self.decoder.cond)

        self.duration_predictor.resize_speaker_embeddings(speaker_embedding_size)
        self._init_weights(self.duration_predictor.cond)

        self.posterior_encoder.resize_speaker_embeddings(speaker_embedding_size)
        self._init_weights(self.posterior_encoder.wavenet.cond_layer)

        self.config.num_speakers = new_num_speakers
        self.config.speaker_embedding_size = speaker_embedding_size

    def get_input_embeddings(self):
        """Return the text encoder's token embedding table.

        Transformers' resize and inspection utilities rely on this standard accessor.
        """
        return self.text_encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Replace the text encoder's token embedding table.

        This setter allows vocabulary resizing while keeping the model API compatible
        with pretrained-model utilities.
        """
        self.text_encoder.set_input_embeddings(value)

    def apply_weight_norm(self):
        """Apply weight normalization to the generator components used during training.

        The training loop optimizes these components with adversarial losses using the
        weight-normalized parameterization expected by VITS.
        """
        self.decoder.apply_weight_norm()
        self.flow.apply_weight_norm()
        self.posterior_encoder.apply_weight_norm()

    def remove_weight_norm(self):
        """Remove weight normalization from the generator components before export.

        This converts the trained parameters to their ordinary inference form before
        saving a deployable checkpoint.
        """
        self.decoder.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.posterior_encoder.remove_weight_norm()

    def discriminate(self, hidden_states):
        """Run the attached discriminator on a waveform and return scores and feature maps.

        Keeping this delegation on the pretraining model provides a single interface
        for discriminator evaluation during adversarial optimization.
        """
        return self.discriminator(hidden_states)

    def get_encoder(self):
        """Return the text encoder for model integrations that expect an encoder accessor.

        The accessor exposes token representations without requiring callers to know
        the model's internal module layout.
        """
        return self.text_encoder

    def _inference_forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        speaker_embeddings: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """Synthesize waveform audio from text IDs without reference spectrogram labels.

        This path predicts durations from text, expands the prior across generated
        frames, samples the latent flow, and decodes audio; it is used when no training
        spectrogram labels are supplied.
        """
        text_encoder_output = self.text_encoder(
            input_ids=input_ids,
            padding_mask=padding_mask,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = text_encoder_output[0] if not return_dict else text_encoder_output.last_hidden_state
        hidden_states = hidden_states.transpose(1, 2)
        input_padding_mask = padding_mask.transpose(1, 2)
        prior_means = text_encoder_output[1] if not return_dict else text_encoder_output.prior_means
        prior_log_variances = text_encoder_output[2] if not return_dict else text_encoder_output.prior_log_variances

        if self.config.use_stochastic_duration_prediction:
            log_duration = self.duration_predictor(
                hidden_states,
                input_padding_mask,
                speaker_embeddings,
                reverse=True,
                noise_scale=self.noise_scale_duration,
            )
        else:
            log_duration = self.duration_predictor(hidden_states, input_padding_mask, speaker_embeddings)

        length_scale = 1.0 / self.speaking_rate
        duration = torch.ceil(torch.exp(log_duration) * input_padding_mask * length_scale)
        predicted_lengths = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()

        # Create a padding mask for the output lengths of shape (batch, 1, max_output_length)
        indices = torch.arange(predicted_lengths.max(), dtype=predicted_lengths.dtype, device=predicted_lengths.device)
        output_padding_mask = indices.unsqueeze(0) < predicted_lengths.unsqueeze(1)
        output_padding_mask = output_padding_mask.unsqueeze(1).to(input_padding_mask.dtype)

        # Reconstruct an attention tensor of shape (batch, 1, out_length, in_length)
        attn_mask = torch.unsqueeze(input_padding_mask, 2) * torch.unsqueeze(output_padding_mask, -1)
        batch_size, _, output_length, input_length = attn_mask.shape
        cum_duration = torch.cumsum(duration, -1).view(batch_size * input_length, 1)
        indices = torch.arange(output_length, dtype=duration.dtype, device=duration.device)
        valid_indices = indices.unsqueeze(0) < cum_duration
        valid_indices = valid_indices.to(attn_mask.dtype).view(batch_size, input_length, output_length)
        padded_indices = valid_indices - nn.functional.pad(valid_indices, [0, 0, 1, 0, 0, 0])[:, :-1]
        attn = padded_indices.unsqueeze(1).transpose(2, 3) * attn_mask

        # Expand prior distribution
        prior_means = torch.matmul(attn.squeeze(1), prior_means).transpose(1, 2)
        prior_log_variances = torch.matmul(attn.squeeze(1), prior_log_variances).transpose(1, 2)

        prior_latents = prior_means + torch.randn_like(prior_means) * torch.exp(prior_log_variances) * self.noise_scale
        latents = self.flow(prior_latents, output_padding_mask, speaker_embeddings, reverse=True)

        spectrogram = latents * output_padding_mask
        waveform = self.decoder(spectrogram, speaker_embeddings)
        waveform = waveform.squeeze(1)
        sequence_lengths = predicted_lengths * np.prod(self.config.upsample_rates)

        if not return_dict:
            outputs = (waveform, sequence_lengths, spectrogram) + text_encoder_output[3:]
            return outputs

        return VitsModelOutput(
            waveform=waveform,
            sequence_lengths=sequence_lengths,
            spectrogram=spectrogram,
            hidden_states=text_encoder_output.hidden_states,
            attentions=text_encoder_output.attentions,
        )

    @add_start_docstrings_to_model_forward(VITS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=VitsModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        speaker_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.FloatTensor] = None,
        labels_attention_mask: Optional[torch.Tensor] = None,
        monotonic_alignment_function: Optional[Callable] = None,
    ) -> Union[Tuple[Any], VitsModelOutput]:
        r"""
        labels (`torch.FloatTensor` of shape `(batch_size, config.spectrogram_bins, sequence_length)`, *optional*):
            Float values of target spectrogram.
        labels_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on labels. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        monotonic_alignment_function (`Callable`, *optional*):
            Monotonic alignment function. Used for training, i.e when `labels` are provided. By default, it will use a
            Pytorch implementation of the monotonic alignment function which is awfully slow. An alternative relying on
            cython is proposed in examples/pytorch/text-to-speech/run_vits_finetuning.py
        Returns:

        Example:

        ```python
        >>> from transformers import VitsTokenizer, VitsModelForPreTraining, set_seed
        >>> import torch

        >>> tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
        >>> model = VitsModelForPreTraining.from_pretrained("facebook/mms-tts-eng")

        >>> inputs = tokenizer(text="Hello - my dog is cute", return_tensors="pt")

        >>> set_seed(555)  # make deterministic

        >>> with torch.no_grad():
        ...     outputs = model(inputs["input_ids"])
        >>> outputs.waveform.shape
        torch.Size([1, 45824])
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        monotonic_alignment_function = (
            monotonic_align_max_path if monotonic_alignment_function is None else monotonic_alignment_function
        )

        if attention_mask is not None:
            input_padding_mask = attention_mask.unsqueeze(-1).float()
        else:
            input_padding_mask = torch.ones_like(input_ids).unsqueeze(-1).float()

        if self.config.num_speakers > 1 and speaker_id is not None:
            if isinstance(speaker_id, int):
                speaker_id = torch.full(size=(1,), fill_value=speaker_id, device=self.device)
            elif isinstance(speaker_id, (list, tuple, np.ndarray)):
                speaker_id = torch.tensor(speaker_id, device=self.device)

            if not ((0 <= speaker_id).all() and (speaker_id < self.config.num_speakers).all()).item():
                raise ValueError(f"Set `speaker_id` in the range 0-{self.config.num_speakers - 1}.")
            if not (len(speaker_id) == 1 or len(speaker_id == len(input_ids))):
                raise ValueError(
                    f"You passed {len(speaker_id)} `speaker_id` but you should either pass one speaker id or `batch_size` `speaker_id`."
                )

            speaker_embeddings = self.embed_speaker(speaker_id).unsqueeze(-1)
        else:
            speaker_embeddings = None

        # if inference, return inference forward of VitsModel
        if labels is None:
            return self._inference_forward(
                input_ids,
                attention_mask,
                speaker_embeddings,
                output_attentions,
                output_hidden_states,
                return_dict,
                input_padding_mask,
            )

        if labels_attention_mask is not None:
            labels_padding_mask = labels_attention_mask.unsqueeze(1).float()
        else:
            labels_attention_mask = torch.ones((labels.shape[0], labels.shape[2])).float().to(self.device)
            labels_padding_mask = labels_attention_mask.unsqueeze(1)

        text_encoder_output = self.text_encoder(
            input_ids=input_ids,
            padding_mask=input_padding_mask,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = text_encoder_output[0] if not return_dict else text_encoder_output.last_hidden_state
        hidden_states = hidden_states.transpose(1, 2)
        input_padding_mask = input_padding_mask.transpose(1, 2)
        prior_means = text_encoder_output[1] if not return_dict else text_encoder_output.prior_means
        prior_log_variances = text_encoder_output[2] if not return_dict else text_encoder_output.prior_log_variances

        latents, posterior_means, posterior_log_variances = self.posterior_encoder(
            labels, labels_padding_mask, speaker_embeddings
        )
        prior_latents = self.flow(latents, labels_padding_mask, speaker_embeddings, reverse=False)

        prior_means, prior_log_variances = prior_means.transpose(1, 2), prior_log_variances.transpose(1, 2)
        with torch.no_grad():
            # negative cross-entropy

            # [batch_size, d, latent_length]
            prior_variances = torch.exp(-2 * prior_log_variances)
            # [batch_size, 1, latent_length]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - prior_log_variances, [1], keepdim=True)
            # [batch_size, text_length, d] x [batch_size, d, latent_length] = [batch_size, text_length, latent_length]
            neg_cent2 = torch.matmul(-0.5 * (prior_latents**2).transpose(1, 2), prior_variances)
            # [batch_size, text_length, d] x [batch_size, d, latent_length] = [batch_size, text_length, latent_length]
            neg_cent3 = torch.matmul(prior_latents.transpose(1, 2), (prior_means * prior_variances))
            # [batch_size, 1, latent_length]
            neg_cent4 = torch.sum(-0.5 * (prior_means**2) * prior_variances, [1], keepdim=True)

            # [batch_size, text_length, latent_length]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(input_padding_mask, 2) * torch.unsqueeze(labels_padding_mask, -1)

            attn = monotonic_alignment_function(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        durations = attn.sum(2)

        if self.config.use_stochastic_duration_prediction:
            log_duration = self.duration_predictor(
                hidden_states, input_padding_mask, speaker_embeddings, durations=durations, reverse=False
            )
            log_duration = log_duration / torch.sum(input_padding_mask)
        else:
            log_duration_padded = torch.log(durations + 1e-6) * input_padding_mask
            log_duration = self.duration_predictor(hidden_states, input_padding_mask, speaker_embeddings)
            log_duration = torch.sum((log_duration - log_duration_padded) ** 2, [1, 2]) / torch.sum(input_padding_mask)

        # expand priors
        prior_means = torch.matmul(attn.squeeze(1), prior_means.transpose(1, 2)).transpose(1, 2)
        prior_log_variances = torch.matmul(attn.squeeze(1), prior_log_variances.transpose(1, 2)).transpose(1, 2)

        label_lengths = labels_attention_mask.sum(dim=1)
        latents_slice, ids_slice = rand_slice_segments(latents, label_lengths, segment_size=self.segment_size)

        waveform = self.decoder(latents_slice, speaker_embeddings)

        if not return_dict:
            outputs = (
                waveform,
                log_duration,
                attn,
                ids_slice,
                input_padding_mask,
                labels_padding_mask,
                latents,
                prior_latents,
                prior_means,
                prior_log_variances,
                posterior_means,
                posterior_log_variances,
            )
            return outputs

        return VitsTrainingOutput(
            waveform=waveform,
            log_duration=log_duration,
            attn=attn,
            ids_slice=ids_slice,
            input_padding_mask=input_padding_mask,
            labels_padding_mask=labels_padding_mask,
            latents=latents,
            prior_latents=prior_latents,
            prior_means=prior_means,
            prior_log_variances=prior_log_variances,
            posterior_means=posterior_means,
            posterior_log_variances=posterior_log_variances,
        )
