from typing import Dict, Tuple

import torch
import torchaudio

from scripts.DataLoader import DataConfig

class AudioProcessor:
    """
    Computes both:
    - 513-bin Linear Spectrograms (required by VITS PosteriorEncoder enc_q)
    - 80-band Log-Mel Spectrograms (required by VITS Mel reconstruction loss)

    Whilst also processing and cleaning audio data.
    """
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.resamplers: Dict[int, torchaudio.transforms.Resample] = {}
        self.hann_window = torch.hann_window(cfg.win_length)
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=cfg.n_mels,
            sample_rate=cfg.target_sr,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            n_stft=cfg.n_fft // 2 + 1,
            norm="slaney",
            mel_scale="slaney",
        )

    def preprocess_waveform(self, waveform: torch.Tensor, original_sr: int) -> torch.Tensor:
        """Mono audio, resampling, DC offset removal, peak norm [-0.95, 0.95], and -40dB silence trim."""
        # Ensures mono audio (Safety check LJSpeech is already mono audio)
        target_sr = self.cfg.target_sr
        if waveform.dim() > 1 and waveform.size(0) > 1: waveform = torch.mean(waveform, dim=0, keepdim=True) 

        # Resample to LJSpeech Frequency again just a safety check if audio gets changed from format shifting
        if original_sr != target_sr:
            if original_sr not in self.resamplers: self.resamplers[original_sr] = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr) 
            waveform = self.resamplers[original_sr](waveform)

        # Remove DC offset and normalise amplitude to [-0.95, 0.95]
        waveform = waveform - waveform.mean(dim=-1, keepdim=True)
        peak = torch.max(torch.abs(waveform))
        if peak > 1e-8: waveform = (waveform / peak) * 0.95 # Incase of silence divide by zero errors.

        # RMS energy silence trimming 
        win_len = int(target_sr * 0.020) # 20ms window
        hop_len = int(target_sr * 0.005) # 5ms hop

        frames = waveform.squeeze(0).unfold(0, win_len, hop_len) # Converts into raw audio amplitudes
        rms = torch.sqrt(torch.mean(frames ** 2, dim=-1) + 1e-10)
        threshold = torch.max(rms) * (10.0 ** (-self.cfg.trim_db / 20.0))
        active_idx = torch.nonzero(rms > threshold, as_tuple=False).squeeze(-1)

        if active_idx.numel() > 0:
            start_sample = max(0, int(active_idx[0].item()) * hop_len)
            end_sample = min(waveform.size(-1), int(active_idx[-1].item()) * hop_len + win_len)
            waveform = waveform[:, start_sample:end_sample]

        # Append 50ms trailing silence pad
        if self.cfg.pad_tail > 0:
            pad_samples = int(target_sr * (self.cfg.pad_tail))
            waveform = torch.nn.functional.pad(waveform, (0, pad_samples), mode="constant", value=0.0)

        return waveform

    def compute_spectrograms(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the:
        - linear_spec : shape (513, Frames) — used by VITS PosteriorEncoder (enc_q)
        - log_mel : shape (80, Frames) — used for Mel-loss for the discriminator
        """
        pad_amount = int((self.cfg.n_fft - self.cfg.hop_length) / 2)
        padded_wav = torch.nn.functional.pad(waveform.unsqueeze(0), (pad_amount, pad_amount), mode="reflect").squeeze(0)

        stft_complex = torch.stft(
            padded_wav,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=self.hann_window.to(waveform.device),
            center=False,
            return_complex=True,
        )

        linear_spec = torch.sqrt(torch.clamp(stft_complex.real ** 2 + stft_complex.imag ** 2, min=1e-9)).squeeze(0)
        mel_spec = self.mel_scale.to(waveform.device)(linear_spec)
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))
        return linear_spec, log_mel