"""Text -> waveform with a trained VITS checkpoint (the single synthesis path used everywhere).

Model code: the official VITS implementation (Kim, Kong & Son 2021,
https://github.com/jaywalnut310/vits, MIT licence), vendored in the repo as the
folder given by ``vits_dir``. Nothing here changes the model; this module only
loads a checkpoint and calls ``SynthesizerTrn.infer``.

Text front-end: by default the repo's own ``scripts/TextProcessor.py`` (the same
one the data loader uses for training), so token IDs at inference match training.
``frontend="vits"`` uses the official VITS ``text`` module instead.

Sampling: VITS is stochastic at inference (noise in the prior and in the
stochastic duration predictor), so every call takes a ``seed``. Same text + same
seed + same checkpoint -> the same waveform.
"""
from __future__ import annotations

import sys
import time
import types
import zlib
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

# Official VITS inference defaults (jaywalnut310/vits inference.ipynb)
NOISE_SCALE = 0.667
NOISE_SCALE_W = 0.8
LENGTH_SCALE = 1.0


def seed_for(utt_id: str, base_seed: int = 1234) -> int:
    """Stable per-utterance seed: independent of order, identical across experiments."""
    return zlib.crc32(f"{base_seed}:{utt_id}".encode("utf-8")) % (2 ** 31)


def import_vits(vits_dir: PathLike):
    """Imports the official VITS ``models``/``utils`` modules from ``vits_dir``."""
    vits_dir = Path(vits_dir).resolve()
    if not (vits_dir / "models.py").is_file():
        raise FileNotFoundError(f"No VITS code at {vits_dir} (expected models.py). "
                                f"Pass --vits_dir pointing at the vits-main folder.")
    if str(vits_dir) not in sys.path:
        sys.path.insert(0, str(vits_dir))
    try:
        import monotonic_align  # noqa: F401  compiled Cython, only used during training
    except ImportError:
        # Inference never runs monotonic alignment search, so a placeholder lets
        # evaluation run without compiling the Cython extension.
        sys.modules["monotonic_align"] = types.ModuleType("monotonic_align")
    import models  # noqa: E402
    import utils  # noqa: E402
    return models, utils


def _load_state_dict(checkpoint_path: PathLike) -> Dict[str, "object"]:
    import torch

    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except Exception:
        # Official VITS checkpoints also pickle optimiser state; only load files you trust.
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}


class VITSSynthesizer:
    """Loads a VITS generator checkpoint and turns text into a waveform."""

    def __init__(self, config_path: PathLike, checkpoint_path: PathLike, vits_dir: PathLike = "vits-main",
                 device: str = "cpu", speaker_id: Optional[int] = None, frontend: str = "repo",
                 noise_scale: float = NOISE_SCALE, noise_scale_w: float = NOISE_SCALE_W,
                 length_scale: float = LENGTH_SCALE, strict: bool = True):
        import torch

        models, utils = import_vits(vits_dir)
        self.hps = utils.get_hparams_from_file(str(config_path))
        self.sample_rate = int(self.hps.data.sampling_rate)
        self.device = device
        self.noise_scale, self.noise_scale_w, self.length_scale = noise_scale, noise_scale_w, length_scale
        self.frontend = frontend
        self._init_frontend()

        state = _load_state_dict(checkpoint_path)
        n_vocab = int(state["enc_p.emb.weight"].shape[0])
        if n_vocab != self.vocab_size:
            raise ValueError(
                f"Checkpoint expects {n_vocab} text symbols but the '{frontend}' front-end has "
                f"{self.vocab_size}. Token IDs would point at the wrong embeddings; fix the symbol "
                f"table so it matches the checkpoint (see tests/test_text_frontend.py).")
        n_speakers = int(state["emb_g.weight"].shape[0]) if "emb_g.weight" in state else 0
        if n_speakers > 0 and speaker_id is None:
            raise ValueError(f"This checkpoint is multi-speaker ({n_speakers} speakers): pass --speaker_id "
                             f"(the ID the LJSpeech clips were trained with).")
        if n_speakers == 0 and speaker_id is not None:
            raise ValueError("This checkpoint is single-speaker: drop --speaker_id.")
        self.speaker_id = speaker_id
        self.n_speakers = n_speakers

        self.model = models.SynthesizerTrn(
            n_vocab, self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=n_speakers, **self.hps.model).to(device).eval()
        result = self.model.load_state_dict(state, strict=False)
        # Generator-only checkpoints are expected; anything else missing is a real problem.
        missing = [k for k in result.missing_keys]
        unexpected = [k for k in result.unexpected_keys]
        if strict and (missing or unexpected):
            raise RuntimeError(f"Checkpoint/model mismatch. Missing: {missing[:10]} "
                               f"Unexpected: {unexpected[:10]}")
        self.checkpoint_path = str(checkpoint_path)
        self._torch = torch

    def _init_frontend(self) -> None:
        import logging
        # phonemizer warns "words count mismatch" whenever punctuation is preserved; harmless noise here.
        logging.getLogger("phonemizer").setLevel(logging.ERROR)
        add_blank = bool(getattr(self.hps.data, "add_blank", True))
        if self.frontend == "repo":
            from scripts.TextProcessor import TextProcessor
            self.text_processor = TextProcessor(add_blank=add_blank)
            self.vocab_size = self.text_processor.vocab_size
        elif self.frontend == "vits":
            import commons
            from text import text_to_sequence
            from text.symbols import symbols

            cleaners = list(self.hps.data.text_cleaners)
            self._vits_encode = lambda t: (commons.intersperse(text_to_sequence(t, cleaners), 0)
                                           if add_blank else text_to_sequence(t, cleaners))
            self.vocab_size = len(symbols)
        else:
            raise ValueError("frontend must be 'repo' or 'vits'")

    def text_to_ids(self, text: str):
        if self.frontend == "repo":
            return self.text_processor.text_to_sequence(text)
        return self._torch.LongTensor(self._vits_encode(text))

    def synthesize(self, text: str, seed: Optional[int] = 1234) -> Tuple[np.ndarray, Dict[str, float]]:
        """Returns (waveform float32 at ``self.sample_rate``, timing info)."""
        torch = self._torch
        t0 = time.perf_counter()
        ids = self.text_to_ids(text)
        t1 = time.perf_counter()
        if seed is not None:
            torch.manual_seed(int(seed))  # seeds CPU and all CUDA devices
        with torch.inference_mode():
            x = ids.unsqueeze(0).to(self.device)
            x_lengths = torch.LongTensor([ids.size(0)]).to(self.device)
            sid = torch.LongTensor([self.speaker_id]).to(self.device) if self.speaker_id is not None else None
            audio = self.model.infer(x, x_lengths, sid=sid, noise_scale=self.noise_scale,
                                     noise_scale_w=self.noise_scale_w,
                                     length_scale=self.length_scale)[0][0, 0]
            audio = audio.float().cpu().numpy()  # .cpu() waits for the GPU to finish
        t2 = time.perf_counter()
        return audio, {"frontend_s": t1 - t0, "model_s": t2 - t1, "n_tokens": int(ids.size(0))}
