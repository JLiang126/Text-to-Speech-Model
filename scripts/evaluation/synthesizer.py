"""Text -> waveform with a trained VITS model (the one synthesis path used everywhere).

Two checkpoint formats, one interface (``synthesize(text, seed)`` -> (waveform, timing),
``sample_rate``, ``speaker_id``, ``n_speakers``):

1. Official VITS format (``G_<step>.pth`` + JSON config): ``VITSSynthesizer``.
   Model code: the official implementation (Kim, Kong & Son 2021,
   https://github.com/jaywalnut310/vits, MIT licence), in the folder given by ``vits_dir``.
   Text front-end: the repo's ``scripts/TextProcessor.py`` (the data loader's front-end),
   or ``frontend="vits"`` for the official ``text`` module.
2. Hugging Face format (``transformers.VitsModel``): ``HFVITSSynthesizer``. This is the
   format finetune-hf-vits (Y. Lacombe, https://github.com/ylacombe/finetune-hf-vits,
   reference copy in ``finetune-hf-vits-main/``) loads and saves, e.g. the pretrained
   ``ylacombe/vits-vctk-with-discriminator``, a run's output folder, or one of its
   ``checkpoint-<step>`` folders. Text front-end: the checkpoint's own ``VitsTokenizer``,
   i.e. exactly what fine-tuning used.

Nothing here changes a model; this module only loads checkpoints and runs inference.
Before either front-end, ``normalise_input_text`` spells out numbers, money, % and &
(otherwise digits are deleted; see text_normalise.py).

Sampling: VITS is random at inference (noise in the prior and in the stochastic
duration predictor), so every call takes a ``seed``. Same text + same seed + same
checkpoint gives the same waveform.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import os
import re
import sys
import time
import types
import zlib
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from scripts.evaluation.text_normalise import normalise_input_text

PathLike = Union[str, Path]

# Official VITS inference defaults (jaywalnut310/vits inference.ipynb; also the VitsConfig defaults)
NOISE_SCALE = 0.667
NOISE_SCALE_W = 0.8
LENGTH_SCALE = 1.0

_STEP_PATTERNS = (re.compile(r"G_(\d+)\.pth$"), re.compile(r"checkpoint-(\d+)$"))


def seed_for(utt_id: str, base_seed: int = 1234) -> int:
    """Stable per-utterance seed: independent of order, identical across experiments."""
    return zlib.crc32(f"{base_seed}:{utt_id}".encode("utf-8")) % (2 ** 31)


def checkpoint_step(path: Optional[PathLike]) -> Optional[int]:
    """Training step from a checkpoint name: G_10000.pth or checkpoint-10000 -> 10000."""
    if not path:
        return None
    name = Path(str(path).rstrip("/\\")).name
    for pattern in _STEP_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def import_vits(vits_dir: PathLike):
    """Imports the official VITS ``models`` and ``utils`` modules from ``vits_dir``."""
    vits_dir = Path(vits_dir).resolve()
    if not (vits_dir / "models.py").is_file():
        raise FileNotFoundError(f"No VITS code at {vits_dir} (expected models.py). "
                                f"Pass --vits_dir or set VITS_DIR to the vits-main folder.")
    if str(vits_dir) not in sys.path:
        sys.path.insert(0, str(vits_dir))
    try:
        importlib.import_module("monotonic_align")  # compiled Cython, only used during training
    except ImportError:
        # Inference never runs monotonic alignment search, so a placeholder lets
        # evaluation run without compiling the Cython extension.
        sys.modules["monotonic_align"] = types.ModuleType("monotonic_align")
    return importlib.import_module("models"), importlib.import_module("utils")


def _use_bundled_espeak() -> None:
    """Points phonemizer at the espeak-ng library shipped by espeakng-loader (as TextProcessor does)."""
    try:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    except Exception:  # fall back to a system-wide espeak-ng install
        pass


def _strip_module_prefix(state: Dict[str, object]) -> Dict[str, object]:
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}


def _load_state_dict(checkpoint_path: PathLike) -> Dict[str, object]:
    import torch

    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except Exception:
        # Official VITS checkpoints also pickle the optimiser state; only load files you trust.
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    return _strip_module_prefix(state)


class _Synthesizer:
    """What both formats share: speaker-ID validation and input-text preparation."""

    n_speakers = 0
    expand_numbers = True
    _speaker_id: Optional[int] = None

    @property
    def speaker_id(self) -> Optional[int]:
        return self._speaker_id

    @speaker_id.setter
    def speaker_id(self, value: Optional[int]) -> None:
        if self.n_speakers > 0:
            if value is None:
                raise ValueError(f"This checkpoint is multi-speaker ({self.n_speakers} speakers): pass "
                                 f"--speaker_id (the ID the LJSpeech clips were trained with).")
            if not 0 <= int(value) < self.n_speakers:
                raise ValueError(f"--speaker_id must be between 0 and {self.n_speakers - 1}, got {value}.")
            self._speaker_id = int(value)
        else:
            if value is not None:
                raise ValueError("This checkpoint is single-speaker: drop --speaker_id.")
            self._speaker_id = None

    def prepare_text(self, text: str) -> str:
        return normalise_input_text(text) if self.expand_numbers else text


# ------------------------------------------------------------ official VITS format
class VITSSynthesizer(_Synthesizer):
    """Loads an official-format VITS generator checkpoint and turns text into a waveform."""

    def __init__(self, config_path: PathLike, checkpoint_path: PathLike, vits_dir: PathLike = "vits-main",
                 device: str = "cpu", speaker_id: Optional[int] = None, frontend: str = "repo",
                 noise_scale: float = NOISE_SCALE, noise_scale_w: float = NOISE_SCALE_W,
                 length_scale: float = LENGTH_SCALE, expand_numbers: bool = True, strict: bool = True):
        import torch

        models, utils = import_vits(vits_dir)
        self.hps = utils.get_hparams_from_file(str(config_path))
        self.sample_rate = int(self.hps.data.sampling_rate)
        self.device = device
        self.noise_scale, self.noise_scale_w, self.length_scale = noise_scale, noise_scale_w, length_scale
        self.frontend = frontend
        self.expand_numbers = expand_numbers
        self._torch = torch
        self._init_frontend()

        state = _load_state_dict(checkpoint_path)
        n_vocab = int(state["enc_p.emb.weight"].shape[0])
        if n_vocab != self.vocab_size:
            raise ValueError(
                f"Checkpoint expects {n_vocab} text symbols but the '{frontend}' front-end has "
                f"{self.vocab_size}. Token IDs would point at the wrong embeddings; fix the symbol "
                f"table so it matches the checkpoint (see tests/test_text_frontend.py).")
        self.n_speakers = int(state["emb_g.weight"].shape[0]) if "emb_g.weight" in state else 0
        self.speaker_id = speaker_id  # validated by the setter

        self.model = models.SynthesizerTrn(
            n_vocab, self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.n_speakers, **self.hps.model).to(device).eval()
        result = self.model.load_state_dict(state, strict=False)
        if strict and (result.missing_keys or result.unexpected_keys):
            raise RuntimeError(f"Checkpoint/config mismatch. Missing: {list(result.missing_keys)[:10]} "
                               f"Unexpected: {list(result.unexpected_keys)[:10]} "
                               f"(is --config the one this checkpoint was trained with?)")
        self.checkpoint_path = str(checkpoint_path)

    def _init_frontend(self) -> None:
        # phonemizer warns "words count mismatch" whenever punctuation is preserved; harmless here.
        logging.getLogger("phonemizer").setLevel(logging.ERROR)
        add_blank = bool(getattr(self.hps.data, "add_blank", True))
        if self.frontend == "repo":
            from scripts.TextProcessor import TextProcessor
            self.text_processor = TextProcessor(add_blank=add_blank)
            self.vocab_size = self.text_processor.vocab_size
        elif self.frontend == "vits":
            _use_bundled_espeak()
            commons = importlib.import_module("commons")
            text_module = importlib.import_module("text")
            symbols = importlib.import_module("text.symbols").symbols
            cleaners = list(self.hps.data.text_cleaners)

            def encode(t: str):
                ids = text_module.text_to_sequence(t, cleaners)
                return commons.intersperse(ids, 0) if add_blank else ids

            self._vits_encode = encode
            self.vocab_size = len(symbols)
        else:
            raise ValueError("frontend must be 'repo' or 'vits'")

    def text_to_ids(self, text: str):
        text = self.prepare_text(text)
        if self.frontend == "repo":
            return self.text_processor.text_to_sequence(text)
        return self._torch.LongTensor(self._vits_encode(text))

    def speaker_table(self) -> Optional[np.ndarray]:
        """The speaker embedding matrix (n_speakers x dim), or None for single-speaker models."""
        if not self.n_speakers:
            return None
        return self.model.emb_g.weight.detach().float().cpu().numpy()

    def synthesize(self, text: str, seed: Optional[int] = 1234) -> Tuple[np.ndarray, Dict[str, float]]:
        """Returns (waveform float32 at ``self.sample_rate``, timing info)."""
        torch = self._torch
        t0 = time.perf_counter()
        ids = self.text_to_ids(text)
        t1 = time.perf_counter()
        if seed is not None:
            torch.manual_seed(int(seed))  # seeds the CPU and every CUDA device
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


# ------------------------------------------------------------ Hugging Face format
def resolve_hf_model(model: PathLike) -> Tuple[Union[str, Path], Optional[Path]]:
    """Where to read config/tokenizer from, and (for checkpoint-<step> folders) the weights file.

    - a folder with config.json (a finished finetune-hf-vits run, or any saved VitsModel)
      -> (folder, None)
    - a ``checkpoint-<step>`` folder written by ``accelerator.save_state`` during training:
      it holds only weights, so config/tokenizer come from the run folder above it
      -> (run folder, weights file)
    - anything else is treated as a Hugging Face hub id -> (id, None)
    """
    path = Path(model)
    if path.is_dir():
        if (path / "config.json").is_file():
            return path, None
        if checkpoint_step(path) is not None and (path.parent / "config.json").is_file():
            return path.parent, _find_generator_weights(path)
        raise FileNotFoundError(f"{path} has no config.json, and it is not a checkpoint-<step> folder "
                                f"inside a run folder that has one.")
    if path.is_file():
        raise ValueError(f"--model must be a folder or a hub id, not a file: {path}")
    return str(model), None


def _weight_keys(path: Path):
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            return list(f.keys())
    return list(_load_state_dict(path).keys())


def _find_generator_weights(folder: Path) -> Path:
    """accelerate saves one file per prepared model (generator, discriminator): pick the generator."""
    candidates = sorted(folder.glob("model*.safetensors")) + sorted(folder.glob("pytorch_model*.bin"))
    for candidate in candidates:
        keys = [k[len("module."):] if k.startswith("module.") else k for k in _weight_keys(candidate)]
        if any(k.startswith("text_encoder.") for k in keys):
            return candidate
    raise FileNotFoundError(f"No generator weights (text_encoder.*) in {folder}; "
                            f"found {[c.name for c in candidates]}")


def _read_weights(path: Path) -> Dict[str, object]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return _strip_module_prefix(load_file(str(path)))
    return _load_state_dict(path)


_WEIGHT_NORM_LAYOUTS = ((".weight_g", ".weight_v"),  # torch.nn.utils.weight_norm
                        (".parametrizations.weight.original0",  # torch.nn.utils.parametrizations.weight_norm
                         ".parametrizations.weight.original1"))


def _adapt_weight_norm(state: Dict[str, object], target_keys) -> Dict[str, object]:
    """Converts weight-norm parameters in ``state`` to the layout the target model expects.

    finetune-hf-vits trains with extra weight norm switched on (decoder, flow conv_pre/post)
    and removes it only before the final save_pretrained, so checkpoint-<step> folders still
    hold (g, v) pairs, in either PyTorch layout. For each pair:
    - the target keeps weight norm on that layer -> copy (renaming between layouts if needed);
    - the target has a plain weight -> fold: weight = g * v / ||v||, the norm taken over every
      dimension where g has size 1 (default dim=0 weight norm -> all dims except the first).
    """
    target_keys = set(target_keys)
    adapted: Dict[str, object] = {}
    for key, value in state.items():
        for g_suffix, v_suffix in _WEIGHT_NORM_LAYOUTS:
            if key.endswith(v_suffix):
                break  # handled together with its g
            if key.endswith(g_suffix):
                base = key[: -len(g_suffix)]
                g, v = value, state[base + v_suffix]
                for target_g, target_v in _WEIGHT_NORM_LAYOUTS:
                    if base + target_g in target_keys:
                        adapted[base + target_g], adapted[base + target_v] = g, v
                        break
                else:
                    import torch
                    gf, vf = g.float(), v.float()
                    dims = [d for d in range(vf.dim()) if gf.dim() == 0 or gf.shape[d] == 1]
                    norm = torch.linalg.vector_norm(vf, dim=dims, keepdim=True)
                    adapted[base + ".weight"] = (gf * vf / norm).to(v.dtype)
                break
        else:
            adapted[key] = value
    return adapted


class HFVITSSynthesizer(_Synthesizer):
    """Loads a Hugging Face-format VITS model (transformers ``VitsModel``) for inference."""

    def __init__(self, model: PathLike, device: str = "cpu", speaker_id: Optional[int] = None,
                 noise_scale: float = NOISE_SCALE, noise_scale_w: float = NOISE_SCALE_W,
                 length_scale: float = LENGTH_SCALE, expand_numbers: bool = True, strict: bool = True):
        import torch
        import transformers
        from transformers import VitsConfig, VitsModel, VitsTokenizer

        logging.getLogger("phonemizer").setLevel(logging.ERROR)
        _use_bundled_espeak()  # the tokenizer phonemises with espeak-ng
        source, weights = resolve_hf_model(model)
        previous = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()  # key mismatches are checked below instead
        try:
            self.tokenizer = VitsTokenizer.from_pretrained(str(source))
            if weights is None:
                net, info = VitsModel.from_pretrained(str(source), output_loading_info=True)
                missing = list(info.get("missing_keys", []))
                unexpected = list(info.get("unexpected_keys", []))
            else:
                net = VitsModel(VitsConfig.from_pretrained(str(source)))
                state = _adapt_weight_norm(_read_weights(weights), net.state_dict().keys())
                result = net.load_state_dict(state, strict=False)
                missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
        finally:
            transformers.logging.set_verbosity(previous)
        unexpected = [k for k in unexpected if not k.startswith("discriminator.")]  # training-only part
        if strict and (missing or unexpected):
            raise RuntimeError(f"{model}: weights do not match VitsModel. Missing: {missing[:10]} "
                               f"Unexpected: {unexpected[:10]}")

        self.model = net.to(device).eval()
        # VitsModel reads these attributes at inference (length = durations / speaking_rate)
        self.model.noise_scale = noise_scale
        self.model.noise_scale_duration = noise_scale_w
        self.model.speaking_rate = 1.0 / length_scale
        self.noise_scale, self.noise_scale_w, self.length_scale = noise_scale, noise_scale_w, length_scale
        self.device = device
        self.expand_numbers = expand_numbers
        self.frontend = "hf"
        self.sample_rate = int(self.model.config.sampling_rate)
        num_speakers = int(getattr(self.model.config, "num_speakers", 1) or 1)
        self.n_speakers = num_speakers if num_speakers > 1 else 0
        self.speaker_id = speaker_id  # validated by the setter
        self.checkpoint_path = str(model)
        self._torch = torch

    def text_to_ids(self, text: str):
        return self.tokenizer(self.prepare_text(text), return_tensors="pt")

    def speaker_table(self) -> Optional[np.ndarray]:
        """The speaker embedding matrix (n_speakers x dim), or None for single-speaker models."""
        if not self.n_speakers:
            return None
        return self.model.embed_speaker.weight.detach().float().cpu().numpy()

    def synthesize(self, text: str, seed: Optional[int] = 1234) -> Tuple[np.ndarray, Dict[str, float]]:
        """Returns (waveform float32 at ``self.sample_rate``, timing info)."""
        torch = self._torch
        t0 = time.perf_counter()
        inputs = self.text_to_ids(text)
        t1 = time.perf_counter()
        if seed is not None:
            torch.manual_seed(int(seed))
        with torch.inference_mode():
            output = self.model(input_ids=inputs["input_ids"].to(self.device),
                                attention_mask=inputs["attention_mask"].to(self.device),
                                speaker_id=self.speaker_id)
            audio = output.waveform[0].float().cpu().numpy()
        t2 = time.perf_counter()
        return audio, {"frontend_s": t1 - t0, "model_s": t2 - t1, "n_tokens": int(inputs["input_ids"].shape[1])}


# ------------------------------------------------------------ one way in for every script
def load_synthesizer(model: Optional[PathLike] = None, config: Optional[PathLike] = None,
                     checkpoint: Optional[PathLike] = None, vits_dir: PathLike = "vits-main",
                     frontend: str = "repo", **kwargs) -> _Synthesizer:
    """``model`` -> Hugging Face format; ``config`` + ``checkpoint`` -> official VITS format."""
    if model and (config or checkpoint):
        raise ValueError("Give either --model (Hugging Face format) or --config/--checkpoint "
                         "(official VITS format), not both.")
    if model:
        return HFVITSSynthesizer(model, **kwargs)
    if config and checkpoint:
        return VITSSynthesizer(config, checkpoint, vits_dir=vits_dir, frontend=frontend, **kwargs)
    raise ValueError("No model given: pass --model (Hugging Face format) or --config and "
                     "--checkpoint (official VITS format).")


def add_model_arguments(parser: argparse.ArgumentParser, with_speaker: bool = True) -> None:
    """The model-selection options shared by synth.py, evaluate.py and rank_speakers.py."""
    g = parser.add_argument_group("model (give --model, or --config and --checkpoint)")
    g.add_argument("--model", help="Hugging Face-format VITS: a hub id (e.g. ylacombe/vits-vctk-with-discriminator), "
                                   "a finetune-hf-vits output folder, or one of its checkpoint-<step> folders")
    g.add_argument("--config", help="official VITS format: JSON config the checkpoint was trained with")
    g.add_argument("--checkpoint", help="official VITS format: generator checkpoint G_<step>.pth")
    g.add_argument("--vits_dir", default=os.environ.get("VITS_DIR", "vits-main"),
                   help="official VITS code folder (default: $VITS_DIR or ./vits-main)")
    g.add_argument("--frontend", choices=["repo", "vits"], default="repo",
                   help="official format only: repo = scripts/TextProcessor.py (same as training); "
                        "vits = official text module")
    if with_speaker:
        g.add_argument("--speaker_id", type=int, default=None,
                       help="speaker index for multi-speaker checkpoints: the one the LJSpeech clips were "
                            "fine-tuned with")


def model_given(args: argparse.Namespace) -> bool:
    return bool(args.model or (args.config and args.checkpoint))


def synthesizer_from_args(args: argparse.Namespace, device: str, **kwargs) -> _Synthesizer:
    """Builds the synthesizer chosen with the add_model_arguments options."""
    kwargs.setdefault("speaker_id", getattr(args, "speaker_id", None))
    return load_synthesizer(model=args.model, config=args.config, checkpoint=args.checkpoint,
                            vits_dir=args.vits_dir, frontend=args.frontend, device=device, **kwargs)
