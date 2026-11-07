import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "best_speaker_embedding.yaml"


def load_config(config_path: Path) -> Dict:
    """Load training settings from a YAML object and resolve derived defaults."""
    contents = config_path.read_text(encoding="utf-8")
    settings = yaml.safe_load(contents)
    if not isinstance(settings, dict):
        raise ValueError(f"Training config must contain a YAML object: {config_path}")

    # A null split directory lets DataConfig derive it from output_dir.
    if settings.get("output_split_dir") is None:
        settings.pop("output_split_dir", None)
    # Preserve the environment-variable default when using the bundled config.
    if config_path.resolve() == DEFAULT_CONFIG_PATH and os.environ.get("LJSPEECH_ROOT"):
        settings["data_root"] = os.environ["LJSPEECH_ROOT"]
    return settings


# Dataclass Configuration and Seed
@dataclass
class DataConfig:
    """
    Dataset Configuration for experiment set up.

    Note that the majority of the configuration is derived from the YAML file specified at runtime.

    """
    experiment_description: str = "Default experiment configuration."

    # Dataset Processing
    data_root: Path = Path(os.environ.get("LJSPEECH_ROOT", "./dataset/"))
    output_split_dir: Optional[Path] = None
    seed: int = 42
    split_ratios: Tuple[float, float, float] = (0.80, 0.10, 0.10)
    subset_size: int = None # Set to None for full dataset
    num_workers: int = 0
    persistent_workers: bool = False

    # Test Preprocessing
    lowercase: bool = False # Lowercases the text during preprocessing. Not really neccessary
    add_blank: bool = True # Added for visibility and for potential encoding testing
    suppress_word_mismatch_warnings: bool = False

    # Audio Preprocessing
    target_sr: int = 22050 # Native LJSpeech sample rate (Hz)
    min_duration_sec: float = 1.0
    max_duration_sec: float = 10.0
    trim_db: float = 40.0
    pad_tail: int = 0.05

    # Short Form Fourier Transform Configuration
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0  

    # checkpoint for the model
    model_name_or_path: str = "ylacombe/vits-ljs-with-discriminator"
    speaker_id: int = 0

    # training configuration
    output_dir: Path = Path("outputs/vits")
    num_train_epochs: int = 200
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    per_device_train_batch_size: int = 16
    learning_rate: float = 2e-5
    adam_beta1: float = 0.8
    adam_beta2: float = 0.99
    warmup_ratio: float = 0.01
    mixed_precision: str = "no"
    encoder_layerdrop: float = 0.0 # Dropout rate for encoder layers. It's 0.1 by default in the original VITS configuration, but 0 works much better here
    freeze_mode: str = "none" # Options: none, freeze_text_encoder, unfreeze_speech_embeddings_only
    save_final_discriminator: bool = False # whether to include the discriminator in the final checkpoint
    speaker_embedding_lr_ratio: float = 1.0

    # evaluation configuration
    do_eval: bool = True
    eval_steps: int = 50
    per_device_eval_batch_size: int = 16
    max_eval_samples: Optional[int] = None
    use_linear_lr_schedule: bool = True

    # loss weights for different components of the model
    weight_disc: float = 3
    weight_fmaps: float = 1
    weight_gen: float = 1
    weight_kl: float = 1.5
    weight_duration: float = 1
    weight_mel: float = 35

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.output_dir = Path(self.output_dir)
        self.output_split_dir = (
            Path(self.output_dir) / "splits"
            if self.output_split_dir is None
            else Path(self.output_split_dir)
        )

        # Ensure freeze_mode is valid
        if self.freeze_mode not in ["none", "freeze_text_encoder", "unfreeze_speech_embeddings_only"]:
            raise ValueError(f"Invalid freeze_mode: {self.freeze_mode}. Must be one of 'none', 'freeze_text_encoder', 'unfreeze_speech_embeddings_only'.")

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["data_root"] = str(self.data_root)
        d["output_split_dir"] = str(self.output_split_dir)
        d["output_dir"] = str(self.output_dir)
        return d

    def validate(self, output_mode: str = "restart") -> None:
        """Resolve the dataset path and validate settings before starting a run."""
        self.data_root = Path(self.data_root)
        if not (self.data_root / "metadata.csv").exists() and (self.data_root / "LJSpeech-1.1").is_dir():
            self.data_root /= "LJSpeech-1.1"
        self.output_dir = Path(self.output_dir)
        if self.output_split_dir is None:
            raise ValueError("output_split_dir must be resolved before validation.")
        self.output_split_dir = Path(self.output_split_dir)
        split_dir_resolved = self.output_split_dir.resolve()
        data_root_resolved = self.data_root.resolve()
        if (
            split_dir_resolved == data_root_resolved
            or split_dir_resolved in data_root_resolved.parents
            or data_root_resolved in split_dir_resolved.parents
        ):
            raise ValueError("output_split_dir must not overlap data_root; the dataset directory is read-only.")

        if self.num_train_epochs < 1 or self.per_device_train_batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("Epochs, batch size and gradient accumulation steps must be positive.")
        if self.learning_rate <= 0 or self.num_workers < 0:
            raise ValueError("Learning rate must be positive and num_workers must be nonnegative.")
        if self.mixed_precision not in ("no", "fp16", "bf16"):
            raise ValueError("mixed_precision must be one of: no, fp16, bf16.")
        if self.subset_size is not None and self.subset_size < 1:
            raise ValueError("subset_size must be positive or null.")
        if self.do_eval and (self.eval_steps < 1 or self.per_device_eval_batch_size < 1):
            raise ValueError("Evaluation interval and batch size must be positive.")
        if self.do_eval and self.max_eval_samples is not None and self.max_eval_samples < 1:
            raise ValueError("max_eval_samples must be positive or null.")
        if any(ratio < 0 for ratio in self.split_ratios) or not math.isclose(sum(self.split_ratios), 1.0):
            raise ValueError("split_ratios must be nonnegative and sum to one.")
        if len(self.split_ratios) != 3:
            raise ValueError("split_ratios must contain train, validation and test ratios.")
        if output_mode == "new" and self.output_dir.exists():
            if not self.output_dir.is_dir() or any(self.output_dir.iterdir()):
                raise ValueError(f"Output directory ({self.output_dir}) is not empty.")