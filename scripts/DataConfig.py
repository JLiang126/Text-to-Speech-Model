import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    split_ratios: Tuple[float, float, float] = (0.90, 0.05, 0.05)
    subset_size: int = None # Set to None for full dataset
    num_workers: int = 0
    persistent_workers: bool = False

    # Test Preprocessing
    lowercase: bool = True
    add_blank: bool = True
    suppress_word_mismatch_warnings: bool = False
    preprocess_token_ids: bool = False

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
    max_eval_samples: int = 25
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