import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

# Dataclass Configuration and Seed
@dataclass
class DataConfig:
    """Dataset Configuration for experiment set up."""
    # Dataset Processing
    data_root: Path = Path(os.environ.get("LJSPEECH_ROOT", "./dataset/"))
    output_split_dir: Path = Path(data_root) / "splits"
    seed: int = 42
    split_ratios: Tuple[float, float, float] = (0.90, 0.05, 0.05)
    subset_size: int = 3000 # Set to None for full dataset

    # Test Preprocessing
    lowercase: bool = True
    add_blank: bool = True

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

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["data_root"] = str(self.data_root)
        d["output_split_dir"] = str(self.output_split_dir)
        return d