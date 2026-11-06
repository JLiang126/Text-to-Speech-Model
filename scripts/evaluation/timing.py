"""Latency and real-time factor (RTF) measurement protocol.

Protocol (reported in summary.json so the report can state it):
1. Warm-up: ``n_warmup`` untimed calls (loads espeak, CUDA kernels, caches).
2. Each timing sentence is synthesised ``n_repeats`` times at batch size 1 with
   a fixed seed, so every repeat produces the same audio length.
3. On GPU, ``torch.cuda.synchronize`` runs before starting and after stopping the
   clock, otherwise asynchronous kernels make the time look shorter than it is.
4. latency = wall-clock seconds from text in to waveform out (front-end included).
   RTF = latency / duration of the generated audio (RTF < 1 is faster than real time).
5. The device, CPU model, thread count and library versions are recorded, because
   timings are only comparable on the same hardware: time every experiment on
   the same machine and the same GPU. Results are saved per device type
   (latency_cuda.json, latency_cpu.json), so GPU and CPU numbers can sit side by side.
"""
from __future__ import annotations

import platform
import subprocess
import time
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

SynthesizeFn = Callable[[str], Tuple[np.ndarray, Dict[str, float]]]


def _cpu_name() -> str:
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if system == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def describe_device(device: str) -> Dict[str, object]:
    """Hardware/software description to report next to every timing number."""
    import torch

    info: Dict[str, object] = {
        "device": str(device), "cpu": _cpu_name(), "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__, "python": platform.python_version(), "os": platform.platform(),
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index or 0
        props = torch.cuda.get_device_properties(index)
        info.update(gpu=torch.cuda.get_device_name(index), cuda=torch.version.cuda,
                    gpu_memory_gb=round(props.total_memory / 1e9, 1))
    return info


def device_tag(device: str) -> str:
    """'cuda:1' -> 'cuda', 'cpu' -> 'cpu': used to name latency files per device type."""
    return str(device).split(":")[0]


def _synchroniser(device: str) -> Callable[[], None]:
    import torch
    if str(device).startswith("cuda"):
        return lambda: torch.cuda.synchronize(device)
    if str(device).startswith("mps"):
        return torch.mps.synchronize
    return lambda: None


def measure_latency(synthesize: SynthesizeFn, texts: Sequence[str], sample_rate: int,
                    device: str = "cpu", n_warmup: int = 3, n_repeats: int = 5) -> pd.DataFrame:
    """Times ``synthesize(text) -> (audio, info)`` and returns one row per timed call."""
    if not texts:
        raise ValueError("measure_latency needs at least one text")
    sync = _synchroniser(device)
    for i in range(n_warmup):
        synthesize(texts[i % len(texts)])
    rows: List[Dict[str, float]] = []
    for text_id, text in enumerate(texts):
        for repeat in range(n_repeats):
            sync()
            start = time.perf_counter()
            audio, info = synthesize(text)
            sync()
            latency = time.perf_counter() - start
            audio_s = len(audio) / float(sample_rate)
            row = {"text_id": text_id, "repeat": repeat, "n_chars": len(text),
                   "latency_s": latency, "audio_s": audio_s,
                   "rtf": latency / audio_s if audio_s > 0 else float("nan")}
            row.update({k: float(v) for k, v in (info or {}).items() if k.endswith("_s")})
            rows.append(row)
    return pd.DataFrame(rows)


def summarise_latency(runs: pd.DataFrame) -> Dict[str, float]:
    """Mean/std/median/p90 of latency and RTF, plus the aggregate RTF (total time / total audio)."""
    summary = {
        "n_texts": int(runs["text_id"].nunique()),
        "n_repeats": int(runs.groupby("text_id").size().max()),
        "latency_mean_s": float(runs["latency_s"].mean()),
        "latency_std_s": float(runs["latency_s"].std(ddof=1)) if len(runs) > 1 else 0.0,
        "latency_median_s": float(runs["latency_s"].median()),
        "latency_p90_s": float(runs["latency_s"].quantile(0.9)),
        "rtf_mean": float(runs["rtf"].mean()),
        "rtf_std": float(runs["rtf"].std(ddof=1)) if len(runs) > 1 else 0.0,
        "rtf_median": float(runs["rtf"].median()),
        "rtf_aggregate": float(runs["latency_s"].sum() / runs["audio_s"].sum()),
    }
    if "frontend_s" in runs.columns:
        summary["frontend_share"] = float(runs["frontend_s"].sum() / runs["latency_s"].sum())
    return summary
