"""
Run all experiments with different configurations.

This script iterates over various experiment configurations defined in YAML files and executes the training and evaluation pipeline for each configuration.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
import subprocess
import sys
from threading import Event
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
CONFIGS = sorted(config_file.name for config_file in CONFIG_DIR.glob("*.yaml"))
GPU_IDS = (0, 1)


def run_experiment(name, gpu_id=None, test_mode='no'):
    config = CONFIG_DIR / name
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    device = f"GPU {gpu_id}" if gpu_id is not None else "Accelerator default device"
    print(f"{device}: running {name}", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_model.py"),
            "--config", str(config),
            "--output-mode", "restart",
            "--test-mode", test_mode,
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )


def simple_run(configs, test_mode='no'):
    for name in configs:
        run_experiment(name, test_mode=test_mode)


def parallel_run(configs, gpu_ids=GPU_IDS, test_mode='no'):
    pending = Queue()
    for name in configs:
        pending.put(name)
    stop_workers = Event()

    def worker(gpu_id):
        while not stop_workers.is_set():
            try:
                name = pending.get_nowait()
            except Empty:
                return
            try:
                if stop_workers.is_set():
                    return
                run_experiment(name, gpu_id, test_mode=test_mode)
            except (OSError, subprocess.CalledProcessError):
                stop_workers.set()
                raise
            finally:
                pending.task_done()

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = [pool.submit(worker, gpu_id) for gpu_id in gpu_ids]
        for future in futures:
            future.result()


if __name__ == "__main__":
    if not CONFIGS:
        raise FileNotFoundError(f"No YAML experiment configs found in {CONFIG_DIR}")

    test_mode = 'no'  # Set to 'yes' to enable test mode, 'no' to disable it

    do_simple_run = False  # Set to False to use parallel run instead
    if do_simple_run:
        simple_run(CONFIGS, test_mode=test_mode)
    else:
        parallel_run(CONFIGS, test_mode=test_mode)
