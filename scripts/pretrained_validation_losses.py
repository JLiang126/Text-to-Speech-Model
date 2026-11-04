#!/usr/bin/env python
"""Validation losses of the pretrained model before any fine-tuning ("epoch 0").

Runs the validation pass of scripts/train_model.py (its ``evaluate`` function, unchanged) on the
pretrained checkpoint, once per starting speaker, on the round-2 validation split. The training
logs start at epoch 1, so this gives the loss curves their starting point. Run from the repo root:

  python scripts/pretrained_validation_losses.py --speaker_ids 7 24

The settings (sample rate, loss weights, seed...) come from a round-2 run's training_config.json, and
the validation clips from that run's saved split, so epoch 0 is scored exactly like epochs 1-15.
Two differences, both negligible for a plot: the runs evaluated in bf16 on a GPU, this script in
float32 on the CPU; and the losses include VITS's random segment and noise sampling, as every epoch does.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

RUN_CONFIG = "outputs_v2/best_speaker_embedding/run/training_config.json"
OUT = "outputs_v2/pretrained_validation_losses.csv"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=RUN_CONFIG, help="a round-2 run's training_config.json (default: %(default)s)")
    p.add_argument("--speaker_ids", type=int, nargs="+", default=[7, 24], help="starting speakers to score")
    p.add_argument("--split_dir", help="folder with the run's train/val/test manifests "
                                       "(default: the splits/ folder next to the config's run/ folder)")
    p.add_argument("--data_root", help="LJSpeech folder (default: the config's data_root)")
    p.add_argument("--model", help="pretrained checkpoint (default: the config's model_name_or_path)")
    p.add_argument("--batch_size", type=int, default=16, help="validation batch size (default: %(default)s)")
    p.add_argument("--num_workers", type=int, default=0, help="data-loading workers (default: %(default)s)")
    p.add_argument("--out", default=OUT, help="CSV to write (default: %(default)s)")
    return p.parse_args(argv)


def load_run_config(args: argparse.Namespace):
    """The run's DataConfig, pointed at its own split and set up for a CPU-friendly validation pass."""
    from scripts.DataConfig import DataConfig, load_config

    config = Path(args.config)
    # training_config.json is JSON: read it as such (YAML would read "4e-05" as a string).
    settings = json.loads(config.read_text(encoding="utf-8")) if config.suffix == ".json" else load_config(config)
    split_dir = Path(args.split_dir) if args.split_dir else config.parent.parent / "splits"
    settings.update(output_split_dir=str(split_dir), output_dir=str(config.parent.parent),
                    per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
                    num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                    do_eval=True, max_eval_samples=None)
    if args.data_root:
        settings["data_root"] = args.data_root
    if args.model:
        settings["model_name_or_path"] = args.model
    if not torch.cuda.is_available():
        settings["mixed_precision"] = "no"
    cfg = DataConfig(**settings)
    cfg.validate(output_mode="restart")
    return cfg


def pretrained_losses(cfg, speaker_id: int) -> Dict[str, float]:
    """The losses train_model.py logs, for the pretrained model with this speaker, on the validation split."""
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from scripts.AudioProcessor import AudioProcessor
    from scripts.model import VitsModelForPreTraining
    from scripts.monotonic_alignment import MonotonicAlignment
    from scripts.TextProcessor import configure_word_mismatch_warnings, load_vctk_tokenizer
    from scripts.train_model import evaluate, load_data

    configure_word_mismatch_warnings(cfg.suppress_word_mismatch_warnings)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    set_seed(cfg.seed)
    cfg.speaker_id = speaker_id
    tokenizer = load_vctk_tokenizer(cfg.model_name_or_path)
    model = VitsModelForPreTraining.from_pretrained(cfg.model_name_or_path)
    if not 0 <= speaker_id < model.config.num_speakers:
        raise ValueError(f"speaker_id must be between 0 and {model.config.num_speakers - 1}; got {speaker_id}")
    segment_frames = model.segment_size
    discriminator = model.discriminator
    del model.discriminator
    # The weight normalisation train_model.py applies before training (it does not change the outputs).
    model.decoder.apply_weight_norm()
    for flow in model.flow.flows:
        torch.nn.utils.weight_norm(flow.conv_pre)
        torch.nn.utils.weight_norm(flow.conv_post)
    for disc in discriminator.discriminators:
        disc.apply_weight_norm()

    _, val_loader = load_data(cfg, tokenizer, cfg.num_workers, speaker_id, segment_frames)
    model, discriminator, val_loader = accelerator.prepare(model, discriminator, val_loader)
    losses = evaluate(model, discriminator, val_loader, MonotonicAlignment().maximum_path, AudioProcessor(cfg),
                      speaker_id, segment_frames, accelerator, cfg)
    return {**losses, "n_val": len(val_loader.dataset)}


def main(argv: Optional[List[str]] = None) -> pd.DataFrame:
    from scripts.evaluation.synthesizer import _use_bundled_espeak

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _use_bundled_espeak()  # the tokenizer phonemises with espeak-ng; no system install needed
    cfg = load_run_config(args)
    rows = []
    for speaker_id in args.speaker_ids:
        logging.info("Scoring the pretrained model with speaker %d on %s", speaker_id, cfg.output_split_dir)
        rows.append({"speaker_id": speaker_id, "epoch": 0, **pretrained_losses(cfg, speaker_id)})
        logging.info("speaker %d: %s", speaker_id, {k: round(v, 4) for k, v in rows[-1].items()
                                                     if isinstance(v, float)})
    table = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Wrote {out}")
    return table


if __name__ == "__main__":
    main()
