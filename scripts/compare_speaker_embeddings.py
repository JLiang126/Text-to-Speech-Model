#!/usr/bin/env python
"""Compares each run's speaker vector with the pretrained one: how far did fine-tuning move it?

For the configured speaker_id of every checkpoint (scripts/evaluation/checkpoints.py,
``speaker_embedding_change``):
  rel_l2       ||after - before|| / ||before||
  cosine       cosine(after, before): 1 = same direction (ignores the weight-decay shrink)
  decay_rel_l2 the same change for rows that get no gradient (weight decay only): the floor
  move_vs_gap  the move as a share of the median distance between two different pretrained voices
  nearest      the pretrained speaker whose vector is now closest

Run from the repo root:
  python scripts/compare_speaker_embeddings.py                      # every checkpoint in experiments.csv
  python scripts/compare_speaker_embeddings.py --checkpoint checkpoints/model.best_speaker_embedding.safetensors --speaker_id 7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # lets `python scripts/compare_speaker_embeddings.py` import the package
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from scripts.evaluation.checkpoints import compare_speaker_embeddings  # noqa: E402
from scripts.evaluation.synthesizer import DEFAULT_BASE_MODEL  # noqa: E402


def main(argv: Optional[List[str]] = None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", default="experiments/experiments.csv", help="checkpoints and their speaker_id")
    p.add_argument("--checkpoint", action="append", default=None, help="a checkpoint instead of the table (repeatable)")
    p.add_argument("--speaker_id", type=int, action="append", default=None, help="one per --checkpoint")
    p.add_argument("--base_model", default=DEFAULT_BASE_MODEL, help="the pretrained model the runs started from")
    p.add_argument("--out", default="results/experiments/report/speaker_embedding_change.csv")
    args = p.parse_args(argv)

    if args.checkpoint:
        ids = args.speaker_id or []
        if len(ids) != len(args.checkpoint):
            raise SystemExit("give one --speaker_id per --checkpoint")
        checkpoints = {Path(c).stem: (c, s) for c, s in zip(args.checkpoint, ids)}
    else:
        from scripts.evaluate_experiments import load_experiments, resolve_checkpoint
        table = load_experiments(Path(args.experiments))
        checkpoints = {}
        for name, row in table.iterrows():
            path = resolve_checkpoint(name, row["checkpoint"])
            if Path(path).exists():
                checkpoints[name] = (path, int(row["speaker_id"]))
            else:
                print(f"[skip] {name}: checkpoint not found ({row['checkpoint']})")
    if not checkpoints:
        raise SystemExit("no checkpoints to compare")

    result = compare_speaker_embeddings(checkpoints, args.base_model)
    shown = result.assign(rel_l2=100 * result["rel_l2"], decay_rel_l2=100 * result["decay_rel_l2"],
                          move_vs_gap=100 * result["move_vs_gap"])
    columns = {"experiment": "run", "speaker_id": "speaker", "rel_l2": "rel. L2 %", "cosine": "cosine",
               "decay_rel_l2": "weight decay only %", "move_vs_gap": "% of gap to another voice",
               "nearest_pretrained": "nearest pretrained voice"}
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(shown[list(columns)].rename(columns=columns).to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"[info] wrote {out}")
    return result


if __name__ == "__main__":
    main()
