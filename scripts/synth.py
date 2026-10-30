#!/usr/bin/env python
"""Single "text -> .wav" entry point.

Run from the repo root:

  # one sentence (Hugging Face format: a finetune-hf-vits output folder or a hub id)
  python scripts/synth.py --text "Hello world." --model runs/expA --speaker_id <ID> --out wav_examples/hello.wav

  # the 12 fixed qualitative prompts -> wav_examples/expA_01.wav ... expA_12.wav (+ expA_prompts.json)
  python scripts/synth.py --prompts --exp expA --model runs/expA --speaker_id <ID> --out_dir wav_examples

  # official VITS format instead of --model
  python scripts/synth.py --text "Hello world." --config configs/expA.json \
      --checkpoint checkpoints/expA/G_latest.pth --speaker_id <ID>

--speaker_id is only for multi-speaker checkpoints (e.g. the VCTK-pretrained model); use the ID
the LJSpeech clips were fine-tuned with. Digits, money, % and & are spelled out before phonemisation.
The official VITS code folder defaults to $VITS_DIR or ./vits-main.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # lets `python scripts/synth.py` import the `scripts` package
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.audio import save_wav  # noqa: E402
from scripts.evaluation.prompts import PROMPTS  # noqa: E402
from scripts.evaluation.synthesizer import (LENGTH_SCALE, NOISE_SCALE, NOISE_SCALE_W,  # noqa: E402
                                            add_model_arguments, model_given, synthesizer_from_args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--text", help="sentence to synthesise")
    what.add_argument("--prompts", action="store_true", help="synthesise the 12 fixed qualitative prompts")
    add_model_arguments(p)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    p.add_argument("--seed", type=int, default=1234, help="sampling seed (same seed -> same audio)")
    p.add_argument("--noise_scale", type=float, default=NOISE_SCALE)
    p.add_argument("--noise_scale_w", type=float, default=NOISE_SCALE_W)
    p.add_argument("--length_scale", type=float, default=LENGTH_SCALE, help=">1 slower, <1 faster")
    p.add_argument("--out", default="wav_examples/output.wav", help="output path for --text")
    p.add_argument("--exp", default="expA", help="file prefix for --prompts (expA -> expA_01.wav)")
    p.add_argument("--out_dir", default="wav_examples", help="output folder for --prompts")
    return p


def resolve_device(requested: str) -> str:
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if not model_given(args):
        raise SystemExit("Pass --model, or --config and --checkpoint.")
    synth = synthesizer_from_args(args, resolve_device(args.device), noise_scale=args.noise_scale,
                                  noise_scale_w=args.noise_scale_w, length_scale=args.length_scale)
    if args.text is not None:
        audio, info = synth.synthesize(args.text, seed=args.seed)
        path = save_wav(args.out, audio, synth.sample_rate)
        print(f"Wrote {path} ({len(audio) / synth.sample_rate:.2f} s audio, "
              f"{info['frontend_s'] + info['model_s']:.2f} s to synthesise)")
        return

    out_dir = Path(args.out_dir)
    index = []
    for i, (category, text) in enumerate(PROMPTS, start=1):
        audio, _ = synth.synthesize(text, seed=args.seed)
        path = save_wav(out_dir / f"{args.exp}_{i:02d}.wav", audio, synth.sample_rate)
        index.append({"file": path.name, "category": category, "text": text,
                      "spoken_text": synth.prepare_text(text)})  # after numbers are spelled out
        print(f"{path.name}  [{category}]  {text}")
    meta = {"experiment": args.exp, "checkpoint": args.checkpoint or args.model, "seed": args.seed,
            "speaker_id": args.speaker_id, "prompts": index}
    (out_dir / f"{args.exp}_prompts.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
