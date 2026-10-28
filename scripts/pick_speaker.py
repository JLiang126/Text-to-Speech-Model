#!/usr/bin/env python
"""Ranks every speaker in the VCTK-pretrained checkpoint by similarity to the LJSpeech voice.

Run from the repo root (needs internet once, to download the ECAPA speaker model):

  python scripts/pick_speaker.py --config vits-main/configs/vctk_base.json \
      --checkpoint checkpoints/pretrained_vctk.pth --out results/speaker_ranking.csv

The top row is the suggested --speaker_id for fine-tuning with the multi-speaker
config (keep the same ID for every experiment). Each speaker says the same 3
sentences with the same seed, so speakers differ only in voice.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.audio import prepare, resample  # noqa: E402
from scripts.evaluation.metrics import ASR_SR, SpeakerEncoder  # noqa: E402
from scripts.evaluation.prompts import PROMPTS  # noqa: E402
from scripts.evaluation.speakers import rank_speakers, reference_centroid  # noqa: E402
from scripts.evaluation.synthesizer import VITSSynthesizer  # noqa: E402

# neutral sentences from prompts.py: phonetically balanced, formal, question
DEFAULT_TEXTS = [PROMPTS[0][1], PROMPTS[1][1], PROMPTS[6][1]]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="VITS JSON config of the multi-speaker checkpoint")
    p.add_argument("--checkpoint", required=True, help="multi-speaker generator checkpoint")
    p.add_argument("--vits_dir", default=os.environ.get("VITS_DIR", "vits-main"))
    p.add_argument("--ref_manifest", default="dataset/splits/train_manifest.csv",
                   help="real LJSpeech clips that define the target voice")
    p.add_argument("--n_ref", type=int, default=50)
    p.add_argument("--data_root", default="dataset/LJSpeech-1.1")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="results/speaker_ranking.csv")
    args = p.parse_args(argv)

    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    synth = VITSSynthesizer(args.config, args.checkpoint, vits_dir=args.vits_dir, device=device, speaker_id=0)
    if synth.n_speakers == 0:
        raise SystemExit("This checkpoint is single-speaker; there is no speaker to pick.")
    encoder = SpeakerEncoder(device)
    target = reference_centroid(encoder.embed, args.ref_manifest, args.data_root, args.n_ref, args.seed)

    def synthesize(text: str, speaker_id: int):
        synth.speaker_id = speaker_id
        audio, _ = synth.synthesize(text, seed=args.seed)
        return resample(prepare(audio, synth.sample_rate), synth.sample_rate, ASR_SR)

    ranking = rank_speakers(synthesize, encoder.embed, target, DEFAULT_TEXTS, range(synth.n_speakers))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.out, index=False)
    print(ranking.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"[info] wrote {args.out}. Suggested --speaker_id: {int(ranking.loc[0, 'speaker_id'])}")


if __name__ == "__main__":
    main()
