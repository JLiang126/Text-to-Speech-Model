#!/usr/bin/env python
"""Objective evaluation of one system on the test split.

A "system" is either a checkpoint (expA, expB, or the pretrained model before
fine-tuning, which is the floor) or --ground_truth (the real LJSpeech recordings,
which is the ceiling every metric is read against).

Run from the repo root:

  python scripts/evaluate.py --exp ground_truth --ground_truth
  python scripts/evaluate.py --exp pretrained --config vits-main/configs/vctk_base.json \
      --checkpoint checkpoints/pretrained_vctk.pth --speaker_id <ID>
  python scripts/evaluate.py --exp expA --config configs/expA.json \
      --checkpoint checkpoints/expA/G_10000.pth --speaker_id <ID> --out_dir results/expA_step10000
  # CPU timing for the same system (adds latency_cpu.json, leaves the other results alone)
  python scripts/evaluate.py --exp expA ... --out_dir results/expA_step10000 --metrics rtf --device cpu
  # quick smoke test: 5 sentences, metrics that need no downloads
  python scripts/evaluate.py --exp expA ... --limit 5 --metrics mcd,rtf

Writes to the output folder (default results/<exp>/):
  per_utterance.csv       one row per test sentence (every metric + the ASR transcript)
  summary.json            means with 95% bootstrap CIs, corpus WER/CER, device, settings, versions
  latency_<device>.json   latency/RTF summary for that device type (cuda or cpu)
  rtf_runs_<device>.csv   every timed synthesis call
  wavs/<utt_id>.wav       the synthesised test sentences (checkpoint systems only)

Metrics (choose with --metrics, default all):
  mcd   mel-cepstral distortion vs the real recording, DTW-aligned (dB, lower is better)
  wer   word/character error rate of a Whisper transcript (lower is better)
  secs  ECAPA speaker similarity to the LJSpeech voice (-1..1, higher is better)
  utmos predicted naturalness, about 1-5 (higher is better)
  rtf   latency and real-time factor on the 12 fixed prompts (lower is faster)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Callable, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # lets `python scripts/evaluate.py` import the `scripts` package
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from scripts.evaluation.audio import (EVAL_PEAK, EVAL_TRIM_DB, load_audio, prepare,  # noqa: E402
                                      resample, save_wav)
from scripts.evaluation.manifest import load_manifest  # noqa: E402
from scripts.evaluation.metrics import (ASR_SR, SpeakerEncoder, UTMOSPredictor, WhisperASR,  # noqa: E402
                                        cosine_similarity, get_text_normaliser, mcd_dtw,
                                        word_char_errors)
from scripts.evaluation.prompts import prompt_texts  # noqa: E402
from scripts.evaluation.speakers import reference_centroid  # noqa: E402
from scripts.evaluation.stats import bootstrap_ci, bootstrap_ratio_ci  # noqa: E402
from scripts.evaluation.synthesizer import VITSSynthesizer, seed_for  # noqa: E402
from scripts.evaluation.timing import (describe_device, device_tag, measure_latency,  # noqa: E402
                                       summarise_latency)

ALL_METRICS = ("mcd", "wer", "secs", "utmos", "rtf")
PER_UTTERANCE_METRICS = {"mcd", "wer", "secs", "utmos"}
SUMMARY_COLUMNS = ("mcd_dtw", "dur_ratio", "wer", "cer", "secs", "secs_paired", "utmos")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", required=True, help="system name, e.g. expA (also the default output folder)")
    p.add_argument("--ground_truth", action="store_true", help="score the real recordings (the ceiling)")
    p.add_argument("--config", help="VITS JSON config the checkpoint was trained with")
    p.add_argument("--checkpoint", help="generator checkpoint G_*.pth")
    p.add_argument("--step", type=int, default=None, help="training step (default: parsed from G_<step>.pth)")
    p.add_argument("--vits_dir", default=os.environ.get("VITS_DIR", "vits-main"),
                   help="official VITS code folder (default: $VITS_DIR or ./vits-main)")
    p.add_argument("--speaker_id", type=int, default=None, help="speaker ID, multi-speaker checkpoints only")
    p.add_argument("--frontend", choices=["repo", "vits"], default="repo",
                   help="repo = scripts/TextProcessor.py (same as training); vits = official text module")
    p.add_argument("--manifest", default="dataset/splits/test_manifest.csv", help="test split manifest")
    p.add_argument("--ref_manifest", default=None,
                   help="real clips for the speaker reference (default: train_manifest.csv beside --manifest)")
    p.add_argument("--n_ref", type=int, default=50, help="number of reference clips for the speaker centroid")
    p.add_argument("--data_root", default="dataset/LJSpeech-1.1", help="fallback folder for wav paths")
    p.add_argument("--metrics", default=",".join(ALL_METRICS), help=f"comma list from {ALL_METRICS}")
    p.add_argument("--whisper_model", default="small.en", help="Whisper size for WER (tiny.en ... large-v3)")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    p.add_argument("--cpu_threads", type=int, default=None, help="fix torch's CPU thread count (CPU timing)")
    p.add_argument("--seed", type=int, default=1234, help="base sampling seed (per-sentence seeds derive from it)")
    p.add_argument("--limit", type=int, default=None, help="only the first N test sentences (smoke tests)")
    p.add_argument("--n_warmup", type=int, default=3, help="untimed warm-up calls before timing")
    p.add_argument("--n_repeats", type=int, default=5, help="timed repeats per sentence")
    p.add_argument("--rtf_texts", choices=["prompts", "test"], default="prompts",
                   help="time the 12 fixed prompts or the first 10 test sentences")
    p.add_argument("--no_save_wavs", action="store_true", help="don't keep the synthesised test wavs")
    p.add_argument("--out_dir", default=None, help="default: results/<exp>")
    return p


def resolve_device(requested: str) -> str:
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def try_load(name: str, factory: Callable[[], object], failures: Dict[str, str]) -> Optional[object]:
    """Loads a metric model. On failure it records why and returns None, so that metric becomes NaN."""
    try:
        return factory()
    except Exception as exc:  # no internet, missing package, incompatible versions...
        failures[name] = f"{type(exc).__name__}: {exc}"
        print(f"[warn] '{name}' unavailable, its columns will be NaN -> {failures[name][:200]}")
        return None


def package_versions() -> Dict[str, str]:
    versions = {}
    for pkg in ("torch", "numpy", "librosa", "jiwer", "openai-whisper", "speechbrain", "phonemizer",
                "espeakng-loader", "inflect"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def summarise(per_utt: pd.DataFrame, seed: int) -> Dict[str, Dict[str, float]]:
    summary = {}
    for col in SUMMARY_COLUMNS:
        if col in per_utt.columns and per_utt[col].notna().any():
            mean, lo, hi = bootstrap_ci(per_utt[col].to_numpy(dtype=float), seed=seed)
            summary[col] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": int(per_utt[col].notna().sum())}
    return summary


def corpus_rates(per_utt: pd.DataFrame, seed: int) -> Dict[str, Dict[str, float]]:
    """Corpus WER/CER = total errors / total reference words (characters), with bootstrap CIs."""
    rates = {}
    for name, err, ref in (("wer", "word_errors", "ref_words"), ("cer", "char_errors", "ref_chars")):
        if err in per_utt.columns and per_utt[err].notna().any():
            point, lo, hi = bootstrap_ratio_ci(per_utt[err].to_numpy(dtype=float),
                                               per_utt[ref].to_numpy(dtype=float), seed=seed)
            rates[name] = {"value": point, "ci_low": lo, "ci_high": hi}
    return rates


def score_test_set(args, test: pd.DataFrame, synth: Optional[VITSSynthesizer], metrics: set,
                   device: str, out_dir: Path, failures: Dict[str, str]) -> pd.DataFrame:
    """Synthesises (or loads) every test sentence and computes the per-sentence metrics."""
    asr = try_load("wer", lambda: WhisperASR(args.whisper_model, device), failures) if "wer" in metrics else None
    spk = try_load("secs", lambda: SpeakerEncoder(device), failures) if "secs" in metrics else None
    mos = try_load("utmos", lambda: UTMOSPredictor(device), failures) if "utmos" in metrics else None
    normaliser = get_text_normaliser()

    centroid = None
    if spk is not None:  # "the LJSpeech voice" = mean embedding of real *train* clips (never test clips)
        ref_path = Path(args.ref_manifest) if args.ref_manifest else Path(args.manifest).with_name("train_manifest.csv")
        centroid = reference_centroid(spk.embed, ref_path, args.data_root, args.n_ref, args.seed)

    try:
        from tqdm import tqdm
    except ImportError:  # tqdm is optional
        def tqdm(iterable, **_):
            return iterable

    rows = []
    for row in tqdm(list(test.itertuples(index=False)), desc=args.exp):
        gt, gt_sr = load_audio(row.wav_path)
        record: Dict[str, object] = {"utt_id": row.utt_id, "text": row.ref_text}
        if synth is None:
            audio, sr = gt, gt_sr
        else:
            audio, info = synth.synthesize(row.ref_text, seed=seed_for(row.utt_id, args.seed))
            sr = synth.sample_rate
            record.update(synth_time_s=info["frontend_s"] + info["model_s"], n_tokens=info["n_tokens"])
            if not args.no_save_wavs:
                save_wav(out_dir / "wavs" / f"{row.utt_id}.wav", audio, sr)

        audio_p, gt_p = prepare(audio, sr), prepare(gt, gt_sr)
        record["duration_s"] = len(audio_p) / sr
        record["ref_duration_s"] = len(gt_p) / gt_sr
        if "mcd" in metrics and synth is not None:
            record.update(mcd_dtw(resample(gt_p, gt_sr, sr), audio_p, sr))
        audio_16k = resample(audio_p, sr, ASR_SR)
        if asr is not None:
            hypothesis = asr.transcribe(audio_16k)
            record["asr_text"] = hypothesis
            record.update(word_char_errors(row.ref_text, hypothesis, normaliser))
        if spk is not None:
            embedding = spk.embed(audio_16k)
            record["secs"] = cosine_similarity(embedding, centroid)
            if synth is not None:
                record["secs_paired"] = cosine_similarity(embedding, spk.embed(resample(gt_p, gt_sr, ASR_SR)))
        if mos is not None:
            record["utmos"] = mos.score(audio_16k)
        rows.append(record)
    return pd.DataFrame(rows)


def main(argv=None) -> Dict[str, object]:
    args = build_parser().parse_args(argv)
    metrics = {m.strip() for m in args.metrics.split(",") if m.strip()}
    unknown = metrics - set(ALL_METRICS)
    if unknown:
        raise SystemExit(f"Unknown metrics {sorted(unknown)}; choose from {ALL_METRICS}")
    if not args.ground_truth and not (args.config and args.checkpoint):
        raise SystemExit("Checkpoint systems need --config and --checkpoint (or use --ground_truth).")
    if args.ground_truth and metrics == {"rtf"}:
        raise SystemExit("Ground truth has no synthesis to time; drop --metrics rtf.")

    import torch
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    device = resolve_device(args.device)
    tag = device_tag(device)
    out_dir = Path(args.out_dir or Path("results") / args.exp)
    out_dir.mkdir(parents=True, exist_ok=True)

    synth = None
    if not args.ground_truth:
        synth = VITSSynthesizer(args.config, args.checkpoint, vits_dir=args.vits_dir, device=device,
                                speaker_id=args.speaker_id, frontend=args.frontend)

    need_test_set = bool(metrics & PER_UTTERANCE_METRICS) or ("rtf" in metrics and args.rtf_texts == "test")
    test = load_manifest(args.manifest, args.data_root, args.limit) if need_test_set else None
    failures: Dict[str, str] = {}
    step = args.step
    if step is None and args.checkpoint:
        match = re.search(r"_(\d+)\.pth$", str(args.checkpoint))
        step = int(match.group(1)) if match else None

    latency = None
    if "rtf" in metrics and synth is not None:
        texts = prompt_texts() if args.rtf_texts == "prompts" else test["ref_text"].tolist()[:10]
        runs = measure_latency(lambda t: synth.synthesize(t, seed=args.seed), texts, synth.sample_rate,
                               device=device, n_warmup=args.n_warmup, n_repeats=args.n_repeats)
        runs.to_csv(out_dir / f"rtf_runs_{tag}.csv", index=False)
        latency = {**summarise_latency(runs), "device": describe_device(device), "texts": args.rtf_texts,
                   "n_warmup": args.n_warmup, "experiment": args.exp, "step": step}
        (out_dir / f"latency_{tag}.json").write_text(json.dumps(latency, indent=2), encoding="utf-8")
        print(f"[info] {args.exp} on {latency['device'].get('gpu', latency['device']['cpu'])}: "
              f"latency {latency['latency_mean_s']:.3f} s (sd {latency['latency_std_s']:.3f}), "
              f"RTF {latency['rtf_mean']:.3f} (sd {latency['rtf_std']:.3f})")

    if not metrics & PER_UTTERANCE_METRICS:  # timing-only run: leave per_utterance.csv / summary.json alone
        print(f"[info] wrote {out_dir}/latency_{tag}.json and rtf_runs_{tag}.csv")
        return {"latency": latency}

    print(f"[info] {args.exp}: {len(test)} test sentences from {args.manifest} on {device}")
    per_utt = score_test_set(args, test, synth, metrics, device, out_dir, failures)
    per_utt.to_csv(out_dir / "per_utterance.csv", index=False)
    summary = {
        "experiment": args.exp, "ground_truth": bool(args.ground_truth), "step": step,
        "checkpoint": args.checkpoint, "config": args.config, "speaker_id": args.speaker_id,
        "manifest": str(args.manifest), "n_utterances": int(len(per_utt)),
        "metrics": summarise(per_utt, args.seed), "corpus": corpus_rates(per_utt, args.seed),
        "latency": {tag: latency} if latency else {}, "device": describe_device(device),
        "settings": {"seed": args.seed, "frontend": args.frontend, "whisper_model": args.whisper_model,
                     "n_ref_clips": args.n_ref, "peak": EVAL_PEAK, "trim_db": EVAL_TRIM_DB,
                     "noise_scale": synth.noise_scale if synth else None,
                     "noise_scale_w": synth.noise_scale_w if synth else None,
                     "length_scale": synth.length_scale if synth else None,
                     "expand_numbers": synth.expand_numbers if synth else None},
        "unavailable_metrics": failures, "versions": package_versions(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary)
    print(f"[info] wrote {out_dir}/per_utterance.csv and summary.json")
    return summary


def print_summary(summary: Dict[str, object]) -> None:
    print(f"\n=== {summary['experiment']} ({summary['n_utterances']} sentences) ===")
    for name, s in summary["metrics"].items():
        print(f"{name:12s} {s['mean']:8.3f}  95% CI [{s['ci_low']:.3f}, {s['ci_high']:.3f}]  n={s['n']}")
    for name, s in summary["corpus"].items():
        print(f"corpus {name:5s} {s['value']:8.3f}  95% CI [{s['ci_low']:.3f}, {s['ci_high']:.3f}]")


if __name__ == "__main__":
    main()
