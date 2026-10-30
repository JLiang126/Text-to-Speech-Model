#!/usr/bin/env python
"""Ranks every speaker of the VCTK-pretrained VITS by how much it sounds like LJSpeech.

Used to choose the starting speaker for the speaker-similarity experiment: fine-tune
from the closest and the furthest VCTK voice (ideally also the median one), everything
else identical. Method and caveats: scripts/evaluation/speakers.py.

Run from the repo root. Quick CPU check on a small subset (n_utts x 109 syntheses):

  python scripts/rank_speakers.py --n_utts 5

Full run on a GPU (every training sentence; --resume continues a stopped run):

  python scripts/rank_speakers.py --n_utts 0 --device cuda --resume

The model defaults to the checkpoint fine-tuning starts from
(ylacombe/vits-vctk-with-discriminator, downloaded on first use). Official VITS format:
  --config vits-main/configs/vctk_base.json --checkpoint checkpoints/pretrained_vctk.pth

Writes to --out_dir (default results/speaker_ranking/):
  speaker_ranking.csv   one row per speaker index, closest first: mean paired cosine with
                        95% CI, rank CI, P(closest), P(furthest), centroid cosine, pitch,
                        VCTK speaker ID
  summary.json          closest / median / furthest picks, reliability, settings
  speaker_ranking.png   scores with CIs, and score against pitch
  embeddings.npz        every ECAPA embedding (what --resume reads)
  wavs/                 the first sentence spoken by the three picks, plus the real recording
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # lets `python scripts/rank_speakers.py` import the `scripts` package
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.evaluation.audio import load_audio, prepare, resample, save_wav  # noqa: E402
from scripts.evaluation.manifest import load_manifest  # noqa: E402
from scripts.evaluation.metrics import ASR_SR, SpeakerEncoder  # noqa: E402
from scripts.evaluation.speakers import (VCTK_N_SPEAKERS, centroid_similarity, f0_median,  # noqa: E402
                                         load_vctk_speakers, paired_similarity, pick_conditions,
                                         rank_speakers, split_half_reliability)
from scripts.evaluation.synthesizer import (add_model_arguments, model_given, seed_for,  # noqa: E402
                                            synthesizer_from_args)

DEFAULT_MODEL = "ylacombe/vits-vctk-with-discriminator"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(p, with_speaker=False)
    p.add_argument("--manifest", default="dataset/splits/train_manifest.csv",
                   help="sentences and their real recordings (keep the train split: test must stay unseen)")
    p.add_argument("--data_root", default="dataset/LJSpeech-1.1", help="fallback folder for wav paths")
    p.add_argument("--n_utts", type=int, default=5, help="sentences per speaker, sampled with --seed (0 = all)")
    p.add_argument("--speakers", default="all", help="speaker indices to rank: all, or e.g. 0-9,20,33")
    p.add_argument("--include_untrained", action="store_true",
                   help="also rank embeddings that were never trained (index 108 of the VCTK model)")
    p.add_argument("--n_boot", type=int, default=1000, help="bootstrap resamples for the CIs")
    p.add_argument("--f0_utts", type=int, default=3, help="sentences used for the pitch column (0 = skip)")
    p.add_argument("--seed", type=int, default=1234, help="sentence sampling and per-sentence synthesis seed")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    p.add_argument("--resume", action="store_true", help="continue from embeddings.npz in --out_dir")
    p.add_argument("--save_every", type=int, default=20, help="save embeddings every N sentences")
    p.add_argument("--out_dir", default="results/speaker_ranking")
    return p


def parse_speakers(spec: str, n_speakers: int) -> List[int]:
    if spec.strip().lower() == "all":
        return list(range(n_speakers))
    ids = set()
    for part in filter(None, (s.strip() for s in spec.split(","))):
        if "-" in part:
            first, last = part.split("-", 1)
            ids.update(range(int(first), int(last) + 1))
        else:
            ids.add(int(part))
    bad = sorted(i for i in ids if not 0 <= i < n_speakers)
    if bad:
        raise SystemExit(f"--speakers must be within 0-{n_speakers - 1}; got {bad}")
    return sorted(ids)


def resolve_device(requested: str) -> str:
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _save_cache(path: Path, **arrays) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def plot_ranking(table: pd.DataFrame, summary: Dict[str, object], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, blue_light, muted, ink, ink2 = "#2a78d6", "#86b6ef", "#898781", "#0b0b0b", "#52514e"
    has_f0 = bool(table["f0_median_hz"].notna().any())
    fig, axes = plt.subplots(1, 2 if has_f0 else 1, figsize=(13 if has_f0 else 8, 4.4), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    trained = table["eligible"].to_numpy()
    x = np.arange(1, len(table) + 1)
    y = table["similarity"].to_numpy()

    ax = axes[0, 0]
    yerr = np.clip([y - table["ci_low"].to_numpy(), table["ci_high"].to_numpy() - y], 0, None)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=blue_light, elinewidth=1.0, zorder=2)
    ax.scatter(x[trained], y[trained], s=16, color=blue, zorder=3, label="trained VCTK speaker")
    if (~trained).any():
        ax.scatter(x[~trained], y[~trained], s=16, color=muted, zorder=3, label="never-trained embedding")
    for label, info in summary["conditions"].items():
        row = table.index[table["speaker_index"] == info["speaker_index"]][0]
        ax.annotate(f"{label}: {info['speaker_index']} ({info.get('vctk_id') or 'n/a'})", (x[row], y[row]),
                    xytext=(6, 8 if label != "furthest" else -14), textcoords="offset points",
                    fontsize=8, color=ink)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("speaker, sorted by similarity (rank)", color=ink2)
    ax.set_ylabel("mean paired cosine to LJSpeech (ECAPA)", color=ink2)
    ax.set_title(f"Similarity of each VCTK voice to LJSpeech\n"
                 f"{summary['n_utts']} sentences per speaker, 95% bootstrap CI", fontsize=10, color=ink)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    if has_f0:
        ax2 = axes[0, 1]
        f0 = table["f0_median_hz"].to_numpy()
        ax2.scatter(f0[trained], y[trained], s=16, color=blue, zorder=3, label="trained VCTK speaker")
        if (~trained).any():
            ax2.scatter(f0[~trained], y[~trained], s=16, color=muted, zorder=3, label="never-trained embedding")
        lj_f0 = summary["pitch"]["lj_f0_median_hz"]
        if lj_f0 == lj_f0:  # not NaN
            ax2.axvline(lj_f0, color=ink2, lw=1, ls="--")
            ax2.annotate(f"LJSpeech {lj_f0:.0f} Hz", (lj_f0, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(4, -12), textcoords="offset points", fontsize=8, color=ink2)
        rho = summary["pitch"]["spearman_similarity_vs_pitch_closeness"]
        ax2.set_xlabel("median pitch of the synthetic voice (Hz)", color=ink2)
        ax2.set_ylabel("mean paired cosine to LJSpeech (ECAPA)", color=ink2)
        ax2.set_title(f"Similarity against pitch\nSpearman rho with pitch closeness to LJSpeech = {rho:.2f}",
                      fontsize=10, color=ink)

    for a in axes.ravel():
        a.set_facecolor("#fcfcfb")
        a.grid(True, color="#e8e7e3", lw=0.6)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(muted)
        a.tick_params(colors=ink2, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main(argv=None) -> Dict[str, object]:
    args = build_parser().parse_args(argv)
    if not model_given(args):
        args.model = DEFAULT_MODEL
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    synth = synthesizer_from_args(args, device, speaker_id=0)
    if synth.n_speakers == 0:
        raise SystemExit("This checkpoint is single-speaker; there are no speakers to rank.")
    model_name = str(args.model or args.checkpoint)
    speakers = parse_speakers(args.speakers, synth.n_speakers)
    manifest = load_manifest(args.manifest, args.data_root)
    if args.n_utts and args.n_utts < len(manifest):
        manifest = manifest.sample(n=args.n_utts, random_state=args.seed)
    manifest = manifest.reset_index(drop=True)
    utt_ids = manifest["utt_id"].astype(str).tolist()
    n_utts, n_spk, n_f0 = len(utt_ids), len(speakers), min(max(args.f0_utts, 0), len(utt_ids))
    print(f"[info] {model_name}: {n_spk} speakers x {n_utts} sentences = {n_spk * n_utts} syntheses on {device}")

    encoder = SpeakerEncoder(device)
    cache = out_dir / "embeddings.npz"
    arrays = None
    if args.resume and cache.is_file():
        saved = np.load(cache, allow_pickle=False)
        same = (saved["utt_ids"].tolist() == utt_ids and saved["speakers"].tolist() == speakers
                and str(saved["model"]) == model_name and saved["f0_synth"].shape[1] == n_f0)
        if not same:
            raise SystemExit(f"--resume: {cache} was made with other sentences/speakers/model/--f0_utts; "
                             f"use the same options or a new --out_dir.")
        arrays = {k: saved[k].copy() for k in ("real", "synth", "done", "f0_real", "f0_synth")}
        print(f"[info] resuming: {int(arrays['done'].sum())}/{n_utts} sentences already done")

    start, n_new = time.perf_counter(), 0
    for i, row in manifest.iterrows():
        if arrays is not None and arrays["done"][i]:
            continue
        audio, sr = load_audio(row.wav_path)
        real16 = resample(prepare(audio, sr), sr, ASR_SR)
        real_emb = encoder.embed(real16)
        if arrays is None:
            dim = real_emb.shape[-1]
            arrays = {"real": np.zeros((n_utts, dim), np.float32),
                      "synth": np.zeros((n_spk, n_utts, dim), np.float32),
                      "done": np.zeros(n_utts, bool),
                      "f0_real": np.full(n_f0, np.nan), "f0_synth": np.full((n_spk, n_f0), np.nan)}
        arrays["real"][i] = real_emb
        if i < n_f0:
            arrays["f0_real"][i] = f0_median(real16, ASR_SR)
        seed = seed_for(row.utt_id, args.seed)  # same noise draw for every speaker of this sentence
        for j, speaker in enumerate(speakers):
            synth.speaker_id = speaker
            wav, _ = synth.synthesize(row.ref_text, seed=seed)
            wav16 = resample(prepare(wav, synth.sample_rate), synth.sample_rate, ASR_SR)
            arrays["synth"][j, i] = encoder.embed(wav16)
            if i < n_f0:
                arrays["f0_synth"][j, i] = f0_median(wav16, ASR_SR)
        arrays["done"][i] = True
        n_new += 1
        done = int(arrays["done"].sum())
        elapsed = time.perf_counter() - start
        print(f"[{done}/{n_utts}] {row.utt_id}  {elapsed / 60:.1f} min, "
              f"about {(n_utts - done) * elapsed / n_new / 60:.1f} min left", flush=True)
        if n_new % args.save_every == 0:
            _save_cache(cache, utt_ids=np.array(utt_ids), speakers=np.array(speakers),
                        model=np.array(model_name), **arrays)
    _save_cache(cache, utt_ids=np.array(utt_ids), speakers=np.array(speakers), model=np.array(model_name),
                **arrays)

    # ---------------------------------------------------------------- ranking
    paired = paired_similarity(arrays["synth"], arrays["real"])
    eligible = np.ones(n_spk, bool)
    vctk = load_vctk_speakers() if synth.n_speakers == VCTK_N_SPEAKERS else None
    if vctk is not None and not args.include_untrained:
        eligible = np.array([s in vctk.index for s in speakers])
    table = rank_speakers(paired, speakers, eligible, n_boot=args.n_boot)
    position = {s: j for j, s in enumerate(speakers)}
    rows = [position[s] for s in table["speaker_index"]]
    table["centroid_similarity"] = centroid_similarity(arrays["synth"], arrays["real"])[rows]
    table["centroid_rank"] = pd.Series(pd.NA, index=table.index, dtype="Int64")
    elig = table["eligible"]
    table.loc[elig, "centroid_rank"] = (table.loc[elig, "centroid_similarity"]
                                        .rank(ascending=False, method="first").astype(int).to_numpy())
    with np.errstate(all="ignore"):
        table["f0_median_hz"] = (np.nanmedian(arrays["f0_synth"], axis=1)[rows] if n_f0
                                 else np.full(len(rows), np.nan))
    speaker_table = synth.speaker_table()
    table["embedding_norm"] = np.linalg.norm(speaker_table[table["speaker_index"]], axis=1)
    if vctk is not None:
        table["vctk_id"] = [vctk["vctk_id"].get(s, "") for s in table["speaker_index"]]
        table["n_vctk_train_utts"] = [int(vctk["n_train_utts"].get(s, 0)) for s in table["speaker_index"]]
    table.to_csv(out_dir / "speaker_ranking.csv", index=False)

    # ---------------------------------------------------------------- summary
    from scipy.stats import spearmanr
    with np.errstate(all="ignore"):
        lj_f0 = float(np.nanmedian(arrays["f0_real"])) if n_f0 else float("nan")
    ranked = table[elig]
    agreement = (float(spearmanr(ranked["similarity"], ranked["centroid_similarity"]).statistic)
                 if len(ranked) > 2 else float("nan"))
    pitch_ok = ranked["f0_median_hz"].notna()
    pitch_rho = (float(spearmanr(ranked.loc[pitch_ok, "similarity"],
                                 -(ranked.loc[pitch_ok, "f0_median_hz"] - lj_f0).abs()).statistic)
                 if pitch_ok.sum() > 2 and lj_f0 == lj_f0 else float("nan"))
    conditions = {}
    for label, index in pick_conditions(table).items():
        row = table[table["speaker_index"] == index].iloc[0]
        conditions[label] = {"speaker_index": int(index), "vctk_id": row.get("vctk_id", None),
                             "similarity": float(row["similarity"]),
                             "ci": [float(row["ci_low"]), float(row["ci_high"])],
                             "rank_ci": [float(row["rank_ci_low"]), float(row["rank_ci_high"])],
                             "p_closest": float(row["p_closest"]), "p_furthest": float(row["p_furthest"]),
                             "f0_median_hz": float(row["f0_median_hz"])}
    summary = {
        "model": model_name, "n_speakers_in_model": synth.n_speakers, "n_speakers_ranked": int(elig.sum()),
        "excluded_untrained": [int(s) for s in table.loc[~elig, "speaker_index"]],
        "n_utts": n_utts, "manifest": str(args.manifest), "conditions": conditions,
        "reliability": split_half_reliability(paired, eligible),
        "agreement_paired_vs_centroid_spearman": agreement,
        "pitch": {"lj_f0_median_hz": lj_f0, "spearman_similarity_vs_pitch_closeness": pitch_rho,
                  "n_utts": n_f0},
        "settings": {"seed": args.seed, "n_boot": args.n_boot, "device": device,
                     "noise_scale": synth.noise_scale, "noise_scale_w": synth.noise_scale_w,
                     "length_scale": synth.length_scale, "frontend": synth.frontend},
        "runtime_min": round((time.perf_counter() - start) / 60, 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    plot_ranking(table, summary, out_dir / "speaker_ranking.png")

    # one sentence by each pick + the real recording, for a listening check
    wav_dir = out_dir / "wavs"
    first = manifest.iloc[0]
    audio, sr = load_audio(first.wav_path)
    save_wav(wav_dir / f"real_{first.utt_id}.wav", audio, sr)
    for label, info in conditions.items():
        synth.speaker_id = info["speaker_index"]
        wav, _ = synth.synthesize(first.ref_text, seed=seed_for(first.utt_id, args.seed))
        save_wav(wav_dir / f"{label}_spk{info['speaker_index']}_{first.utt_id}.wav", wav, synth.sample_rate)

    shown = ["rank", "speaker_index", "vctk_id", "similarity", "ci_low", "ci_high", "p_closest", "p_furthest",
             "f0_median_hz"]
    shown = [c for c in shown if c in table.columns]
    print("\nClosest 5:\n" + ranked[shown].head(5).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nFurthest 5:\n" + ranked[shown].tail(5).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    rel = summary["reliability"]
    print(f"\nSplit-half reliability: Spearman {rel['split_half_spearman']:.3f} "
          f"(Spearman-Brown {rel['spearman_brown']:.3f}); paired vs centroid ranking agree at {agreement:.3f}")
    if summary["excluded_untrained"]:
        print(f"Excluded (embedding never trained): {summary['excluded_untrained']}")
    for label, info in conditions.items():
        print(f"{label:8s} speaker {info['speaker_index']:3d} ({info['vctk_id'] or 'n/a'}) "
              f"cos {info['similarity']:.3f} [{info['ci'][0]:.3f}, {info['ci'][1]:.3f}]  "
              f"pitch {info['f0_median_hz']:.0f} Hz")
    print(f"[info] wrote {out_dir}/speaker_ranking.csv, summary.json, speaker_ranking.png, wavs/")
    return {"table": table, "summary": summary}


if __name__ == "__main__":
    main()
