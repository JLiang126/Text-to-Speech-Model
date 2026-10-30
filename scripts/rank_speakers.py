#!/usr/bin/env python
"""Ranks every speaker of the VCTK-pretrained VITS by how much it sounds like LJSpeech.

Used to choose the starting speaker for the speaker-similarity experiment: fine-tune
from the closest and the furthest VCTK voice (ideally also the median one), everything
else identical. Method and caveats: scripts/evaluation/speakers.py.

Run from the repo root. Quick CPU check on a small subset (n_utts x 109 syntheses):

  python scripts/rank_speakers.py --n_utts 5

Full run on a GPU (every training sentence; --resume continues a stopped run):

  python scripts/rank_speakers.py --n_utts 0 --device cuda --resume

Re-analyse a finished run without synthesising again (same options + --resume), or extend it
with more sentences (larger --n_utts + --resume keeps the sentences already done).
Cross-check a few speakers with more sentences of MCD, in a separate folder:

  python scripts/rank_speakers.py --n_utts 5 --speakers 7,24,47 --mcd_utts 5 --out_dir results/speaker_check

The model defaults to the checkpoint fine-tuning starts from
(ylacombe/vits-vctk-with-discriminator, downloaded on first use). Official VITS format:
  --config vits-main/configs/vctk_base.json --checkpoint checkpoints/pretrained_vctk.pth

Writes to --out_dir (default results/speaker_ranking/):
  speaker_ranking.csv   one row per speaker index, closest first: mean paired cosine with
                        95% CI, rank CI, P(closest), P(furthest), centroid cosine, MCD,
                        pitch, VCTK speaker ID
  summary.json          closest / median / furthest picks, picks within each pitch group,
                        reference levels, reliability, agreement between measures, settings
  speaker_ranking.png   scores with CIs coloured by pitch group, and score against pitch
  embeddings.npz        every ECAPA embedding, pitch and MCD value (what --resume reads)
  wavs/                 the first sentence spoken by the picks, plus the real recording
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # lets `python scripts/rank_speakers.py` import the `scripts` package
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.evaluation.audio import load_audio, prepare, resample, save_wav  # noqa: E402
from scripts.evaluation.manifest import load_manifest  # noqa: E402
from scripts.evaluation.metrics import ASR_SR, SpeakerEncoder, mcd_dtw  # noqa: E402
from scripts.evaluation.speakers import (VCTK_N_SPEAKERS, centroid_similarity, f0_median,  # noqa: E402
                                         load_vctk_speakers, paired_similarity, pick_conditions,
                                         pitch_groups, rank_speakers, similarity_baselines,
                                         split_half_reliability)
from scripts.evaluation.synthesizer import (add_model_arguments, model_given, seed_for,  # noqa: E402
                                            synthesizer_from_args)

DEFAULT_MODEL = "ylacombe/vits-vctk-with-discriminator"
CACHE_KEYS = ("real", "synth", "done", "f0_real", "f0_synth", "mcd")


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
    p.add_argument("--f0_threshold", type=float, default=160.0,
                   help="median pitch (Hz) that splits low- and high-pitched voices")
    p.add_argument("--mcd_utts", type=int, default=3,
                   help="sentences used for the MCD cross-check against the real recording (0 = skip)")
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


def _load_cache(path: Path, utt_ids: List[str], speakers: List[int], model_name: str,
                n_f0: int, n_mcd: int) -> Dict[str, np.ndarray]:
    saved = np.load(path, allow_pickle=False)
    saved_ids = saved["utt_ids"].tolist()
    extends = len(saved_ids) < len(utt_ids) and utt_ids[: len(saved_ids)] == saved_ids  # larger --n_utts
    same = ((saved_ids == utt_ids or extends) and saved["speakers"].tolist() == speakers
            and str(saved["model"]) == model_name and saved["f0_synth"].shape[1] == n_f0)
    if "mcd" in saved.files and saved["mcd"].shape[1] != n_mcd:
        same = False
    if not same:
        raise SystemExit(f"--resume: {path} was made with other sentences/speakers/model/--f0_utts/--mcd_utts; "
                         f"use the same options or a new --out_dir.")
    arrays = {k: saved[k].copy() for k in CACHE_KEYS if k in saved.files}
    if extends:  # keep the finished sentences, add empty slots for the new ones
        extra = len(utt_ids) - len(saved_ids)
        arrays["real"] = np.concatenate([arrays["real"], np.zeros((extra,) + arrays["real"].shape[1:],
                                                                  arrays["real"].dtype)])
        arrays["synth"] = np.concatenate([arrays["synth"], np.zeros(arrays["synth"].shape[:1] + (extra,)
                                                                    + arrays["synth"].shape[2:],
                                                                    arrays["synth"].dtype)], axis=1)
        arrays["done"] = np.concatenate([arrays["done"], np.zeros(extra, bool)])
        print(f"[info] extending {len(saved_ids)} finished sentences to {len(utt_ids)}")
    if "mcd" not in arrays:  # made before MCD was added: those sentences have none
        arrays["mcd"] = np.full((len(speakers), n_mcd), np.nan)
        if n_mcd and arrays["done"][:n_mcd].any():
            print("[info] this run has no MCD for sentences it already finished; for the MCD cross-check, "
                  "run again with a new --out_dir")
    return arrays


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr

    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    keep = ~(np.isnan(a) | np.isnan(b))
    if keep.sum() < 3:
        return float("nan")
    return float(spearmanr(a[keep], b[keep]).statistic)


def plot_ranking(table: pd.DataFrame, summary: Dict[str, object], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    low_c, high_c, light, muted, ink, ink2 = "#2a78d6", "#eb6834", "#c9c8c3", "#898781", "#0b0b0b", "#52514e"
    threshold = summary["pitch_groups"]["threshold_hz"]
    f0 = table["f0_median_hz"].to_numpy()
    trained = table["eligible"].to_numpy()
    groups = [("low pitch (< %.0f Hz)" % threshold, low_c, trained & (f0 < threshold)),
              ("high pitch (>= %.0f Hz)" % threshold, high_c, trained & (f0 >= threshold)),
              ("pitch not measured", muted, trained & np.isnan(f0)),
              ("never-trained embedding", light, ~trained)]
    has_f0 = bool(np.isfinite(f0).any())
    fig, axes = plt.subplots(1, 2 if has_f0 else 1, figsize=(13 if has_f0 else 8, 4.6), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    x = np.arange(1, len(table) + 1)
    y = table["similarity"].to_numpy()

    ax = axes[0, 0]
    yerr = np.clip([y - table["ci_low"].to_numpy(), table["ci_high"].to_numpy() - y], 0, None)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=light, elinewidth=1.0, zorder=2)
    for label, color, mask in groups:
        if mask.any():
            ax.scatter(x[mask], y[mask], s=18, color=color, zorder=3, label=label)
    for label, info in summary["conditions"].items():
        row = table.index[table["speaker_index"] == info["speaker_index"]][0]
        ax.annotate(f"{label}: {info['speaker_index']} ({info.get('vctk_id') or 'n/a'})", (x[row], y[row]),
                    xytext=(6, 8 if label != "furthest" else -14), textcoords="offset points",
                    fontsize=8, color=ink)
    base = summary["baselines"]
    ax.text(0.01, 0.02, f"reference levels: LJ vs LJ {base['real_vs_real']:.2f}, "
                        f"two VCTK voices {base['different_synthetic_speakers']:.2f}",
            transform=ax.transAxes, fontsize=8, color=ink2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("speaker, sorted by similarity (rank)", color=ink2)
    ax.set_ylabel("mean paired cosine to LJSpeech (ECAPA)", color=ink2)
    ax.set_title(f"Similarity of each VCTK voice to LJSpeech\n"
                 f"{summary['n_utts']} sentences per speaker, 95% bootstrap CI", fontsize=10, color=ink)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    if has_f0:
        ax2 = axes[0, 1]
        for label, color, mask in groups:
            if mask.any():
                ax2.scatter(f0[mask], y[mask], s=18, color=color, zorder=3, label=label)
        lj_f0 = summary["pitch"]["lj_f0_median_hz"]
        if lj_f0 == lj_f0:  # not NaN
            ax2.axvline(lj_f0, color=ink2, lw=1, ls="--")
            ax2.annotate(f"LJSpeech {lj_f0:.0f} Hz", (lj_f0, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(4, -12), textcoords="offset points", fontsize=8, color=ink2)
        ax2.axvline(threshold, color=light, lw=1, ls=":")
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
    # one fixed shuffle: the first N sentences are the same for any --n_utts >= N, so a run can be extended
    n_take = min(args.n_utts, len(manifest)) if args.n_utts else len(manifest)
    manifest = manifest.sample(n=n_take, random_state=args.seed).reset_index(drop=True)
    utt_ids = manifest["utt_id"].astype(str).tolist()
    n_utts, n_spk = len(utt_ids), len(speakers)
    n_f0 = min(max(args.f0_utts, 0), n_utts)
    n_mcd = min(max(args.mcd_utts, 0), n_utts)
    print(f"[info] {model_name}: {n_spk} speakers x {n_utts} sentences = {n_spk * n_utts} syntheses on {device}")

    encoder = SpeakerEncoder(device)
    cache = out_dir / "embeddings.npz"
    arrays = None
    if args.resume and cache.is_file():
        arrays = _load_cache(cache, utt_ids, speakers, model_name, n_f0, n_mcd)
        print(f"[info] resuming: {int(arrays['done'].sum())}/{n_utts} sentences already done")

    def save():
        _save_cache(cache, utt_ids=np.array(utt_ids), speakers=np.array(speakers),
                    model=np.array(model_name), **arrays)

    start, n_new = time.perf_counter(), 0
    for i, row in manifest.iterrows():
        if arrays is not None and arrays["done"][i]:
            continue
        audio, sr = load_audio(row.wav_path)
        real_prep = prepare(audio, sr)
        real16 = resample(real_prep, sr, ASR_SR)
        real_emb = encoder.embed(real16)
        if arrays is None:
            dim = real_emb.shape[-1]
            arrays = {"real": np.zeros((n_utts, dim), np.float32),
                      "synth": np.zeros((n_spk, n_utts, dim), np.float32),
                      "done": np.zeros(n_utts, bool),
                      "f0_real": np.full(n_f0, np.nan), "f0_synth": np.full((n_spk, n_f0), np.nan),
                      "mcd": np.full((n_spk, n_mcd), np.nan)}
        arrays["real"][i] = real_emb
        if i < n_f0:
            arrays["f0_real"][i] = f0_median(real16, ASR_SR)
        seed = seed_for(row.utt_id, args.seed)  # same noise draw for every speaker of this sentence
        for j, speaker in enumerate(speakers):
            synth.speaker_id = speaker
            wav, _ = synth.synthesize(row.ref_text, seed=seed)
            wav_prep = prepare(wav, synth.sample_rate)
            wav16 = resample(wav_prep, synth.sample_rate, ASR_SR)
            arrays["synth"][j, i] = encoder.embed(wav16)
            if i < n_f0:
                arrays["f0_synth"][j, i] = f0_median(wav16, ASR_SR)
            if i < n_mcd:  # model-free cross-check: spectral distance to the real recording
                arrays["mcd"][j, i] = mcd_dtw(real_prep, resample(wav_prep, synth.sample_rate, sr), sr)["mcd_dtw"]
        arrays["done"][i] = True
        n_new += 1
        done = int(arrays["done"].sum())
        elapsed = time.perf_counter() - start
        print(f"[{done}/{n_utts}] {row.utt_id}  {elapsed / 60:.1f} min, "
              f"about {(n_utts - done) * elapsed / n_new / 60:.1f} min left", flush=True)
        if n_new % args.save_every == 0:
            save()
    save()

    # ---------------------------------------------------------------- ranking
    paired = paired_similarity(arrays["synth"], arrays["real"])
    eligible = np.ones(n_spk, bool)
    vctk = load_vctk_speakers() if synth.n_speakers == VCTK_N_SPEAKERS else None
    if vctk is not None and not args.include_untrained:
        eligible = np.array([s in vctk.index for s in speakers])
    table = rank_speakers(paired, speakers, eligible, n_boot=args.n_boot)
    position = {s: j for j, s in enumerate(speakers)}
    rows = [position[s] for s in table["speaker_index"]]
    elig = table["eligible"]

    def rank_within_eligible(values: pd.Series, ascending: bool) -> pd.Series:
        ranks = pd.Series(pd.NA, index=table.index, dtype="Int64")
        ok = elig & values.notna()
        ranks[ok] = values[ok].rank(ascending=ascending, method="first").astype(int).to_numpy()
        return ranks

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows when a measure was skipped
        table["centroid_similarity"] = centroid_similarity(arrays["synth"], arrays["real"])[rows]
        table["centroid_rank"] = rank_within_eligible(table["centroid_similarity"], ascending=False)
        table["mcd_db"] = np.nanmean(arrays["mcd"], axis=1)[rows] if n_mcd else np.nan
        table["mcd_rank"] = rank_within_eligible(table["mcd_db"], ascending=True)
        table["f0_median_hz"] = np.nanmedian(arrays["f0_synth"], axis=1)[rows] if n_f0 else np.nan
    speaker_table = synth.speaker_table()
    table["embedding_norm"] = np.linalg.norm(speaker_table[table["speaker_index"]], axis=1)
    if vctk is not None:
        table["vctk_id"] = [vctk["vctk_id"].get(s, "") for s in table["speaker_index"]]
        table["n_vctk_train_utts"] = [int(vctk["n_train_utts"].get(s, 0)) for s in table["speaker_index"]]
    table.to_csv(out_dir / "speaker_ranking.csv", index=False)

    # ---------------------------------------------------------------- summary
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lj_f0 = float(np.nanmedian(arrays["f0_real"])) if n_f0 else float("nan")
    ranked = table[elig]
    conditions = {}
    for label, index in pick_conditions(table).items():
        row = table[table["speaker_index"] == index].iloc[0]
        conditions[label] = {"speaker_index": int(index), "vctk_id": row.get("vctk_id", None),
                             "similarity": float(row["similarity"]),
                             "ci": [float(row["ci_low"]), float(row["ci_high"])],
                             "rank_ci": [float(row["rank_ci_low"]), float(row["rank_ci_high"])],
                             "p_closest": float(row["p_closest"]), "p_furthest": float(row["p_furthest"]),
                             "f0_median_hz": float(row["f0_median_hz"]), "mcd_db": float(row["mcd_db"])}
    summary = {
        "model": model_name, "n_speakers_in_model": synth.n_speakers, "n_speakers_ranked": int(elig.sum()),
        "excluded_untrained": [int(s) for s in table.loc[~elig, "speaker_index"]],
        "n_utts": n_utts, "manifest": str(args.manifest), "conditions": conditions,
        "pitch_groups": pitch_groups(table, args.f0_threshold),
        "baselines": similarity_baselines(arrays["synth"], arrays["real"], eligible),
        "reliability": split_half_reliability(paired, eligible),
        "agreement": {"ecapa_paired_vs_centroid_spearman": _spearman(ranked["similarity"],
                                                                     ranked["centroid_similarity"]),
                      "ecapa_vs_mcd_spearman": _spearman(ranked["similarity"], -ranked["mcd_db"]),
                      "mcd_n_utts": n_mcd},
        "pitch": {"lj_f0_median_hz": lj_f0,
                  "spearman_similarity_vs_pitch_closeness": _spearman(ranked["similarity"],
                                                                      -(ranked["f0_median_hz"] - lj_f0).abs()),
                  "n_utts": n_f0},
        "embedding_norm": {"trained_median": float(ranked["embedding_norm"].median()),
                           "untrained": [float(v) for v in table.loc[~elig, "embedding_norm"]]},
        "settings": {"seed": args.seed, "n_boot": args.n_boot, "device": device,
                     "noise_scale": synth.noise_scale, "noise_scale_w": synth.noise_scale_w,
                     "length_scale": synth.length_scale, "frontend": synth.frontend,
                     "sample_rate": synth.sample_rate, "f0_threshold_hz": args.f0_threshold},
        "runtime_min": round((time.perf_counter() - start) / 60, 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    plot_ranking(table, summary, out_dir / "speaker_ranking.png")

    # the first sentence by each pick (overall and per pitch group) + the real recording, for listening
    wav_dir = out_dir / "wavs"
    first = manifest.iloc[0]
    audio, sr = load_audio(first.wav_path)
    save_wav(wav_dir / f"real_{first.utt_id}.wav", audio, sr)
    picks = {label: info["speaker_index"] for label, info in conditions.items()}
    for group in ("low", "high"):
        info = summary["pitch_groups"].get(group, {})
        if info.get("n"):
            picks[f"{group}pitch_closest"] = info["closest"]
            picks[f"{group}pitch_furthest"] = info["furthest"]
    for label, index in picks.items():
        synth.speaker_id = index
        wav, _ = synth.synthesize(first.ref_text, seed=seed_for(first.utt_id, args.seed))
        save_wav(wav_dir / f"{label}_spk{index}_{first.utt_id}.wav", wav, synth.sample_rate)

    shown = ["rank", "speaker_index", "vctk_id", "similarity", "ci_low", "ci_high", "p_closest", "p_furthest",
             "mcd_db", "f0_median_hz"]
    shown = [c for c in shown if c in table.columns]
    print("\nClosest 5:\n" + ranked[shown].head(5).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nFurthest 5:\n" + ranked[shown].tail(5).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    rel, agree, base = summary["reliability"], summary["agreement"], summary["baselines"]
    print(f"\nSplit-half reliability: Spearman {rel['split_half_spearman']:.3f} "
          f"(Spearman-Brown {rel['spearman_brown']:.3f})")
    print(f"Agreement with the ECAPA ranking: centroid {agree['ecapa_paired_vs_centroid_spearman']:.3f}, "
          f"MCD {agree['ecapa_vs_mcd_spearman']:.3f} (MCD on {n_mcd} sentences)")
    print(f"Reference levels: LJ vs LJ {base['real_vs_real']:.3f} | same VCTK voice "
          f"{base['same_synthetic_speaker']:.3f} | two different VCTK voices "
          f"{base['different_synthetic_speakers']:.3f}")
    print(f"LJSpeech median pitch: {lj_f0:.0f} Hz; model sample rate: {synth.sample_rate} Hz")
    if summary["excluded_untrained"]:
        print(f"Excluded (embedding never trained): {summary['excluded_untrained']}")
    for label, info in conditions.items():
        print(f"{label:8s} speaker {info['speaker_index']:3d} ({info['vctk_id'] or 'n/a'}) "
              f"cos {info['similarity']:.3f} [{info['ci'][0]:.3f}, {info['ci'][1]:.3f}]  "
              f"pitch {info['f0_median_hz']:.0f} Hz")
    for group in ("low", "high"):
        info = summary["pitch_groups"].get(group, {})
        if info.get("n"):
            print(f"{group}-pitch voices: n={info['n']}, mean cos {info['mean_similarity']:.3f}, "
                  f"closest {info['closest']} ({info['closest_similarity']:.3f}), "
                  f"furthest {info['furthest']} ({info['furthest_similarity']:.3f})")
    print(f"[info] wrote {out_dir}/speaker_ranking.csv, summary.json, speaker_ranking.png, wavs/")
    return {"table": table, "summary": summary}


if __name__ == "__main__":
    main()
