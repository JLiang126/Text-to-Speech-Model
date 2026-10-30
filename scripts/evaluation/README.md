# Evaluation

Scores every system on the same held-out test sentences, with the same preprocessing,
seeds and hardware, so the only thing that differs between experiments is the
controlled change. Owner: Adi.

## What's here

| File | What it does |
|---|---|
| `scripts/synth.py` | Single "text -> .wav" entry point. `--prompts` writes the 12 fixed prompts to `wav_examples/<exp>_NN.wav` |
| `scripts/evaluate.py` | Scores one system on the test split. Writes per-sentence metrics, a summary with 95% CIs, and latency/RTF |
| `scripts/compare.py` | Combines several systems into the report table, paired significance tests and figures |
| `scripts/rank_speakers.py` | Ranks the VCTK speakers by how close they sound to LJSpeech, with bootstrap uncertainty: picks the closest / median / furthest starting speaker for the speaker-similarity experiment |
| `scripts/evaluation/` | The building blocks: `audio`, `manifest`, `metrics`, `timing`, `stats`, `synthesizer`, `text_normalise`, `speakers`, `prompts`, plus `vctk_speakers.csv` (speaker index -> VCTK ID) |
| `tests/` | `python -m pytest tests -q`. Runs offline. The official-format tests need `VITS_DIR` to point at the VITS code |

## Two checkpoint formats

Every script takes either:
- **`--model`**: Hugging Face format (`transformers.VitsModel`), what `finetune-hf-vits-main/` trains. Give a hub id (`ylacombe/vits-vctk-with-discriminator`, the pretrained start), a finished run's `output_dir`, or one of its `checkpoint-<step>` folders (weight norm is folded back automatically, config and tokenizer come from the run folder). Text goes through the checkpoint's own `VitsTokenizer`, as in fine-tuning.
- **`--config` + `--checkpoint`**: official VITS format (`G_<step>.pth`), text through `scripts/TextProcessor.py`.

## Run order (from the repo root)

```bash
pip install -r requirements.txt -r requirements-eval.txt

# 0. Rank the VCTK speakers against LJSpeech (quick CPU check; GPU: --n_utts 0 --device cuda)
python scripts/rank_speakers.py --n_utts 5

# 1. Ceiling: the real recordings
python scripts/evaluate.py --exp ground_truth --ground_truth

# 2. Floor: the pretrained model before any fine-tuning, with the experiment's speaker index
python scripts/evaluate.py --exp pretrained --model ylacombe/vits-vctk-with-discriminator --speaker_id <ID>

# 3. Each experiment, at the final step (and at earlier steps too, for learning curves)
python scripts/evaluate.py --exp expA --model runs/expA --speaker_id <ID>
python scripts/evaluate.py --exp expA --model runs/expA/checkpoint-2000 --speaker_id <ID> \
    --out_dir results/expA_step2000

# 4. CPU timing for the same checkpoints (adds latency_cpu.json, nothing else changes)
python scripts/evaluate.py --exp expA ... --out_dir results/expA --metrics rtf --device cpu

# 5. Report table, paired tests and figures
python scripts/compare.py results/ground_truth results/pretrained results/expA results/expB \
    --pair expA expB --out results/comparison

# 6. The >=10 qualitative wavs for the submission
python scripts/synth.py --prompts --exp expA --model runs/expA --speaker_id <ID> --out_dir wav_examples
```

Official-format equivalents: replace `--model X` with `--config <json> --checkpoint <G_step.pth>` and set
`VITS_DIR` (PowerShell: `$env:VITS_DIR="vits-main"`).

Notes:
- **Smoke test** with no downloads: `--limit 5 --metrics mcd,rtf`.
- **`--speaker_id`** is only for multi-speaker checkpoints. Use the same index the LJSpeech clips were fine-tuned with, for the pretrained floor too.
- **Learning curves:** pass the same `--exp` name with a different `--out_dir` per step. `compare.py` plots metric vs step automatically.
- **Windows:** the trailing `\` line continuations are bash-only. Put each command on one line.

## Metrics

Every metric is computed per sentence. Each is reported as a mean with a 95% bootstrap
confidence interval (10,000 resamples over sentences).

| Metric | What it measures | Better | How it is computed |
|---|---|---|---|
| MCD-DTW (dB) | Spectral distance from the real recording of the same sentence | lower | 24 mel-cepstral coefficients. c0 is dropped, so loudness doesn't count. Frames are aligned with dynamic time warping (Kubichek 1993) |
| Duration ratio | Speaking rate compared with the real recording | close to 1 | synthesised frames / reference frames |
| WER / CER | Intelligibility: does a speech recogniser hear the right words? | lower | Whisper `small.en`, greedy decoding. Whisper's English normaliser is applied to both sides, so case, punctuation and number formats don't count as errors. Corpus rate = total errors / total reference words |
| Speaker similarity | Does it sound like the LJSpeech speaker? | higher | Cosine similarity of ECAPA-TDNN embeddings (SpeechBrain) to the average embedding of 50 real *training* clips. `secs_paired` compares with the real recording of the same sentence instead |
| UTMOS | Predicted naturalness on a 1-5 MOS scale | higher | UTMOS22 strong learner (Saeki et al. 2022) |
| Latency / RTF | Speed | lower | See the protocol below. The table reports aggregate RTF = total synthesis time / total audio time |

How to read them:
- **Reference points:** every number sits between two of them.
  - **ground_truth** (the real recordings) is the best achievable. Its WER is the speech recogniser's own error rate.
  - **pretrained** (no fine-tuning) shows where we started.

  So "expA closes 80% of the speaker-similarity gap" is a meaningful statement.
- **MCD:** computed the same way for every system, so it is valid for comparing our experiments. It uses mel-filterbank cepstra, scaled to the standard cepstrum convention. Its absolute values differ somewhat from papers that use SPTK mel-cepstra.
- **Learned metrics:** Whisper, ECAPA and UTMOS are trained models, not people. Treat them as proxies and back them up by listening to `wav_examples/`.

## Latency / RTF protocol

1. **Sentences:** the 12 fixed prompts, at batch size 1, with a fixed seed so every repeat produces the same output length.
2. **Warm-up:** 3 untimed calls first. They load espeak, CUDA kernels and caches.
3. **Repeats:** 5 timed repeats per sentence. On GPU, `torch.cuda.synchronize()` runs before starting and after stopping the clock.
4. **Latency** = text in to waveform out, including the espeak front-end. The front-end's share is reported as `frontend_share`.
5. **RTF** = latency / audio duration. RTF < 1 means faster than real time.
6. **Hardware:** the device, CPU/GPU name, thread count and library versions are saved with every result. GPU and CPU results go to separate files (`latency_cuda.json`, `latency_cpu.json`).

**Time every experiment on the same machine and the same GPU.** Otherwise the hardware becomes a second variable. `compare.py` warns if a timing column mixes hardware.

## Fairness controls

- **Same test sentences:** the same test manifest for every system. It is never used for training or for picking checkpoints.
- **Same seeds:** each sentence gets its own seed (CRC32 of `seed:utt_id`), identical across systems. VITS samples noise at inference (`noise_scale` 0.667, `noise_scale_w` 0.8). Without fixed seeds, part of any difference would be sampling luck.
- **Same preprocessing for all audio:** DC removal, peak normalisation to 0.95 and silence trimming at -40 dB, mirroring `scripts/AudioProcessor.py`.
- **Same text front-end as training:** the checkpoint's own `VitsTokenizer` (Hugging Face format) or `scripts/TextProcessor.py` (official format).

## Speaker ranking (`rank_speakers.py`)

The VCTK-pretrained VITS has one learned embedding per speaker index, and fine-tuning on
LJSpeech continues from whichever index the clips are given. The speaker-similarity
experiment changes only that index, so the choice of indices needs to be defensible.

1. `--n_utts` LJSpeech **training** sentences, sampled with a fixed seed (the test split stays unseen).
2. Every speaker index says every sentence, with that sentence's seed (same noise draw for every speaker).
3. Real and synthetic audio get the evaluation preprocessing and an ECAPA-TDNN embedding.
4. **Score** = mean over sentences of cos(speaker s saying sentence i, the real recording of sentence i). Same words on both sides, so the score is about the voice. `centroid_similarity` (mean voice vs mean LJSpeech voice) is reported as a cross-check.
5. **Uncertainty:** 1,000 bootstrap resamples of the sentences (all speakers resampled together) give a 95% CI per score, a 95% CI per rank, and `p_closest` / `p_furthest`, the share of resamples in which a speaker comes out closest / furthest.
6. **Is `--n_utts` enough?** `split_half_spearman` correlates the ranking from odd vs even sentences (Spearman-Brown corrects it to the full set). Near 1 means more sentences would not change the picks.

Things to know before using the picks:
- **Index 108 is excluded.** The official VCTK config allocates 109 embeddings, but the official training filelist only uses indices 0-107 (`vctk_speakers.csv`, 108 speakers). Index 108 was never trained, so it would rank as an outlier for the wrong reason. `--include_untrained` keeps it.
- **Pitch/gender confound.** Pitch (largely gender) is the biggest difference between voices, so "closest vs furthest" can end up meaning "low- vs high-pitched", in either direction. In the first 5-sentence CPU run, the ECAPA-closest voices were low-pitched (about 115 Hz) and the furthest high-pitched (about 200 Hz), the opposite of the naive expectation for a female target. `summary.json` -> `pitch_groups` gives the closest and furthest voice *within* each pitch group (split at `--f0_threshold`, 160 Hz), which allows a design that separates "how similar" from "which pitch group".
- **Reference levels** put the scores on a scale (`summary.json` -> `baselines`): LJSpeech vs LJSpeech on different sentences (the same-speaker ceiling), one VCTK voice vs itself, and two different VCTK voices on the same sentence.
- **Second opinion:** `mcd_db` is the MCD-DTW of each voice against the real recording of the same sentence (first `--mcd_utts` sentences), a measure with no learned model. `agreement.ecapa_vs_mcd_spearman` near 1 means both measures rank the voices the same way; near 0 means the ranking depends on the measure, and only the listening check can arbitrate.
- **The index must reach the model during fine-tuning.** In `finetune-hf-vits-main/run_vits_finetuning.py`, the speaker index is only passed when the dataset has a column literally named `speaker_id` (the collator checks for that name). Without it `speaker_id=None`, the model trains with no speaker conditioning at all, and every "condition" is identical. Add a constant `speaker_id` column set to the chosen index, set `speaker_id_column_name: "speaker_id"`, and leave `filter_on_speaker_id` unset. Evaluate with the same `--speaker_id`.
- **Speaker IDs** (`p225`...) assume the checkpoint keeps the official VITS VCTK speaker order. That holds for conversions of the official release; `ylacombe/vits-vctk-with-discriminator` has no model card to confirm it, so treat the IDs as labels and rely on the indices.

## Statistics in `paired_tests.csv`

For each `--pair A B` and each metric (B minus A, on the sentences both systems share):
- **`mean_diff` and its 95% bootstrap CI.**
- **`wilcoxon_p`:** Wilcoxon signed-rank test, which makes no normality assumption.
- **`wilcoxon_p_holm`:** the p-values adjusted with Holm-Bonferroni across the metrics of that pair, because testing 7 metrics at once inflates false positives.
- **`cohens_dz`:** effect size, the mean difference divided by the SD of the differences. Roughly 0.2 is small, 0.5 medium and 0.8 large.
- **`verdict`:** "B better" or "A better" only when the CI excludes 0 *and* the Holm p < 0.05. Otherwise "no clear difference". `dur_dev` = |duration ratio - 1|.

These cover sentence-to-sentence variation, not training-run variation. That needs a second training seed. Plan the pairs (`--pair`) before looking at results.

## Text front-end facts (found while building this)

- **Symbol table:** `TextProcessor` had an extra `"` in its table (179 symbols vs the checkpoint's 178), which shifted every token ID by one. This is fixed. With the fix, `TextProcessor` produces *exactly* the official VITS token IDs on ordinary sentences, and a test proves it (`test_repo_frontend_gives_the_same_token_ids_as_official_vits`).
- **Numbers:** `clean_normalised_text` deletes digits. LJSpeech training text has no digits, so training is unaffected, but arbitrary input would lose its numbers. `synth.py` and `evaluate.py` therefore spell out numbers, money, `%` and `&` first (`text_normalise.py`, rules from keithito/tacotron). Example: "On March 3rd, 1865, ... 12,500 dollars to 47 workers" becomes "On March third, eighteen sixty-five, ... twelve thousand, five hundred dollars to forty-seven workers".
- **Abbreviations:** these are the one remaining difference from the official front-end. Official VITS turns "Mr." into "mister" (no full stop), while `TextProcessor` keeps the full stop, adding a "." token mid-sentence. That is a candidate for the error analysis (an unnatural pause after titles).

## Outputs

`results/<system>/`:
- `per_utterance.csv`: every metric and the ASR transcript, one row per sentence.
- `summary.json`: means, CIs, corpus WER/CER, device, settings and versions.
- `latency_<cpu|cuda>.json` and `rtf_runs_<cpu|cuda>.csv`.
- `wavs/`: the synthesised test sentences. These are git-ignored, because they are regenerated from checkpoints.

`results/comparison/`:
- `comparison_table.csv` and `comparison_table.md` (paste the `.md` into the report).
- `paired_tests.csv`.
- `comparison.png`.
- `learning_curves.png`, when an experiment was scored at two or more steps.

`results/speaker_ranking/` (from `rank_speakers.py`):
- `speaker_ranking.csv`: one row per speaker index, closest first.
- `summary.json`: the closest / median / furthest picks, reliability, pitch relation, settings.
- `speaker_ranking.png`: scores with CIs, and score against pitch.
- `embeddings.npz`: every embedding; `--resume` continues a stopped run from it.
- `wavs/`: one training sentence by each pick, plus the real recording.

## Notebook cells (section 1.4)

```python
import pandas as pd, json
from IPython.display import Image, Audio, display
display(pd.read_csv("results/comparison/comparison_table.csv"))
display(pd.read_csv("results/comparison/paired_tests.csv")[["a", "b", "metric", "mean_diff", "ci_low", "ci_high", "wilcoxon_p_holm", "cohens_dz", "verdict"]])
display(Image("results/comparison/comparison.png"))
for exp in ("expA", "expB"):
    for p in json.load(open(f"wav_examples/{exp}_prompts.json"))["prompts"][:3]:
        print(exp, p["category"], "-", p["text"]); display(Audio(f"wav_examples/{p['file']}"))
```

## Known limitations

- **Untested on real audio:** the learned metrics download their weights on first use (Whisper `small.en` alone is about 460 MB). If a download fails, that metric is recorded as NaN and listed under `unavailable_metrics` in `summary.json`; the rest of the run continues.
- **CI width:** with fewer than about 20 test sentences, the bootstrap CIs are too narrow. Use the full 150.

## Sources

- **VITS:** Kim, Kong & Son (2021), https://github.com/jaywalnut310/vits (MIT).
- **Hugging Face VITS and fine-tuning code:** `transformers` `VitsModel`; Y. Lacombe, https://github.com/ylacombe/finetune-hf-vits (MIT).
- **VCTK corpus:** Yamagishi, Veaux & MacDonald (2019), CSTR VCTK Corpus v0.92, University of Edinburgh.
- **Bootstrap:** Efron & Tibshirani (1993), *An Introduction to the Bootstrap*.
- **Split-half reliability:** Spearman (1910) and Brown (1910), British Journal of Psychology.
- **YIN pitch:** de Cheveigné & Kawahara (2002), JASA 111(4).
- **Whisper:** Radford et al. (2023), https://github.com/openai/whisper (MIT).
- **ECAPA-TDNN:** Desplanques, Thienpondt & Demuynck (2020), https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb (Apache-2.0).
- **UTMOS:** Saeki et al. (2022), via https://github.com/tarepan/SpeechMOS (MIT).
- **MCD:** Kubichek (1993), 'Mel-cepstral distance measure for objective speech quality assessment', IEEE PACRIM.
- **Speaker similarity as SECS:** Casanova et al. (2022), YourTTS.
- **Number expansion:** keithito/tacotron `text/numbers.py` (MIT).
- **Harvard sentence** in the prompts: IEEE (1969) recommended practice for speech quality measurements.
- **Holm (1979):** 'A simple sequentially rejective multiple test procedure', Scandinavian Journal of Statistics.
