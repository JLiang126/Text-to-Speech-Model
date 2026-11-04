# Evaluation

Scores every system on the same held-out test sentences, with the same preprocessing,
seeds and hardware, so the only thing that differs between experiments is the
controlled change. Owner: Adi.

## What's here

| File | What it does |
|---|---|
| `scripts/synth.py` | Single "text -> .wav" entry point. `--prompts` writes the 12 fixed prompts to `wav_examples/<exp>_NN.wav` |
| `scripts/evaluate.py` | Scores one system on the test split. Writes per-sentence metrics, a summary with 95% CIs, and latency/RTF |
| `scripts/compare.py` | Combines several systems into a results table, paired significance tests and a figure |
| `scripts/evaluate_experiments.py` | Evaluates every experiment of a round in one resumable run and writes the report tables, figures and checks |
| `scripts/report_figures.py` | The report's two summary figures: every round-2 run before and after fine-tuning, and the losses by epoch |
| `scripts/pretrained_validation_losses.py` | Epoch 0 of the loss curves: the pretrained model's losses on the round-2 validation split, through `train_model.py`'s own validation pass |
| `scripts/data_checks.py` | Split and loader checks: metadata coverage, unique IDs, disjoint splits, every manifest file present and in spec, proportions, and every item of a batch in range. Also counts how many evaluated sentences were in a training split |
| `scripts/compare_speaker_embeddings.py` | How far fine-tuning moved each run's speaker vector: relative L2 and cosine between the pretrained and fine-tuned vector, against the weight-decay floor and the distance between two different voices. The same numbers appear in `report/checks.csv` |
| `scripts/rank_speakers.py` | Ranks the VCTK speakers by how close they sound to LJSpeech, with bootstrap uncertainty; it picked the starting speakers 7 (closest) and 24 (furthest) |
| `scripts/evaluation/` | The building blocks: `audio`, `manifest`, `metrics`, `timing`, `stats`, `synthesizer`, `text_normalise`, `speakers`, `prompts`, `checkpoints` (weight checks), `training_logs` (loss histories of the runs), plus `vctk_speakers.csv` (speaker index -> VCTK ID) |
| `tests/` | `python -m pytest tests -q`. Runs offline, on tiny random models |

## Evaluating every experiment in one go (`scripts/evaluate_experiments.py`)

0. **Score on held-out sentences.** `python scripts/data_checks.py --config configs/no_droplayer.yaml` rebuilds the split of notebook section 1.1 (seed 42, 90/5/5) into `dataset/splits/` and checks it. The first 150 of its 646 test sentences are scored for every system. Round 2 split the same seeded shuffle 80/10/10, so they are test sentences in both rounds. `evaluate.py` refuses to score a model on sentences from the run's own training or validation manifest (`splits/` in the run folder, else the manifests beside `--manifest`); `--allow_overlap` overrides this.
1. **Inputs.** Each round has a table of runs: `experiments/experiments.csv` (round 1) and `experiments/experiments_v2.csv` (round 2). The `checkpoint` column says where each weights file goes (git ignores them). The run folders with the training logs are `outputs/` (round 1) and `outputs_v2/` (round 2).
2. **Run**, from the repo root (CPU budget: 150 sentences; drop `--limit` on a GPU):
   ```bash
   python scripts/evaluate_experiments.py --limit 150
   python scripts/evaluate_experiments.py --experiments experiments/experiments_v2.csv --plan experiments/report_plan_v2.json --results_dir results/experiments_v2 --runs_dir outputs_v2 --limit 150 --skip_timing
   ```
   Each run:
   - checks every checkpoint against the base model (`report/checks.csv`): the configured speaker row must be the one that trained, and frozen modules must be unchanged. It also compares each run's `training_config.json` with its settings, and counts the scored test sentences that were in the run's training split (must be 0);
   - scores each experiment, the real recordings (ceiling) and the pretrained model with each run's speaker index (floors `pretrained_spk<k>`), in the priority order of the report plan (`experiments/report_plan.json`, `experiments/report_plan_v2.json`);
   - times one system (all runs share the architecture; round 1) and synthesises the 12 prompts per system;
   - writes `report/report.md` with the experiment table, the checks, an overview, one section per question (table, figure, paired tests), a before-vs-after figure, and each section's validation losses from the training logs.
3. **Resume:** a system counts as finished only if it was scored on exactly the same sentences (`summary.json` stores a fingerprint of the test IDs), so results from another split are re-scored, never reused. `--report_only` rebuilds the report from what has finished.
4. **New checkpoints:** add a row to the round's table (speaker_id, settings and a `description` for the report) and the name to a section of its report plan.
5. **Report figures:**
   ```bash
   python scripts/pretrained_validation_losses.py --speaker_ids 7 24   # once: epoch 0, about 10-20 min on a CPU
   python scripts/report_figures.py                                    # results/experiments_v2/report/figures/
   ```

## Checkpoints

Every script takes `--model`, a Hugging Face-format VITS (`transformers.VitsModel`), the format `finetune-hf-vits` and `scripts/train_model.py` save: a hub id (`ylacombe/vits-vctk-with-discriminator`, the pretrained start), a run's output folder, one of its `checkpoint-<step>` folders (weight norm is folded back automatically), or a bare weights file such as `model.worst_speaker_embedding.safetensors`. A bare weights file takes its config and tokenizer from `--base_model` (default `ylacombe/vits-vctk-with-discriminator`), which fine-tuning does not change. Text goes through the training front-end (see Fairness controls).

## Single-system commands (from the repo root)

```bash
pip install -r requirements.txt -r requirements-eval.txt

# Rank the VCTK speakers against LJSpeech (quick CPU check; GPU: --n_utts 0 --device cuda)
python scripts/rank_speakers.py --n_utts 5

# Ceiling: the real recordings
python scripts/evaluate.py --exp ground_truth --ground_truth

# Floor: the pretrained model before any fine-tuning, with a run's speaker index
python scripts/evaluate.py --exp pretrained_spk24 --model ylacombe/vits-vctk-with-discriminator --speaker_id 24

# One run, and CPU timing for it (adds latency_cpu.json, nothing else changes)
python scripts/evaluate.py --exp worst_speaker_embedding --model outputs_v2/worst_speaker_embedding/run/model.worst_speaker_embedding.safetensors --speaker_id 24
python scripts/evaluate.py --exp worst_speaker_embedding ... --metrics rtf --device cpu

# The 12 qualitative prompts for the submission
python scripts/synth.py --prompts --exp worst_speaker_embedding --model outputs_v2/worst_speaker_embedding/run/model.worst_speaker_embedding.safetensors --speaker_id 24 --out_dir wav_examples
```

Notes:
- **Smoke test** with no downloads: `--limit 5 --metrics mcd,rtf`.
- **`--speaker_id`** is only for multi-speaker checkpoints. Use the index the LJSpeech clips were fine-tuned with, for the pretrained floor too.

## Metrics

Every metric is computed per sentence and reported as a mean with a 95% bootstrap
confidence interval (10,000 resamples over sentences).

| Metric | What it measures | Better | How it is computed |
|---|---|---|---|
| MCD-DTW (dB) | Spectral distance from the real recording of the same sentence | lower | 24 mel-cepstral coefficients. c0 is dropped, so loudness doesn't count. Frames are aligned with dynamic time warping (Kubichek 1993) |
| Duration ratio | Speaking rate compared with the real recording | close to 1 | synthesised frames / reference frames |
| WER / CER | Intelligibility: does a speech recogniser hear the right words? | lower | Whisper `small.en`, greedy decoding. Whisper's English normaliser is applied to both sides, so case, punctuation and number formats don't count as errors. Corpus rate = total errors / total reference words |
| Speaker similarity | Does it sound like the LJSpeech speaker? | higher | Cosine similarity of ECAPA-TDNN embeddings (SpeechBrain) to the average embedding of 50 real *training* clips. `secs_paired` compares with the real recording of the same sentence instead |
| UTMOS | Predicted naturalness on a 1-5 MOS scale | higher | UTMOS22 strong learner (Saeki et al. 2022) |
| Median pitch (Hz) | Where the voice sits, compared with LJ (about 207 Hz) | close to LJ | YIN, median over the frames within 20 dB of the loudest; `f0_dev` is the absolute difference from the real recording of the same sentence |
| Latency / RTF | Speed | lower | See the protocol below. Aggregate RTF = total synthesis time / total audio time |

How to read them:
- **Reference points:** every number sits between two of them. The real recordings (`ground_truth`) are the best achievable; their WER is the speech recogniser's own error rate. The pretrained model (`pretrained_spk<k>`) shows where a run started. For example, full fine-tuning takes speaker similarity from 0.21 to 0.86, against 0.85 for the real recordings.
- **MCD:** computed the same way for every system, so it is valid for comparing our experiments. It uses mel-filterbank cepstra scaled to the standard cepstrum convention; absolute values differ somewhat from papers that use SPTK mel-cepstra.
- **Learned metrics:** Whisper, ECAPA and UTMOS are trained models, not people. Treat them as proxies and back them up by listening to `wav_examples/`.

## Latency / RTF protocol

1. **Sentences:** the 12 fixed prompts, at batch size 1, with a fixed seed so every repeat produces the same output length.
2. **Warm-up:** 3 untimed calls first. They load espeak, CUDA kernels and caches.
3. **Repeats:** 5 timed repeats per sentence. On GPU, `torch.cuda.synchronize()` runs before starting and after stopping the clock.
4. **Latency** = text in to waveform out, including the espeak front-end. The front-end's share is reported as `frontend_share`.
5. **RTF** = latency / audio duration. RTF < 1 means faster than real time.
6. **Hardware:** the device, CPU/GPU name, thread count and library versions are saved with every result. GPU and CPU results go to separate files (`latency_cuda.json`, `latency_cpu.json`).

Time every system on the same machine; otherwise the hardware becomes a second variable. `compare.py` warns if a timing column mixes hardware.

## Fairness controls

- **Same, held-out test sentences** for every system (see step 0 above). They are never used for training or for picking checkpoints.
- **Same seeds:** each sentence gets its own seed (CRC32 of `seed:utt_id`), identical across systems. VITS samples noise at inference (`noise_scale` 0.667, `noise_scale_w` 0.8). Without fixed seeds, part of any difference would be sampling luck.
- **Same preprocessing for all audio:** DC removal, peak normalisation to 0.95 and silence trimming at -40 dB, mirroring `scripts/AudioProcessor.py`.
- **Same text front-end as training:** `clean_normalised_text` and the fixed VCTK tokenizer (`load_vctk_tokenizer`) from `scripts/TextProcessor.py`, exactly as `DataLoader` prepares the training text.

## Speaker ranking (`rank_speakers.py`)

The VCTK-pretrained VITS has one learned embedding per speaker index, and fine-tuning on
LJSpeech continues from whichever index the clips are given. The speaker experiments change
only that index, so the choice of indices needs to be defensible.

1. `--n_utts` LJSpeech **training** sentences, sampled with a fixed seed (the test split stays unseen).
2. Every speaker index says every sentence, with that sentence's seed (same noise draw for every speaker).
3. Real and synthetic audio get the evaluation preprocessing and an ECAPA-TDNN embedding.
4. **Score** = mean over sentences of cos(speaker s saying sentence i, the real recording of sentence i). Same words on both sides, so the score is about the voice. `centroid_similarity` (mean voice vs mean LJSpeech voice) is reported as a cross-check.
5. **Uncertainty:** 1,000 bootstrap resamples of the sentences (all speakers resampled together) give a 95% CI per score, a 95% CI per rank, and `p_closest` / `p_furthest`, the share of resamples in which a speaker comes out closest / furthest.
6. **Is `--n_utts` enough?** `split_half_spearman` correlates the ranking from odd vs even sentences (Spearman-Brown corrects it to the full set). Near 1 means more sentences would not change the picks.

Things to know about the picks:
- **Index 108 is excluded.** The VCTK release allocates 109 embeddings, but its training filelist only uses indices 0-107 (`vctk_speakers.csv`, 108 speakers). Index 108 was never trained, so its voice is an arbitrary mix rather than a VCTK speaker. `--include_untrained` keeps it.
- **Pitch/gender confound.** Pitch (largely gender) is the biggest difference between voices. Here the ECAPA-closest voices were low-pitched (speaker 7, male, about 117 Hz) and the furthest high-pitched (speaker 24, female, about 199 Hz), the opposite of the naive expectation for a female target. `summary.json` -> `pitch_groups` gives the closest and furthest voice within each pitch group.
- **Reference levels** put the scores on a scale (`summary.json` -> `baselines`): LJSpeech vs LJSpeech on different sentences (the same-speaker ceiling), one VCTK voice vs itself, and two different VCTK voices on the same sentence.
- **Second opinion:** `mcd_db` is the MCD-DTW of each voice against the real recording of the same sentence, a measure with no learned model. `results/speaker_check/` repeats the ranking for a shortlist of 11 speakers, 7 and 24 among them, as a cross-check.
- **Speaker IDs** (`p225`...) assume the checkpoint keeps the VCTK release's speaker order. `ylacombe/vits-vctk-with-discriminator` has no model card to confirm it, so treat the IDs as labels and rely on the indices.

## Statistics in `paired_tests.csv`

For each `--pair A B` and each metric (B minus A, on the sentences both systems share):
- **`mean_diff` and its 95% bootstrap CI.**
- **`wilcoxon_p`:** Wilcoxon signed-rank test, which makes no normality assumption.
- **`wilcoxon_p_holm`:** the p-values adjusted with Holm-Bonferroni across the metrics of that pair, because testing several metrics at once inflates false positives.
- **`cohens_dz`:** effect size, the mean difference divided by the SD of the differences. Roughly 0.2 is small, 0.5 medium and 0.8 large.
- **`verdict`:** "B better" or "A better" only when the CI excludes 0 *and* the Holm p < 0.05. Otherwise "no clear difference". `dur_dev` = |duration ratio - 1|.

These cover sentence-to-sentence variation, not training-run variation, which would need repeated training seeds.

## Text front-end facts (found while building this)

- **Symbol table:** `TextProcessor` had an extra `"` in its table (179 symbols vs the checkpoint's 178), which shifted every token ID by one. This is fixed, and `tests/test_text_frontend.py` checks the table against the checkpoint's order.
- **Numbers:** `clean_normalised_text` deletes digits. LJSpeech training text has no digits, so training is unaffected, but arbitrary input would lose its numbers. `synth.py` and `evaluate.py` therefore spell out numbers, money, `%` and `&` first (`text_normalise.py`, rules from keithito/tacotron). Example: "On March 3rd, 1865, ... 12,500 dollars to 47 workers" becomes "On March third, eighteen sixty-five, ... twelve thousand, five hundred dollars to forty-seven workers".
- **Abbreviations:** the front-end keeps the full stop of "Dr.", "Mr." or "St.", adding a sentence-final "." token in mid-sentence; LJSpeech's normalised training text spells abbreviations out.
- **Homographs:** espeak-ng phonemises each word without its context, so past-tense "read" and the metal "lead" get the vowel of present-tense "read" and of "to lead".

## Outputs

`results/experiments/` (round 1) and `results/experiments_v2/` (round 2), one folder per system:
- `per_utterance.csv`: every metric and the ASR transcript, one row per sentence.
- `summary.json`: means, CIs, corpus WER/CER, the test-set fingerprint, device, settings and versions.
- `latency_<cpu|cuda>.json` and `rtf_runs_<cpu|cuda>.csv` (round 1's timed system).
- `wavs/` and `prompts/`: the synthesised test sentences and prompts. git ignores them, because they are regenerated from the checkpoints.

and `report/`: `report.md`, `checks.csv`, one folder per question (`comparison_table.md`, `paired_tests.csv`, `comparison.png`), the loss tables and figures, and for round 2 `figures/` (the report's summary figures).

`results/speaker_ranking/` (from `rank_speakers.py`): `speaker_ranking.csv` (one row per speaker index, closest first), `summary.json` (picks, reliability, pitch groups, settings) and `speaker_ranking.png`.

## Notebook cells

`55_A3.ipynb` runs these tools in place:
- **Setup (after the imports):** points the phonemiser at the espeak-ng library bundled with `espeakng-loader`, so no system install is needed.
- **1.1 (after the waveform and spectrogram cell):** the split and full-batch checks (`scripts/data_checks.py`).
- **1.2.4:** the text-to-wav demo with `scripts/synth.py` and the submitted run (`FINAL_SYSTEM`, also used in 1.4.2). It skips synthesis when the checkpoint is not downloaded.
- **1.4.1:** the evaluation. By default it rebuilds the reports from saved results; set `RUN_EVALUATION = True` to score. It shows the held-out check, the overview table, latency/RTF, one cell per question with its paired tests, figures and validation losses, and the report's two summary figures.
- **1.4.2:** the 12 prompts, with the submitted system's wavs copied to `wav_examples/` and side-by-side listening.

## Known limitations

- **Downloads:** the learned metrics download their weights on first use (Whisper `small.en` alone is about 460 MB). If a download fails, that metric is recorded as NaN and listed under `unavailable_metrics` in `summary.json`; the rest of the run continues, and the system counts as unfinished.
- **CI width:** with fewer than about 20 test sentences, the bootstrap CIs are too narrow. Use at least 150 of the 646 test sentences.
- **Validation losses** are teacher-forced reconstruction losses (posterior path), not free synthesis; only mel, KL and duration are comparable between runs, since each run has its own discriminator.
- **WER on bracketed text:** Whisper's normaliser deletes text in brackets from the reference but not from the transcript. Two of the 150 test sentences contain brackets, which adds about 12 spurious errors to every system, the real recordings included; comparisons are unaffected.

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
