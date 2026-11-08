# Evaluation

Scores every system on the same held-out test sentences, with the same preprocessing and seeds, so
the only difference between two experiments is the setting they change. Notebook section 1.4 shows
the results. Owner: Adi.

## Files

| File | What it does |
|---|---|
| `scripts/evaluate.py` | Scores one system: per-sentence metrics, a summary with 95% confidence intervals, and optionally the synthesis speed |
| `scripts/synth.py` | Text to .wav with any checkpoint; `--prompts` writes the 12 qualitative prompts |
| `scripts/rank_speakers.py` | Ranks the 108 trained VCTK voices by ECAPA similarity to LJ (`results/speaker_ranking/`) |
| `scripts/pretrained_validation_losses.py` | The pretrained model's validation losses for chosen voices, through `train_model.py`'s own validation pass: epoch 0 of the loss curves, and the mel-loss ranking of voices |
| `scripts/data_checks.py` | Checks the split and the data loader: coverage, unique IDs, disjoint splits, file specs and batch ranges |
| `scripts/evaluation/results.py` | The run table, and every table, figure and audio grid of notebook 1.4 |
| `scripts/evaluation/` | The building blocks: `audio`, `manifest`, `metrics`, `stats`, `synthesizer`, `text_normalise`, `speakers`, `prompts`, `checkpoints` (weight checks), `training_logs` and `timing` |
| `tests/` | `python -m pytest tests -q`; runs offline on tiny random models |

## Scoring

In the notebook, `results.score(RUNS, ids)` scores every system of the run table that has no scores
yet. It always uses the same 150 test sentences (it rebuilds their split and checks its fingerprint)
and writes `results/<name>/`. One system by hand, from the repo root:

```bash
python scripts/evaluate.py --exp worst_speaker_embedding --model outputs/worst_speaker_embedding/run/model.worst_speaker_embedding.safetensors --speaker_id 24 --manifest dataset/eval_split/test_manifest.csv --limit 150 --out_dir results/worst_speaker_embedding
```

`--model` takes a weights file, a run folder or a Hugging Face model id, and `--ground_truth` scores
the real recordings instead. The script refuses to score a model on sentences from its training or
validation split, and resumes where it stopped.

## Metrics

| Metric | How | Better |
|---|---|---|
| MCD (dB) | 24 mel-cepstral coefficients without c0, frames aligned with dynamic time warping (Kubichek, 1993) | lower |
| WER, CER | Whisper `small.en`, greedy decoding, Whisper's English normaliser on both sides | lower |
| Speaker similarity | cosine of ECAPA-TDNN embeddings to the mean embedding of 50 real LJ training clips | higher |
| UTMOS | UTMOS22 strong (Saeki et al., 2022) | higher |
| Pitch (Hz) | YIN, median over the frames within 20 dB of the loudest | closer to LJ (207 Hz) |
| Duration ratio | synthetic over real length | closer to 1 |

Two systems are compared sentence by sentence: the mean difference, a 95% bootstrap confidence
interval (10,000 resamples), and a Wilcoxon signed-rank test with Holm's correction across the
metrics. A difference is clear when the interval excludes 0 and the corrected p < 0.05.

## Fairness

- Every system reads the same sentences, and each sentence has its own sampling seed (CRC32 of
  `seed:utt_id`), so differences are not sampling luck.
- Real and synthetic audio get the same preprocessing: DC removal, peak normalisation to 0.95 and
  silence trimming at -40 dB, as in `scripts/AudioProcessor.py`.
- The text goes through the training front-end (`clean_normalised_text` and `load_vctk_tokenizer`).
- `results.check_checkpoints` confirms that each run changed only what its experiment intended: only
  its speaker row of the embedding table, and no frozen module (`results/checks.csv`).

## Known limitations

- espeak-ng phonemises each word without its context, so homographs such as "read" and "lead" can
  come out wrong, and abbreviations keep their full stop ("doctor. Smith").
- `clean_normalised_text` deletes digits, so `synth.py` and `evaluate.py` spell numbers out first
  (`text_normalise.py`, rules from keithito/tacotron).
- Whisper, ECAPA and UTMOS are trained models, not listeners; the wavs in `wav_examples/` back them up.
- Each condition was trained once, so the intervals cover sentence-to-sentence variation, not
  run-to-run variation.

## Sources

- VITS: Kim, Kong & Son (2021); Hugging Face `VitsModel`; Y. Lacombe, finetune-hf-vits (MIT).
- Bootstrap: Efron & Tibshirani (1993). Holm (1979), Scandinavian Journal of Statistics.
- YIN: de Cheveigné & Kawahara (2002), JASA 111(4).
- Whisper: Radford et al. (2023). ECAPA-TDNN: Desplanques, Thienpondt & Demuynck (2020), via
  `speechbrain/spkrec-ecapa-voxceleb`. UTMOS: Saeki et al. (2022), via `tarepan/SpeechMOS`.
- MCD: Kubichek (1993), IEEE PACRIM.
- Number expansion: keithito/tacotron `text/numbers.py` (MIT).
