# Evaluation

Scores every system on the same held-out test sentences, with the same preprocessing,
seeds and hardware, so the only thing that differs between experiments is the
controlled change.

## Quick start (run from the repo root)

```bash
pip install -r requirements.txt -r requirements-eval.txt
export VITS_DIR=vits-main            # folder with the official VITS code (models.py)

# 1. Ceiling: the real recordings
python scripts/evaluate.py --exp ground_truth --ground_truth

# 2. Floor: the VCTK-pretrained model before any fine-tuning
python scripts/evaluate.py --exp pretrained --config vits-main/configs/vctk_base.json \
    --checkpoint checkpoints/pretrained_vctk.pth --speaker_id <ID>

# 3. Each experiment (repeat for expB, and for several steps to get learning curves)
python scripts/evaluate.py --exp expA --config configs/expA.json \
    --checkpoint checkpoints/expA/G_10000.pth --speaker_id <ID> --out_dir results/expA_step10000

# 4. Table, paired tests and figures for the report
python scripts/compare.py results/ground_truth results/pretrained results/expA_step10000 \
    results/expB_step10000 --pair expA expB --out results/comparison

# 5. The >=10 qualitative wavs for wav_examples/
python scripts/synth.py --prompts --exp expA --config configs/expA.json \
    --checkpoint checkpoints/expA/G_10000.pth --speaker_id <ID> --out_dir wav_examples

# Smoke test without any downloads (5 sentences, MCD + timing only)
python scripts/evaluate.py --exp expA ... --limit 5 --metrics mcd,rtf
```

`--speaker_id` is only for multi-speaker checkpoints. Use the ID the LJSpeech clips
were fine-tuned with. Single-speaker checkpoints take no `--speaker_id`.

Tests: `python -m pytest tests -q`. They need no internet. The VITS test runs when
`VITS_DIR` points at the VITS code.

## Metrics

All metrics are computed per sentence and reported as a mean with a 95% bootstrap
confidence interval (10,000 resamples over sentences).

| Metric | What it measures | Better | How |
|---|---|---|---|
| MCD-DTW (dB) | How far the spectrum is from the real recording of the same sentence | lower | 24 mel-cepstral coefficients (c0 dropped, so loudness doesn't count), aligned with dynamic time warping (Kubichek 1993) |
| Duration ratio | Speaking rate compared with the real recording | closer to 1 | synthesised frames / reference frames |
| WER / CER | Intelligibility: whether a speech recogniser hears the right words | lower | Whisper `small.en`, greedy decoding, Whisper's English text normaliser on both sides. Corpus rate = total errors / total reference words |
| Speaker similarity | How much it sounds like the LJSpeech speaker | higher | cosine similarity of ECAPA-TDNN embeddings (SpeechBrain) to the average of 50 real *training* clips. `secs_paired` compares with the real recording of the same sentence |
| UTMOS | Predicted naturalness (about 1-5) | higher | UTMOS22 strong learner (Saeki et al. 2022) via SpeechMOS |
| Latency / RTF | Speed | lower | see protocol below |

How to read them:
- Every number has two reference points:
  - **ground_truth** (real recordings) is the best achievable. Its WER is the recogniser's own error rate.
  - **pretrained** (no fine-tuning) shows where we started.
- MCD is computed the same way for every system, so it is valid for comparing our experiments. Its absolute values differ from papers that use SPTK mel-cepstra.
- Whisper, ECAPA and UTMOS are learned models, not human listeners. Treat them as proxies and back them up with the audio examples.

## Latency / RTF protocol

- **Sentences and repeats:** the 12 fixed prompts (`scripts/evaluation/prompts.py`), each synthesised 5 times after 3 untimed warm-up calls.
- **Batch and seed:** batch size 1 and a fixed seed, so every repeat has the same output length.
- **GPU timing:** `torch.cuda.synchronize()` runs before starting and after stopping the clock.
- **Definitions:** latency is text in to waveform out, including the espeak front-end (its share is reported as `frontend_share`). RTF = latency / audio duration; RTF < 1 is faster than real time.
- **Hardware:** the device, CPU, GPU and library versions go into `summary.json`. **Time every experiment on the same machine and GPU**, otherwise the hardware becomes a second variable.

## Controls that keep the comparison fair

- **Same sentences:** the same test manifest for every system, never used for training or for picking checkpoints.
- **Same seeds:** each sentence gets its own seed (CRC32 of `base_seed:utt_id`), identical across systems. VITS samples noise at inference (`noise_scale=0.667`, `noise_scale_w=0.8`), so without fixed seeds part of any difference would just be sampling luck.
- **Same preprocessing for all audio:** DC removal, peak normalisation to 0.95, and silence trimming at -40 dB, mirroring `scripts/AudioProcessor.py`.
- **Paired comparisons** (`compare.py`): B minus A on the same sentences, with a bootstrap CI and a Wilcoxon signed-rank p-value.
- **What the CIs cover:** sentence-to-sentence variation only, not training-run variation. That needs a second training seed. With fewer than about 20 sentences the bootstrap CIs are too narrow.

## Outputs

`results/<system>/`:
- `per_utterance.csv`: every metric and the ASR transcript, one row per sentence.
- `summary.json`: means, CIs, corpus WER/CER, latency, device, settings and versions.
- `rtf_runs.csv`: one row per timed call.
- `wavs/`: the synthesised test sentences. These are git-ignored, because they are regenerated from checkpoints.

`results/comparison/`:
- `comparison_table.csv` and `comparison_table.md` (paste the `.md` into the report).
- `paired_tests.csv`.
- `comparison.png`.
- `learning_curves.png`, when an experiment was scored at two or more training steps.

## Known limitations found while testing the pipeline

- **Numbers are dropped:** `clean_normalised_text` deletes digits and symbols. "12,500 dollars in 1865" becomes ", dollars in", so numbers in arbitrary input text disappear. Test sentences are unaffected, because LJSpeech's normalised column already spells numbers out. `tests/test_text_frontend.py::test_numbers_survive_text_cleaning` documents this as an expected failure until numbers are expanded.

## Sources

- **VITS:** Kim, Kong & Son (2021), https://github.com/jaywalnut310/vits (MIT).
- **Whisper:** Radford et al. (2023), https://github.com/openai/whisper (MIT).
- **ECAPA-TDNN:** Desplanques, Thienpondt & Demuynck (2020), https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb (Apache-2.0).
- **UTMOS:** Saeki et al. (2022), via https://github.com/tarepan/SpeechMOS (MIT).
- **MCD:** Kubichek (1993), 'Mel-cepstral distance measure for objective speech quality assessment', IEEE PACRIM.
- **Speaker similarity as SECS:** Casanova et al. (2022), YourTTS.
- **Harvard sentence** in the prompts: IEEE (1969) recommended practice for speech quality measurements.
