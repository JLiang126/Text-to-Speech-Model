"""Evaluation package for the A3 text-to-speech experiments.

Modules
-------
audio        Audio loading, resampling and the shared evaluation preprocessing.
manifest     Reads the split manifests written by ``scripts/DataLoader.build_splits``.
metrics      Objective metrics: MCD-DTW, WER/CER (Whisper), speaker similarity (ECAPA), UTMOS.
timing       Latency / real-time-factor (RTF) measurement protocol.
stats        Bootstrap confidence intervals and paired comparisons between experiments.
synthesizer  Text -> waveform with a trained VITS checkpoint.
prompts      The fixed qualitative prompts used for ``wav_examples/``.

Command-line entry points live one level up: ``scripts/synth.py``,
``scripts/evaluate.py`` and ``scripts/compare.py``.
"""
