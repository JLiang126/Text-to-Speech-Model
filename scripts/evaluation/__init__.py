"""Evaluation package for the A3 text-to-speech experiments (see README.md in this folder).

Modules
-------
audio           Audio loading, resampling and the shared evaluation preprocessing.
manifest        Reads the split manifests written by ``scripts/DataLoader.build_splits``.
text_normalise  Spells out numbers/money/%/& so arbitrary input text keeps its numbers.
synthesizer     Text -> waveform with a trained Hugging Face-format VITS model (seeded, validated).
metrics         MCD-DTW, WER/CER (Whisper), speaker similarity (ECAPA), UTMOS.
checkpoints     Checks that a fine-tuned checkpoint changed what its experiment config says.
speakers        Reference voice centroid and the VCTK speaker ranking (with bootstrap uncertainty).
timing          Latency / real-time-factor (RTF) measurement protocol.
stats           Bootstrap confidence intervals, paired tests, Holm correction.
prompts         The fixed qualitative prompts used for ``wav_examples/``.

Command-line entry points live one level up: ``scripts/synth.py``, ``scripts/evaluate.py``,
``scripts/compare.py``, ``scripts/rank_speakers.py`` and ``scripts/evaluate_experiments.py``.
"""
