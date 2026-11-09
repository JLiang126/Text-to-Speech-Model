# Text-to-Speech-Model

ARTI5001 Assignment 3, group 55: fine-tuning the pretrained VITS model
`ylacombe/vits-vctk-with-discriminator` on LJSpeech. Everything is in `55_A3.ipynb`.

## Setup

```bash
python3 -m venv .venv
pip install -r requirements.txt
```

Put LJSpeech dataset in `dataset/LJSpeech-1.1/`, and run 

```bash
jupyter notebook
```

Keep `experiment_variables.csv` in the repository root. To use the model place
`group55_VITS_checkpoints.zip` in the root and run `55_A3.ipynb`, the notebook checks the
expected files and adds archive files missing from `outputs/` without overwriting existing
run files. The optional training smoke test is disabled by default; enabling it restarts
`outputs/trial_run/`. For full training instead of the supplied checkpoints, set
`full_training = True` near the start of the notebook.

## Folders

| Folder or file | What is in it |
|---|---|
| `experiment_variables.csv` | every experiment's settings, label and description; the notebook's run table reads it |
| `configs/` | one training config per experiment |
| `outputs/` | one folder per run: its checkpoint, settings and loss histories |
| `group55_VITS_checkpoints.zip` | supplied run folders, extracted by the notebook into `outputs/` |
| `results/` | the scores of every system, the checkpoint checks, the speaker ranking, the timing and the report figures |
| `wav_examples/` | the 12 qualitative prompts read by every system |
| `scripts/` | data, training, inference and evaluation code (`scripts/evaluation/README.md` explains the evaluation) |
| `tests/` | `python -m pytest tests -q` |
