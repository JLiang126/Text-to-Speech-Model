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

## Folders

| Folder or file | What is in it |
|---|---|
| `experiment_variables.csv` | every experiment's settings, label and description; the notebook's run table reads it |
| `configs/` | one training config per experiment |
| `outputs/` | one folder per run: the settings it used, its loss histories |
| `checkpoints/` | Holds the checkpoints |
| `results/` | the scores of every system, the checkpoint checks, the speaker ranking, the timing and the report figures |
| `wav_examples/` | the 12 qualitative prompts read by every system |
| `scripts/` | data, training, inference and evaluation code (`scripts/evaluation/README.md` explains the evaluation) |
| `tests/` | `python -m pytest tests -q` |
