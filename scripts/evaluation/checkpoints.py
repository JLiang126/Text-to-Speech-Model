"""Checks that each fine-tuned checkpoint did what its experiment config says.

Every run is compared with the pretrained weights it started from:
- **Speaker row:** only the configured ``speaker_id`` row of the speaker table receives
  gradient, so it should move far more than any other row. The other rows still shrink a
  little, because the optimiser (``torch.optim.AdamW``) applies its default weight decay
  to the whole table.
- **Frozen modules:** a module frozen by the freeze mode should be unchanged. "Unchanged"
  allows ``UNCHANGED_REL`` of relative change, because training switches weight norm on
  and off again, which leaves float rounding of about 1e-7.

The posterior encoder is only used in training, so its change does not affect synthesis.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np

MODULES = ("text_encoder", "duration_predictor", "flow", "posterior_encoder", "decoder", "embed_speaker")
UNCHANGED_REL = 1e-4      # relative change below this counts as unchanged
ROW_DOMINANCE = 5.0       # the trained row must move at least this many times more than any other row

# freeze_mode in experiments.csv -> modules that must stay unchanged
EXPECTED_UNCHANGED = {
    "none": (),
    "freeze_text_encoder": ("text_encoder",),
    "unfreeze_speech_embeddings_only": tuple(m for m in MODULES if m != "embed_speaker"),
}


def state_of(model) -> Dict[str, np.ndarray]:
    """VitsModel -> {parameter name: float64 array}, in one layout for every checkpoint."""
    return {k: v.detach().double().cpu().numpy() for k, v in model.state_dict().items()}


def module_changes(state: Mapping[str, np.ndarray], base: Mapping[str, np.ndarray],
                   modules: Iterable[str] = MODULES) -> Dict[str, float]:
    """Relative L2 change per module: ||theta - theta_base|| / ||theta_base||."""
    changes = {}
    for module in modules:
        keys = [k for k in base if k.split(".", 1)[0] == module]
        if not keys:
            continue
        diff = sum(float(np.sum((state[k] - base[k]) ** 2)) for k in keys)
        norm = sum(float(np.sum(base[k] ** 2)) for k in keys)
        changes[module] = (diff / norm) ** 0.5 if norm > 0 else float("nan")
    return changes


def speaker_row_changes(state: Mapping[str, np.ndarray], base: Mapping[str, np.ndarray]) -> np.ndarray:
    """Relative change of every row of the speaker table (empty for single-speaker models)."""
    if "embed_speaker.weight" not in base:
        return np.zeros(0)
    delta = np.linalg.norm(state["embed_speaker.weight"] - base["embed_speaker.weight"], axis=1)
    return delta / np.maximum(np.linalg.norm(base["embed_speaker.weight"], axis=1), 1e-12)


def check_experiment(state: Mapping[str, np.ndarray], base: Mapping[str, np.ndarray],
                     speaker_id: Optional[int], freeze_mode: str = "none") -> Dict[str, object]:
    """One row of the checks table: which speaker row trained, which modules stayed unchanged."""
    changes = module_changes(state, base)
    rows = speaker_row_changes(state, base)
    result: Dict[str, object] = {f"change_{m}": c for m, c in changes.items()}
    if rows.size >= 2:
        order = np.argsort(-rows)
        top, runner_up = int(order[0]), int(order[1])
        result.update(trained_row=top, trained_row_change=float(rows[top]),
                      next_row=runner_up, next_row_change=float(rows[runner_up]))
        dominant = rows[top] >= ROW_DOMINANCE * max(rows[runner_up], 1e-12)
        result["speaker_ok"] = bool(speaker_id is not None and top == int(speaker_id) and dominant)
    unchanged = [m for m, c in changes.items() if c < UNCHANGED_REL]
    result["unchanged_modules"] = ", ".join(unchanged) if unchanged else "none"
    expected = EXPECTED_UNCHANGED.get(str(freeze_mode).strip().lower())
    result["freeze_ok"] = None if expected is None else set(unchanged) == set(expected)
    return result
