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

``speaker_embedding_change`` compares one speaker's vector before and after fine-tuning
(relative L2 and cosine), with the reference scales needed to read the numbers.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np

SPEAKER_KEY = "embed_speaker.weight"

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
    if SPEAKER_KEY not in base:
        return np.zeros(0)
    delta = np.linalg.norm(state[SPEAKER_KEY] - base[SPEAKER_KEY], axis=1)
    return delta / np.maximum(np.linalg.norm(base[SPEAKER_KEY], axis=1), 1e-12)


def speaker_embedding_change(after: np.ndarray, before: np.ndarray, speaker_id: int) -> Dict[str, float]:
    """How far one speaker's vector moved in fine-tuning, and the scales to read that by.

    ``after`` and ``before`` are speaker tables (n_speakers x dim) of the fine-tuned and the
    pretrained model.
    - ``rel_l2``: ||after_i - before_i|| / ||before_i||.
    - ``cosine``: cosine(after_i, before_i); 1 = same direction. Unlike rel_l2 it ignores the
      uniform shrink from weight decay.
    - ``decay_rel_l2``: median rel_l2 of the other rows, which receive no gradient and only
      shrink by weight decay: the change the optimiser makes without any training signal.
    - ``move_vs_gap``: ||after_i - before_i|| / median_j ||before_j - before_i|| (j != i), the
      move as a share of the typical distance between two different pretrained voices.
    - ``nearest_pretrained``: the pretrained speaker closest (cosine) to after_i.
    - ``closest_other_cosine``: cosine of before_i to its most similar other pretrained voice.
    """
    after, before = np.asarray(after, dtype=np.float64), np.asarray(before, dtype=np.float64)
    if after.shape != before.shape or after.ndim != 2:
        raise ValueError(f"speaker tables differ in shape: {after.shape} vs {before.shape}")
    i = int(speaker_id)
    if not 0 <= i < before.shape[0]:
        raise ValueError(f"speaker_id {i} is outside the table (0..{before.shape[0] - 1})")
    norms = np.maximum(np.linalg.norm(before, axis=1), 1e-12)
    unit = before / norms[:, None]
    others = np.array([j for j in range(before.shape[0]) if j != i])
    delta = after[i] - before[i]
    rel_rows = np.linalg.norm(after - before, axis=1) / norms
    after_unit = after[i] / max(float(np.linalg.norm(after[i])), 1e-12)
    gaps = np.linalg.norm(before[others] - before[i], axis=1)
    return {"speaker_id": i, "norm_before": float(norms[i]), "norm_after": float(np.linalg.norm(after[i])),
            "rel_l2": float(np.linalg.norm(delta) / norms[i]),
            "cosine": float(after_unit @ unit[i]),
            "decay_rel_l2": float(np.median(rel_rows[others])) if others.size else float("nan"),
            "move_vs_gap": float(np.linalg.norm(delta) / np.median(gaps)) if others.size else float("nan"),
            "nearest_pretrained": int(np.argmax(unit @ after_unit)),
            "closest_other_cosine": float(np.max(unit[others] @ unit[i])) if others.size else float("nan")}


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
        if speaker_id is not None and 0 <= int(speaker_id) < rows.size:  # the configured row, before vs after
            move = speaker_embedding_change(state[SPEAKER_KEY], base[SPEAKER_KEY], int(speaker_id))
            result.update(speaker_rel_l2=move["rel_l2"], speaker_cosine=move["cosine"],
                          speaker_move_vs_gap=move["move_vs_gap"], speaker_nearest=move["nearest_pretrained"])
    unchanged = [m for m, c in changes.items() if c < UNCHANGED_REL]
    result["unchanged_modules"] = ", ".join(unchanged) if unchanged else "none"
    expected = EXPECTED_UNCHANGED.get(str(freeze_mode).strip().lower())
    result["freeze_ok"] = None if expected is None else set(unchanged) == set(expected)
    return result
