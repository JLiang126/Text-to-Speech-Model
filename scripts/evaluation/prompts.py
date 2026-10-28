"""Fixed qualitative prompts for ``wav_examples/`` (task sheet: at least 10 different sentences).

None of these sentences are in LJSpeech. Each one targets a known way TTS front-ends
or models fail, so listening to them feeds straight into the report's error
analysis. They are also the default sentences for the latency/RTF measurement.
"""
from __future__ import annotations

from typing import List, Tuple

# (category, text)
PROMPTS: List[Tuple[str, str]] = [
    ("phonetically balanced", "The birch canoe slid on the smooth planks."),            # Harvard sentence list 1
    ("in-domain formal", "The committee examined the evidence with great care before reaching its conclusion."),
    ("numbers and dates", "On March 3rd, 1865, the treasury paid 12,500 dollars to 47 workers."),
    ("abbreviations", "Dr. Smith and Mr. Jones met at St. Paul's Cathedral at 9 a.m."),
    ("rare proper nouns", "Siobhan and Llewellyn drove from Worcestershire to Gloucester."),
    ("long sentence", "Although the winter had been unusually cold, the villagers still gathered every evening "
                      "in the old hall, where they shared stories, played music, and planned the spring harvest together."),
    ("question", "Did you really expect the report to be finished before the end of the week?"),
    ("exclamation", "What a wonderful surprise this turned out to be!"),
    ("homographs", "I read the book yesterday and will read it again; the lead singer held a lead pipe."),
    ("modern vocabulary", "Please email the PDF to the IT helpdesk and restart the Wi-Fi router."),
    ("very short", "Yes."),
    ("punctuation and pauses", "Wait, stop! Listen carefully; then, slowly, walk away."),
]


def prompt_texts() -> List[str]:
    return [text for _, text in PROMPTS]
