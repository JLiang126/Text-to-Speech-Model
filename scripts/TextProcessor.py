import re

from typing import List, Union

import torch
import unicodedata

# 178 symbol vocabulary from the official VITS implementation (jaywalnut310/vits text/symbols.py)
# Ensures input embedding compatibility with pretrained VCTK VITS checkpoints
VITS_SYMBOLS = (
    ["_"] # Pad
    + list(';:,."!?¡¿—…"«»“” ') # Punctuation 
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") # Letters
    + list("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ") # IPA Letters
)

def clean_normalised_text(text: str, lowercase: bool = True) -> str:
    """
    Cleans normalised LJSpeech column 3 transcripts:
    1. NFKC Unicode normalisation
    2. ASCII conversion
    3. Lowercasing
    4. Punctuation mapping
    """
    text = unicodedata.normalize("NFKC", str(text))

    text = text.encode("ascii", "ignore").decode("ascii")

    if lowercase: text = text.lower()

    text = re.sub(r"[;:—–]+", ",", text)
    text = re.sub(r"[^a-zA-Z\s\.,!\?'-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class TextProcessor:
    """Converts normalised text to IPA phonemes using espeak. Intersperses blank tokens for VITS Monotonic Alignment Search."""
    def __init__(self, add_blank: bool = True): # Can built for a possible investigation into the affects for the blank interspered tokens (Can remove)
        self.add_blank = add_blank
        self.symbols = VITS_SYMBOLS
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.id_to_symbol = {i: s for i, s in enumerate(self.symbols)}
        self.blank_id = 0
        self._backend = None

    @property
    def vocab_size(self) -> int: return len(self.symbols)

    def _init_espeak(self) -> None:
        """Initialises the eSpeak-ng phonemiser backend."""
        if self._backend is None:
            import espeakng_loader
            from phonemizer.backend import EspeakBackend
            from phonemizer.backend.espeak.wrapper import EspeakWrapper

            EspeakWrapper.set_library(espeakng_loader.get_library_path())
            EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

            # Matches official VITS settings
            self._backend = EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)

    def phonemize_text(self, text: str) -> str:
        """Converts english text into IPA phonemes using the espeak engine mirroring VITS."""
        self._init_espeak()
        ipa_phonemes = self._backend.phonemize([text], strip=True)
        return ipa_phonemes[0] if ipa_phonemes else ""

    def text_to_sequence(self, text: str, is_cleaned: bool = False) -> torch.LongTensor:
        """Converts text into a 1D LongTensor of VITS IPA token IDs."""
        clean_text = text if is_cleaned else clean_normalised_text(text, lowercase=True)
        ipa_str = self.phonemize_text(clean_text)

        sequence = [self.symbol_to_id[c] for c in ipa_str if c in self.symbol_to_id]
        if not sequence: sequence = [self.symbol_to_id.get(" ", 0)]

        if self.add_blank:
            interspersed = [self.blank_id] * (len(sequence) * 2 + 1)
            interspersed[1::2] = sequence
            sequence = interspersed

        return torch.LongTensor(sequence)

    # For sanity checks
    def sequence_to_text(self, sequence: Union[torch.Tensor, List[int]]) -> str:
        """Reconstructs readable IPA string from tokens."""
        ids = sequence.tolist() if isinstance(sequence, torch.Tensor) else list(sequence)
        return "".join(self.id_to_symbol.get(i, "") for i in ids if i != self.blank_id)