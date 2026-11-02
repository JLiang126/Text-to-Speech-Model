import re

from typing import List, Union

import torch
import unicodedata


import re
import logging

from typing import List, Union

import torch
import unicodedata
from transformers import AddedToken, VitsTokenizer



class WordMismatchFilter(logging.Filter):
    """
    Filter for suppressing Phonemizer's word-count warnings.

    This filter checks if a log record is a warning and if its message starts
    with "words count mismatch on ". If both conditions are met, the record
    is filtered out (i.e., suppressed).
    """
    def filter(self, record: logging.LogRecord) -> bool:
        is_warning = record.levelno == logging.WARNING
        matching_message = record.getMessage().startswith("words count mismatch on ")
        return not (is_warning and matching_message)


def configure_word_mismatch_warnings(suppress: bool) -> None:
    """
    Configure only Phonemizer's word-count warnings in the current process.

    This is used because Phonemizer may emit word-count warnings that are not relevant
    to the current processing context, and we want to selectively suppress them.

    I've added this code since the checkpoint's tokeniser doesn't directly
    allow for setting words_mismatch="ignore" in the phonemizer directly.
    
    Args:
        suppress (bool): If True, suppress word-count warnings; otherwise, show them.

    """
    # initialise the word mismatch filter
    word_mismatch_filter = WordMismatchFilter()

    logger = logging.getLogger("phonemizer")
    if suppress:
        logger.addFilter(word_mismatch_filter)
    else:
        logger.removeFilter(word_mismatch_filter)


# 178 symbol vocabulary from the official VITS implementation (jaywalnut310/vits text/symbols.py)
# Ensures input embedding compatibility with pretrained VCTK VITS checkpoints
# note that the official checkpoint is NOT USED. This is retained for reference only.
VITS_SYMBOLS_official = (
    ["_"] # Pad
    + list(';:,."!?¡¿—…"«»“” ') # Punctuation 
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") # Letters
    + list("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ") # IPA Letters
)

# 178 symbols matching the corrected VCTK checkpoint vocabulary for ylacombe/vits-vctk-with-discriminator
# The unused duplicate-apostrophe slot at ID 174 is reserved for <unk>.
# see vocab.json in the ylacombe/vits-vctk-with-discriminator HuggingFace repository for reference
VITS_SYMBOLS = (
    ["_"] # Pad
    + list(';:,.!?¡¿—…"«»“” ') # Punctuation
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") # Letters
    + list("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘") # IPA Letters
    + ["<unk>", "\u0329", "'", "ᵻ"]
)


def replace_brackets(match: re.Match[str]) -> str:
    """
    Replace bracket groups according to the surrounding text and punctuation.
    """
    text = match.string

    before = text[:match.start()].rstrip() # find the text before the bracket and strip trailing whitespace
    after = text[match.end():].lstrip() # find the text after the bracket and strip leading whitespace
    # A leading bracket has no preceding word to attach a comma to. Delete it.
    if not before:
        return ""

    # if punctuation follows immediately after the brackets, don't add a comma or space
    punctuation_follows = bool(after) and after[0] in ",.;:!?"
    if punctuation_follows:
        return ""

    # If the preceding word already ends in punctuation, retain that without adding another comma
    if before[-1] in ",.;:!?":
        return " "

    # Otherwise, attach a comma to the preceding word and leave a
    # space before the following text: "word (aside)" -> "word, aside".
    return ", "

def clean_normalised_text(text: str, lowercase: bool = False, brackets_to_commas: bool = True) -> str:
    """
    Cleans normalised LJSpeech column 3 transcripts:
    1. NFKC Unicode normalisation
    2. Optional lowercasing
    3. Punctuation mapping

    Note that lowercasing is unnecessary for IPA characters and may affect proper nouns.

    Args:
        text (str): The input text to clean.
        lowercase (bool): Whether to convert the text to lowercase.
        brackets_to_commas (bool): Whether to replace brackets with commas. Otherwise they're stripped.
    Returns:
        str: The cleaned and normalised text.
    """
    text = unicodedata.normalize("NFKC", str(text))

    # MPC: I've commented out the ASCII conversion step because it may remove IPA characters 
    # necessary for the VITS model. I.e "café" becomes "caf"
    # text = text.encode("ascii", "ignore").decode("ascii")

    if lowercase: text = text.lower()

    # Replace brackets with commas if the option is enabled
    # this attempts to handle brackets in a context-sensitive manner, ensuring proper punctuation and spacing around them.
    if brackets_to_commas:
        # handle brackets, square brackets, and possible groups of brackets
        text = re.sub(r"\s*[\[\]()](?:\s*[\[\]()])*\s*", replace_brackets, text)

    text = re.sub(r"[;:—–]+", ",", text) # Replace sequences of semicolons, colons, and dashes with a comma
    text = re.sub(r"[^a-zA-Z\s\.,!\?'-]", "", text) # characters which are outside this group are stripped
    text = re.sub(r"\s+", " ", text).strip() # Collapse repeated whitespace and strip surrounding whitespace
    return text


def load_vctk_tokenizer(checkpoint: str) -> VitsTokenizer:
    """
    Load VCTK with distinct IDs for the unknown token and stroke-i phoneme.

    Note that this fix specifically addresses the ID collision between the 
    "<unk>" token and the phoneme "ᵻ" in the pretrained VCTK tokenizer. 

    Note that this issue was identified by debugging with AI (Github Copilot Sol 6.1) during
    development, however the code is original.
    """

    # load the tokenizer from the pretrained checkpoint with the corrected unknown token mapping
    tokenizer = VitsTokenizer.from_pretrained(checkpoint, unk_token="<unk>",
        added_tokens_decoder={174: AddedToken("<unk>", normalized=False, special=True),},
    )

    # verify that the special token mappings are correct
    if tokenizer.encoder.get("ᵻ") != 177:
        raise ValueError("Unexpected checkpoint mapping for ᵻ")
    # raise an error if ID 174 is occupied by a pretrained symbol other than "<unk>"
    matching = [token for token, token_id in tokenizer.encoder.items() if token_id == 174 and token != "<unk>"]
    if any(matching):
        raise ValueError("ID 174 is occupied by a pretrained symbol")

    # Assign the "<unk>" token to ID 174 explicitly
    tokenizer.encoder["<unk>"] = 174 
    # Assign the "<unk>" token to ID 174 in the decoder as well
    # note that it's in the opposite mapping direction compared to the encoder
    tokenizer.decoder[174] = "<unk>" 
    tokenizer.add_special_tokens({"unk_token": "<unk>"}) # Register the unknown token with the tokenizer to refresh vocabulary metadata
    return tokenizer


class TextProcessor:
    """
    Converts normalised text to IPA phonemes using espeak. Intersperses blank tokens for VITS Monotonic Alignment Search.
    
    NOTE:
    The checkpoint already comes with its own tokenization and symbol set, so we'll just 
    use that instead.
    
    This is not a drop-in replacement for the built-in tokeniser, as the training
    script expects certain methods/attributes to be available on the tokeniser object

    The espeak library version may be different, which can affect pronunciation output.

    The code is retained as a reference to explain the general phonemization process used by VITS.
    
    """
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
        """Converts text into a 1D  of VITS IPA token IDs."""
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