import re
import logging

import unicodedata
from transformers import AddedToken, VitsTokenizer


def configure_espeak() -> None:
    """Configure phonemizer to use the espeak-ng library bundled with its dependency."""
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())


class WordMismatchFilter(logging.Filter):
    """
    Suppress phonemizer's word count warnings.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        is_warning = record.levelno == logging.WARNING
        matching_message = record.getMessage().startswith("words count mismatch on ")
        return not (is_warning and matching_message)


def configure_word_mismatch_warnings(suppress: bool) -> None:
    """
    Configure phonemizer's word count warnings in the current process.
    """
    # initialise the word mismatch filter
    word_mismatch_filter = WordMismatchFilter()

    logger = logging.getLogger("phonemizer")
    if suppress: logger.addFilter(word_mismatch_filter)
    else: logger.removeFilter(word_mismatch_filter)


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
    Load VCTK with distinct IDs.

    Note that this fix specifically addresses the ID collision between the 
    "<unk>" token and the phoneme "ᵻ" in the pretrained VCTK tokenizer. 

    Essentially the token vocabulary was not contiguous, leaving a gap between the IDs of the 
    unknown token and the stroke-i phoneme. It's not an issue if unknown tokens are not used during 
    training or inference, however it's a good idea to fix it.

    Note that this issue was identified by debugging with AI (Github Copilot Sol 6.1) during
    development, however the code is original.
    """

    configure_espeak()

    # load the tokenizer from the pretrained checkpoint with the corrected unknown token mapping
    tokenizer = VitsTokenizer.from_pretrained(checkpoint, unk_token="<unk>",
        added_tokens_decoder={174: AddedToken("<unk>", normalized=False, special=True),},
    )

    # verify that the special token mappings are what we expect
    if tokenizer.encoder.get("ᵻ") != 177: raise ValueError("Unexpected checkpoint mapping for ᵻ")
    # raise an error if ID 174 is occupied by a pretrained symbol other than "<unk>"
    matching = [token for token, token_id in tokenizer.encoder.items() if token_id == 174 and token != "<unk>"]
    if any(matching): raise ValueError("ID 174 is occupied by a pretrained symbol")

    # Assign the "<unk>" token to ID 174 explicitly
    tokenizer.encoder["<unk>"] = 174 
    tokenizer.decoder[174] = "<unk>"  # note that it's in the opposite mapping direction compared to the encoder
    tokenizer.add_special_tokens({"unk_token": "<unk>"}) # Register the unknown token with the tokenizer to refresh vocabulary metadata
    return tokenizer
