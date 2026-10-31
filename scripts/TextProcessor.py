"""
This script contains both an original implementation of phonemisation and 
tokenisation, as well as helpers for the tokeniser that's included with the 
selected checkpoint (ylacombe/vits-vctk-with-discriminator)

The VITS checkpoint tokeniser is used, although the original implementation is
retained here for general reference.

Note that the ylacombe/vits-vctk-with-discriminator checkpoint uses
eSpeak-ng 1.51, whereas espeaking_loader may use a different version (e.g. 1.52 for 
espeakng_loader-0.2.4)

"""

import re
import logging

from typing import List, Union

import torch
import unicodedata
from transformers import AddedToken, VitsTokenizer

class _WordMismatchFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            record.levelno == logging.WARNING
            and record.getMessage().startswith("words count mismatch on ")
        )


_word_mismatch_filter = _WordMismatchFilter()


def configure_word_mismatch_warnings(suppress: bool) -> None:
    """
    Configure only Phonemizer's word-count warnings in the current process.

    This is used because Phonemizer may emit word-count warnings that are not relevant
    to the current processing context, and we want to selectively suppress them.
    
    Args:
        suppress (bool): If True, suppress word-count warnings; otherwise, show them.

    """
    logger = logging.getLogger("phonemizer")
    if suppress:
        logger.addFilter(_word_mismatch_filter)
    else:
        logger.removeFilter(_word_mismatch_filter)


# 178 symbol vocabulary from the official VITS implementation (jaywalnut310/vits text/symbols.py)
# Ensures input embedding compatibility with pretrained VCTK VITS checkpoints
VITS_SYMBOLS_official = (
    ["_"] # Pad
    + list(';:,."!?¡¿—…"«»“” ') # Punctuation 
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") # Letters
    + list("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ") # IPA Letters
)

# 178 symbols matching the corrected VCTK checkpoint vocabulary for ylacombe/vits-vctk-with-discriminator
# The unused duplicate-apostrophe slot at ID 174 is reserved for <unk>.
VITS_SYMBOLS = (
    ["_"] # Pad
    + list(';:,.!?¡¿—…"«»“” ') # Punctuation
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") # Letters
    + list("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘") # IPA Letters
    + ["<unk>", "\u0329", "'", "ᵻ"]
)

def clean_normalised_text(text: str, lowercase: bool = True) -> str:
    """
    Superceded by clean_normalised_text_simple for Unicode-preserving text cleaning.

    Cleans normalised LJSpeech column 3 transcripts:
    1. NFKC Unicode normalisation
    2. ASCII conversion
    3. Lowercasing
    4. Punctuation mapping
    """
    # NFKC Unicode normalization
    # this ensures that characters like é are decomposed into their base character and combining accent, which will be removed in the next step
    text = unicodedata.normalize("NFKC", str(text))

    # note, this converts café to caf even with the above step
    text = text.encode("ascii", "ignore").decode("ascii")

    if lowercase: text = text.lower()

    text = re.sub(r"[;:—–]+", ",", text)
    text = re.sub(r"[^a-zA-Z\s\.,!\?'-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_normalised_text_simple(text: str, brackets_to_commas: bool = True) -> str:
    """Preserve Unicode text and turn bracket boundaries into comma pauses.

    Note: this function was written and documented by AI, under clear instructions
    about how the bracket-to-comma replacement should behave.

    NFKC normalization runs first. Compatibility forms such as full-width
    parentheses become ordinary brackets. Only (, ), [, and ] are handled;
    curly braces are left unchanged. Opening and closing brackets use identical
    rules: there is no bracket-pair validation or special nesting logic.

    With brackets_to_commas=True, consecutive brackets, including mixed types
    separated only by whitespace, form one group. Whitespace immediately before
    and after the group is included in its replacement. Each group is processed
    using the surrounding NFKC-normalized input, in this order:

    1. If there is no preceding non-whitespace text, remove the group and its
       surrounding whitespace. An opening bracket at the start adds no comma.
       The same applies to an unmatched closing bracket at the start.
    2. If the next non-whitespace character is one of , . ; : ! ?, remove the
       group and its surrounding whitespace. Preserve that punctuation and
       attach it directly to the preceding text; do not insert another comma.
    3. Otherwise, if the previous non-whitespace character is one of , . ; : ! ?,
       replace the group and surrounding whitespace with a single space.
       Preserve the existing punctuation without adding another comma.
    4. Otherwise, replace the group and surrounding whitespace with ", ".

    In rule 4, an opening bracket after a word produces a comma attached
    DIRECTLY TO THE END OF THE PREVIOUS WORD, with no space before the comma.
    For example, "First (aside) next" becomes "First, aside, next", not
    "First ,aside, next". Closing brackets follow the same attachment rule.
    The implementation looks at the preceding character, not at word structure:
    quotes and dashes are not punctuation exceptions, so a comma can attach to
    those characters instead. Existing punctuation combinations are not repaired.

    Examples with comma replacement enabled:
        "(Aside) next"             -> "Aside, next"
        ")Aside next"              -> "Aside next"
        "First[aside]next"         -> "First, aside, next"
        "First, (aside) next"      -> "First, aside, next"
        "First: [aside] next"      -> "First: aside, next"
        "First (aside), next"      -> "First, aside, next"
        "First (aside)."           -> "First, aside."
        "First (aside!) next"      -> "First, aside! next"
        "First (aside)"            -> "First, aside,"
        "(Aside)"                  -> "Aside,"
        "First ("                  -> "First,"
        "First ((aside)) next"     -> "First, aside, next"
        "First (outer [inner]) next" -> "First, outer, inner, next"
        "First () next"            -> "First, next"
        "First ) ( next"           -> "First, next"
        "(one)(two)"               -> "one, two,"
        " ( ) [ ] "                -> ""
        'First "(aside)" next'     -> 'First ", aside, " next'
        "First - (aside) next"     -> "First -, aside, next"
        "First,()!"                -> "First,!"

    Thus, empty groups in the middle can still add comma pauses, unmatched
    brackets are still replaced, and closing brackets at the end normally
    leave a trailing comma. Final whitespace trimming does not remove commas.

    With brackets_to_commas=False, replace each bracket with a space instead.
    No comma or punctuation-adjacency rules apply in this mode:
        "First(aside)next"         -> "First aside next"
        "(Aside)"                  -> "Aside"
        "First (aside)."           -> "First aside ."

    In both modes, collapse repeated whitespace and strip surrounding whitespace
    last. Case, accents, digits, and other punctuation are not filtered out.
    """
    text = unicodedata.normalize("NFKC", text)

    if brackets_to_commas:
        def replace_brackets(match: re.Match[str]) -> str:
            before = text[:match.start()].rstrip()
            after = text[match.end():].lstrip()
            if not before or (after and after[0] in ",.;:!?"):
                return ""
            if before[-1] in ",.;:!?":
                return " "
            return ", "

        text = re.sub(
            r"\s*[\[\]()](?:\s*[\[\]()])*\s*", replace_brackets, text,
        )
    else:
        # note: the tokeniser will replace excessive whitespace with single spaces, 
        # so we don't need to worry about multiple spaces here
        text = text.translate(str.maketrans("()[]", "    "))

    return re.sub(r"\s+", " ", text).strip()

def load_vctk_tokenizer(checkpoint: str) -> VitsTokenizer:
    """
    Load VCTK with distinct IDs for the unknown token and stroke-i phoneme.

    Note that this fix specifically addresses the ID collision between the 
    "<unk>" token and the phoneme "ᵻ" in the pretrained VCTK tokenizer. It 
    does not alter other aspects of the vocabulary or the pretrained embeddings.

    Also note that this fix isn't strictly neccessary if no unknown tokens are 
    expected to collide with the phoneme "ᵻ", as is the case thanks to our
    preprocessing. However it's nice to have this safeguard in place to avoid
    potential collisions in future scenarios.

    Args:
        checkpoint (str): Path to the pretrained VCTK tokenizer checkpoint.

    -----This code was written and documented by AI.----

    Cause of the collision:
        The original checkpoint vocabulary has 177 dictionary entries, but its
        IDs span 0 through 177 because ID 174 is unused. The valid phoneme
        "ᵻ" (U+1D7B) already occupies pretrained embedding ID 177.

        The affected default tokenizer loading path assigns an additional
        "<unk>" token using the vocabulary-entry count as the next ID: 177.
        This aliases "<unk>" and "ᵻ". Valid "ᵻ" inputs still reach their correct
        pretrained embedding, but ID 177 decodes as "<unk>", disappears when
        special tokens are skipped, and is misclassified by unknown-ID checks.
        Genuine unknown characters also map to that phoneme's embedding.

        This is a special-token ID collision, not an unsupported phoneme.
        The different character "ɨ" (U+0268, ID 101) is unaffected.

    Fix applied here:
        1. Assign "<unk>" to the unused ID 174 in added_tokens_decoder during
           construction, preventing the automatic assignment of ID 177.
        2. Verify that "ᵻ" still maps to 177 and that no other pretrained symbol
           occupies 174. Unexpected layouts raise an error rather than being
           silently remapped.
        3. Add "<unk>" at 174 to the base encoder and decoder, filling the gap.
        4. Register the already assigned unknown token with add_special_tokens
           to refresh vocabulary metadata without allocating another ID.

    The resulting vocabulary has 178 distinct entries: "<unk>" is 174 and
    "ᵻ" remains 177. All original symbol IDs and model weights are unchanged;
    no embedding resizing or phoneme substitution is required. save_pretrained
    persists the corrected vocabulary and special-token metadata. Loading an
    already corrected checkpoint through this helper is also supported.
    """
    tokenizer = VitsTokenizer.from_pretrained(
        checkpoint,
        unk_token="<unk>",
        added_tokens_decoder={
            174: AddedToken("<unk>", normalized=False, special=True),
        },
    )

    if tokenizer.encoder.get("ᵻ") != 177:
        raise ValueError("Unexpected checkpoint mapping for ᵻ")
    if any(
        token_id == 174 and token != "<unk>"
        for token, token_id in tokenizer.encoder.items()
    ):
        raise ValueError("ID 174 is occupied by a pretrained symbol")

    tokenizer.encoder["<unk>"] = 174
    tokenizer.decoder[174] = "<unk>"
    tokenizer.add_special_tokens({"unk_token": "<unk>"})
    return tokenizer

class TextProcessor:
    """
    Converts normalised text to IPA phonemes using espeak. 
    Intersperses blank tokens for VITS Monotonic Alignment Search.

    NOTE:
    The checkpoint already comes with its own tokenization and symbol set, so we'll just 
    use that instead.
    
    This is not a drop-in replacement for the built-in tokeniser, as the training
    script expects certain methods to be available on the tokeniser object such as 
    pad_token_id, save_pretrained, __call__(text), unk_token_id, unk_token and
    convert_tokens_to_ids

    It also doesn't apply vocabulary-aware normalisation
    
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
        """Converts text to VITS IPA IDs, rejecting unmapped phoneme characters."""
        clean_text = text if is_cleaned else clean_normalised_text_simple(text)
        ipa_str = self.phonemize_text(clean_text)

        # raise error for unknown phoneme characters
        unknown_chars = sorted(set(ipa_str) - self.symbol_to_id.keys())
        if unknown_chars:
            raise ValueError(
                f"Unknown phoneme characters {unknown_chars!r} "
                f"in transcript {clean_text!r}: {ipa_str!r}"
            )
        sequence = [self.symbol_to_id[c] for c in ipa_str]
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