"""Text front-end checks: the symbol table must match the pretrained VCTK checkpoint.

The checkpoint's text-encoder embedding has one row per symbol. If our table differs (an
extra, missing or reordered symbol), every token ID points at the wrong embedding row.
The official strings are copied verbatim from jaywalnut310/vits ``text/symbols.py`` (MIT).
The Hugging Face VCTK checkpoint (ylacombe/vits-vctk-with-discriminator) uses the same
order, except that row 174, a duplicate apostrophe in the official table, holds ``<unk>``
(see ``load_vctk_tokenizer`` in scripts/TextProcessor.py).
"""
import pytest

from scripts.TextProcessor import VITS_SYMBOLS, TextProcessor, clean_normalised_text
from scripts.evaluation.text_normalise import normalise_input_text

OFFICIAL_PAD = '_'
OFFICIAL_PUNCTUATION = ';:,.!?¡¿—…"«»“” '
OFFICIAL_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
OFFICIAL_LETTERS_IPA = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
OFFICIAL_SYMBOLS = [OFFICIAL_PAD] + list(OFFICIAL_PUNCTUATION) + list(OFFICIAL_LETTERS) + list(OFFICIAL_LETTERS_IPA)
UNK_ID = 174


def test_official_table_has_178_symbols():
    assert len(OFFICIAL_SYMBOLS) == 178


def test_symbol_table_matches_the_vctk_checkpoint():
    ours = list(VITS_SYMBOLS)
    assert len(ours) == len(OFFICIAL_SYMBOLS) == 178 and len(set(ours)) == 178
    differences = [i for i, (a, b) in enumerate(zip(ours, OFFICIAL_SYMBOLS)) if a != b]
    assert differences == [UNK_ID], f"symbols differ from the checkpoint order at {differences}"
    assert ours[UNK_ID] == "<unk>" and ours[177] == "ᵻ"


def test_space_token_id_matches_official():
    assert TextProcessor().symbol_to_id[" "] == OFFICIAL_SYMBOLS.index(" ") == 16


def test_blank_tokens_are_interspersed():
    ids = TextProcessor(add_blank=True).text_to_sequence("hello world")
    assert len(ids) % 2 == 1
    assert bool((ids[0::2] == 0).all())


@pytest.mark.xfail(strict=True, reason="Known, handled elsewhere: clean_normalised_text on its own deletes "
                                       "digits. LJSpeech training text has no digits, and synth.py/evaluate.py "
                                       "spell numbers out first (scripts/evaluation/text_normalise.py). If the "
                                       "expansion moves into clean_normalised_text, remove this marker.")
def test_numbers_survive_text_cleaning():
    cleaned = clean_normalised_text("The treasury paid 12,500 dollars in 1865.")
    assert any(ch.isdigit() for ch in cleaned) or "twelve" in cleaned


def test_inference_path_keeps_numbers_through_the_training_cleaner():
    cleaned = clean_normalised_text(normalise_input_text("The treasury paid 12,500 dollars in 1865."))
    assert "twelve thousand" in cleaned and "eighteen sixty-five" in cleaned
