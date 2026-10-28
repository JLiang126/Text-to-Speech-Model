"""Text front-end checks: the symbol table must match the pretrained VITS checkpoint.

The VCTK checkpoint's text-encoder embedding has one row per symbol, in the order of
the official table. If our table differs (an extra, missing or reordered symbol),
every token ID points at the wrong embedding row and a frozen text encoder can never
recover. These strings are copied verbatim from jaywalnut310/vits ``text/symbols.py`` (MIT).
"""
import pytest

from scripts.TextProcessor import VITS_SYMBOLS, TextProcessor, clean_normalised_text

OFFICIAL_PAD = '_'
OFFICIAL_PUNCTUATION = ';:,.!?¡¿—…"«»“” '
OFFICIAL_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
OFFICIAL_LETTERS_IPA = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
OFFICIAL_SYMBOLS = [OFFICIAL_PAD] + list(OFFICIAL_PUNCTUATION) + list(OFFICIAL_LETTERS) + list(OFFICIAL_LETTERS_IPA)


def test_official_table_has_178_symbols():
    assert len(OFFICIAL_SYMBOLS) == 178


def test_symbol_table_matches_official_vits():
    ours = list(VITS_SYMBOLS)
    first_diff = next((i for i, (a, b) in enumerate(zip(ours, OFFICIAL_SYMBOLS)) if a != b), None)
    assert ours == OFFICIAL_SYMBOLS, (
        f"TextProcessor has {len(ours)} symbols, the checkpoint expects {len(OFFICIAL_SYMBOLS)}; "
        f"first difference at index {first_diff}: ours={ours[first_diff]!r} "
        f"official={OFFICIAL_SYMBOLS[first_diff]!r}")


def test_space_token_id_matches_official():
    assert TextProcessor().symbol_to_id[" "] == OFFICIAL_SYMBOLS.index(" ") == 16


def test_blank_tokens_are_interspersed():
    ids = TextProcessor(add_blank=True).text_to_sequence("hello world")
    assert len(ids) % 2 == 1
    assert bool((ids[0::2] == 0).all())


@pytest.mark.xfail(strict=True, reason="Known issue: clean_normalised_text deletes digits, so numbers in "
                                       "arbitrary input text are silently dropped. Expand numbers (e.g. "
                                       "with inflect) before cleaning, then remove this marker.")
def test_numbers_survive_text_cleaning():
    cleaned = clean_normalised_text("The treasury paid 12,500 dollars in 1865.")
    assert any(ch.isdigit() for ch in cleaned) or "twelve" in cleaned
