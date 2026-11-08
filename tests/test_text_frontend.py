"""Text front-end checks: token IDs must match the pretrained VCTK checkpoint.

The checkpoint's text-encoder embedding has one row per symbol, so a token whose ID differs from
the official VITS order reads the wrong row. The official strings are copied verbatim from
jaywalnut310/vits ``text/symbols.py`` (MIT). The Hugging Face VCTK checkpoint
(ylacombe/vits-vctk-with-discriminator) uses the same order, except that row 174, a duplicate
apostrophe in the official table, holds ``<unk>`` (see ``load_vctk_tokenizer`` in
scripts/TextProcessor.py).
"""
from pathlib import Path

import pytest

from scripts.TextProcessor import clean_normalised_text, load_vctk_tokenizer
from scripts.evaluation.text_normalise import normalise_input_text

CHECKPOINT = "ylacombe/vits-vctk-with-discriminator"
OFFICIAL_PAD = '_'
OFFICIAL_PUNCTUATION = ';:,.!?¡¿—…"«»“” '
OFFICIAL_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
OFFICIAL_LETTERS_IPA = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
OFFICIAL_SYMBOLS = [OFFICIAL_PAD] + list(OFFICIAL_PUNCTUATION) + list(OFFICIAL_LETTERS) + list(OFFICIAL_LETTERS_IPA)
UNK_ID = 174
VCTK_SYMBOLS = OFFICIAL_SYMBOLS[:UNK_ID] + ["<unk>"] + OFFICIAL_SYMBOLS[UNK_ID + 1:]  # the checkpoint's row order


@pytest.fixture(scope="module")
def vctk_tokenizer():
    """The tokenizer training uses, read from the local Hugging Face cache (no download)."""
    from huggingface_hub import try_to_load_from_cache

    vocab = try_to_load_from_cache(CHECKPOINT, "vocab.json")
    if not isinstance(vocab, str):
        pytest.skip(f"{CHECKPOINT} is not in the local Hugging Face cache")
    return load_vctk_tokenizer(str(Path(vocab).parent))


def test_token_ids_match_the_checkpoint_rows(vctk_tokenizer):
    assert len(OFFICIAL_SYMBOLS) == 178
    wrong = [(i, s) for i, s in enumerate(VCTK_SYMBOLS) if vctk_tokenizer.encoder.get(s) != i]
    assert wrong == [], f"token IDs differ from the checkpoint's rows at {wrong}"
    assert vctk_tokenizer.decoder[UNK_ID] == "<unk>" and vctk_tokenizer.encoder[" "] == 16


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
