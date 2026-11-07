from phonemizer import phonemize

from scripts.TextProcessor import configure_espeak


def test_bundled_espeak_ng_can_phonemize_text():
    configure_espeak()
    result = phonemize("This is a test.", language="en-us", backend="espeak", strip=True)
    assert result
