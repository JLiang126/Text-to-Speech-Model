"""Spells out numbers, money and a few symbols before text reaches the phonemiser.

Why this exists: ``scripts/TextProcessor.clean_normalised_text`` deletes every
character that is not a letter or basic punctuation, so digits vanish
("paid 12,500 dollars in 1865" -> "paid , dollars in"). LJSpeech's normalised
transcripts already spell numbers out, so training is unaffected. The task, though,
asks for *arbitrary* input text, so the inference path (synth.py, evaluate.py)
expands numbers first.

The number rules are from keithito/tacotron ``text/numbers.py`` (MIT licence,
Copyright (c) 2017 Keith Ito), the same front-end family VITS descends from:
https://github.com/keithito/tacotron/blob/master/text/numbers.py
Adapted here: wrapped in ``normalise_input_text`` and extended with '%' and '&'.
"""
from __future__ import annotations

import re

import inflect

_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")


def _remove_commas(m: re.Match) -> str:
    return m.group(1).replace(",", "")


def _expand_decimal_point(m: re.Match) -> str:
    return m.group(1).replace(".", " point ")


def _expand_dollars(m: re.Match) -> str:
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    dollar_unit = "dollar" if dollars == 1 else "dollars"
    cent_unit = "cent" if cents == 1 else "cents"
    if dollars and cents:
        return f"{dollars} {dollar_unit}, {cents} {cent_unit}"
    if dollars:
        return f"{dollars} {dollar_unit}"
    if cents:
        return f"{cents} {cent_unit}"
    return "zero dollars"


def _expand_ordinal(m: re.Match) -> str:
    return _inflect.number_to_words(m.group(0))


def _expand_number(m: re.Match) -> str:
    num = int(m.group(0))
    if 1000 < num < 3000:  # read like a year: 1865 -> "eighteen sixty-five"
        if num == 2000:
            return "two thousand"
        if 2000 < num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        if num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    return _inflect.number_to_words(num, andword="")


def normalize_numbers(text: str) -> str:
    """keithito/tacotron number expansion (unchanged logic)."""
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


def normalise_input_text(text: str) -> str:
    """Arbitrary input text -> text with every digit and %/& spelled out."""
    text = text.replace("%", " percent").replace("&", " and ")
    text = normalize_numbers(text)
    return re.sub(r"\s+", " ", text).strip()
