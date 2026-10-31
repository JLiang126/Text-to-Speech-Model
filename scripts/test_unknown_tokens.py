import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from transformers import VitsTokenizer

from scripts.DataConfig import DataConfig
from scripts.DataLoader import LJSpeechDataset, build_dataloaders, build_splits
from scripts.TextProcessor import (
    TextProcessor,
    clean_normalised_text,
    clean_normalised_text_simple,
    load_vctk_tokenizer,
)


class UnknownTokenTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        symbols = {
            0: "_", 3: ",", 16: " ", 17: "A", 43: "a",
            174: "<unk>", 177: "\u1d7b",
        }
        vocab = {
            symbols.get(index, f"unused_{index}"): index
            for index in range(178)
        }
        vocab_path = self.directory / "vocab.json"
        vocab_path.write_text(json.dumps(vocab), encoding="utf-8")
        self.tokenizer = VitsTokenizer(
            vocab_file=str(vocab_path), pad_token="_", unk_token="<unk>",
            normalize=False, phonemize=False, add_blank=True,
        )
        self.cfg = DataConfig(
            output_dir=self.directory / "run", target_sr=16000,
            n_fft=16, win_length=16, hop_length=4, n_mels=4,
        )

    def make_dataset(self, texts):
        rows = [
            {"utt_id": f"utt-{index}", "clean_text": text}
            for index, text in enumerate(texts)
        ]
        return LJSpeechDataset(
            pd.DataFrame(rows), cfg=self.cfg, tokenizer=self.tokenizer,
        )

    def test_lazy_tokenization_rejects_unknown_ids_without_caching(self):
        dataset = self.make_dataset(["(a)"])
        with self.assertRaisesRegex(ValueError, "Unknown tokens") as error:
            dataset._get_token_ids(0)
        self.assertIn("utt-0", str(error.exception))
        self.assertIn("(a)", str(error.exception))
        self.assertIn("174", str(error.exception))
        self.assertNotIn(0, dataset._token_cache)

    def test_eager_tokenization_rejects_unknown_ids(self):
        dataset = self.make_dataset(["a", "(a)"])
        with self.assertRaisesRegex(ValueError, "utt-1"):
            dataset.preprocess_token_ids()
        self.assertEqual(set(dataset._token_cache), {0})

    def test_valid_stroke_i_is_preserved_and_cached(self):
        dataset = self.make_dataset(["a\u1d7b"])
        expected = torch.tensor([0, 43, 0, 177, 0], dtype=torch.long)
        torch.testing.assert_close(dataset._get_token_ids(0), expected)
        self.assertIs(dataset._get_token_ids(0), dataset._token_cache[0])

    def test_training_validation_and_test_preprocessing_reject_unknowns(self):
        split_names = ("train", "val", "test")
        self.cfg.preprocess_token_ids = True
        self.cfg.output_split_dir.mkdir(parents=True)
        for invalid_split in split_names:
            with self.subTest(split=invalid_split):
                for split in split_names:
                    pd.DataFrame([{
                        "utt_id": f"utt-{split}",
                        "norm_text": "a@" if split == invalid_split else "a",
                        "clean_text": "a@" if split == invalid_split else "a",
                    }]).to_csv(
                        self.cfg.output_split_dir / f"{split}_manifest.csv",
                        index=False,
                    )
                with self.assertRaisesRegex(ValueError, f"utt-{invalid_split}"):
                    build_dataloaders(
                        self.cfg, num_workers=0, tokenizer=self.tokenizer,
                    )

    def test_fallback_rejects_unknown_phonemes_instead_of_dropping_them(self):
        processor = TextProcessor(add_blank=False)
        with patch.object(processor, "phonemize_text", return_value="a@"):
            with self.assertRaisesRegex(ValueError, "Unknown phoneme") as error:
                processor.text_to_sequence("a", is_cleaned=True)
        self.assertIn("@", str(error.exception))

    def test_fallback_preserves_known_phonemes(self):
        processor = TextProcessor(add_blank=True)
        with patch.object(processor, "phonemize_text", return_value="a"):
            actual = processor.text_to_sequence("a", is_cleaned=True)
        expected = torch.tensor([0, processor.symbol_to_id["a"], 0])
        torch.testing.assert_close(actual, expected)

    def test_fallback_uses_simple_cleaner_for_uncleaned_input(self):
        processor = TextProcessor(add_blank=False)
        with patch.object(processor, "phonemize_text", return_value="a") as phonemize:
            processor.text_to_sequence("Cafe\u0301 [Room 12]")
        phonemize.assert_called_once_with("Caf\u00e9, Room 12,")

    def test_corrected_tokenizer_keeps_unknown_and_stroke_i_distinct_after_reload(self):
        self.tokenizer.save_pretrained(self.directory / "checkpoint")
        tokenizer = load_vctk_tokenizer(str(self.directory / "checkpoint"))
        self.assertEqual(tokenizer.unk_token_id, 174)
        self.assertEqual(tokenizer.convert_tokens_to_ids("\u1d7b"), 177)
        self.assertEqual(tokenizer.convert_ids_to_tokens(177), "\u1d7b")
        self.assertEqual(tokenizer.vocab_size, 178)


class VitsSymbolTests(unittest.TestCase):
    def test_hardcoded_vocabulary_has_unique_checkpoint_compatible_ids(self):
        processor = TextProcessor()
        self.assertEqual(processor.vocab_size, 178)
        self.assertEqual(len(processor.symbol_to_id), 178)
        self.assertEqual(
            processor.symbols[:17],
            ["_"] + list(';:,.!?\u00a1\u00bf\u2014\u2026"\u00ab\u00bb\u201c\u201d '),
        )
        self.assertEqual(
            processor.symbols[17:69],
            list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
        )
        self.assertEqual(
            processor.symbols[174:], ["<unk>", "\u0329", "'", "\u1d7b"],
        )
        self.assertEqual(processor.symbol_to_id["\u026a"], 102)
        for token_id, symbol in enumerate(processor.symbols):
            self.assertEqual(processor.symbol_to_id[symbol], token_id)
            self.assertEqual(processor.id_to_symbol[token_id], symbol)

    def test_fallback_emits_pretrained_ids_for_ipa_and_stroke_i(self):
        processor = TextProcessor(add_blank=True)
        with patch.object(processor, "phonemize_text", return_value="a\u026a\u1d7b"):
            actual = processor.text_to_sequence("a", is_cleaned=True)
        expected = torch.tensor([0, 43, 0, 102, 0, 177, 0], dtype=torch.long)
        torch.testing.assert_close(actual, expected)


class SimpleCleanerTests(unittest.TestCase):
    def test_bracket_commas_attach_to_previous_word(self):
        for text, expected in (
            ("First (aside) next", "First, aside, next"),
            ("First [aside] next", "First, aside, next"),
            ("First(aside)next", "First, aside, next"),
            ("First \t( aside )\nnext", "First, aside, next"),
            ("First ((nested)) next", "First, nested, next"),
            ("First (outer [inner]) next", "First, outer, inner, next"),
        ):
            with self.subTest(text=text):
                self.assertEqual(clean_normalised_text_simple(text), expected)

    def test_leading_brackets_do_not_add_leading_commas(self):
        for text, expected in (
            ("(Aside) next", "Aside, next"),
            (" [Aside] next", "Aside, next"),
            ("\n((Aside))", "Aside,"),
            ("()", ""),
            (" ( [ ] ) ", ""),
        ):
            with self.subTest(text=text):
                self.assertEqual(clean_normalised_text_simple(text), expected)

    def test_existing_punctuation_is_not_duplicated(self):
        for text, expected in (
            ("First, (aside) next", "First, aside, next"),
            ("First (aside), next", "First, aside, next"),
            ("First (aside).", "First, aside."),
            ("(Aside).", "Aside."),
            ("First: [aside]", "First: aside,"),
            ("First (aside!) next", "First, aside! next"),
            ("First () next", "First, next"),
        ):
            with self.subTest(text=text):
                self.assertEqual(clean_normalised_text_simple(text), expected)

    def test_unicode_case_digits_and_other_punctuation_are_preserved(self):
        text = "  Cafe\u0301\tRoom 12; A: B!  "
        self.assertEqual(
            clean_normalised_text_simple(text), "Caf\u00e9 Room 12; A: B!",
        )
        self.assertEqual(clean_normalised_text_simple("well\u2014known"), "well\u2014known")
        self.assertEqual(clean_normalised_text_simple(""), "")

    def test_legacy_cleaner_remains_available(self):
        self.assertEqual(clean_normalised_text("Room 12 caf\u00e9"), "room caf")

    def test_space_replacement_option_is_preserved(self):
        self.assertEqual(
            clean_normalised_text_simple(
                "First (aside) [next]", brackets_to_commas=False,
            ),
            "First aside next",
        )

    def test_new_split_manifests_use_simple_cleaner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            wavs = dataset_root / "wavs"
            wavs.mkdir(parents=True)
            with wave.open(str(wavs / "utt-0.wav"), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(bytes(32000))
            (dataset_root / "metadata.csv").write_text(
                "utt-0|Original|Cafe\u0301 (Room 12).\n", encoding="utf-8",
            )
            cfg = DataConfig(
                data_root=dataset_root, output_dir=root / "run",
                split_ratios=(1.0, 0.0, 0.0), lowercase=True,
            )
            train, _, _, _ = build_splits(cfg)
            self.assertEqual(train.iloc[0]["clean_text"], "Caf\u00e9, Room 12.")


if __name__ == "__main__":
    unittest.main()
