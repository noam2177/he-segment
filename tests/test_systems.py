import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from he_segment.baseline import segment
from he_segment.char_model import predict, train
from he_segment.lexicon import lexicon_segment, protected_from_train
from he_segment.score import error_mix, lexicon_only, mix_share, score_rows
from he_segment.splits import assign_split


class SystemTests(unittest.TestCase):
    def test_split_is_stable_and_one_of_three(self) -> None:
        self.assertEqual(assign_split("מכתב"), assign_split("מכתב"))
        self.assertIn(assign_split("מכתב"), {"train", "dev", "test"})

    def test_lexicon_uses_train_only(self) -> None:
        rows = [
            {"token": "מכתב", "prefixes": [], "stem": "מכתב", "split": "train"},
            {"token": "שלום", "prefixes": [], "stem": "שלום", "split": "test"},
        ]
        protected = protected_from_train(rows)
        self.assertIn("מכתב", protected)
        self.assertNotIn("שלום", protected)
        self.assertEqual(lexicon_segment("מכתב", protected), ((), "מכתב"))

    def test_char_model_learns_a_train_cut(self) -> None:
        rows = [
            {"token": "מהעיר", "prefixes": ["מ", "ה"], "stem": "עיר", "split": "train"},
            {"token": "מהבית", "prefixes": ["מ", "ה"], "stem": "בית", "split": "train"},
            {"token": "לעבודה", "prefixes": ["ל"], "stem": "עבודה", "split": "train"},
            {"token": "בבית", "prefixes": ["ב"], "stem": "בית", "split": "train"},
            {"token": "המשרד", "prefixes": ["ה"], "stem": "משרד", "split": "test"},
        ]
        weights = train(rows, epochs=12)
        self.assertEqual(predict("מהעיר", weights), (("מ", "ה"), "עיר"))
        train_tokens = {row["token"] for row in rows if row["split"] == "train"}
        self.assertNotIn("המשרד", train_tokens)

    def test_gold_file_concatenates(self) -> None:
        rows = json.loads((ROOT / "fixtures" / "gold.json").read_text(encoding="utf-8"))
        seen: set[str] = set()
        for row in rows:
            self.assertEqual("".join(row["prefixes"]) + row["stem"], row["token"])
            self.assertNotIn(row["token"], seen)
            seen.add(row["token"])

    def test_frozen_scores_stay(self) -> None:
        gold = json.loads((ROOT / "fixtures" / "gold.json").read_text(encoding="utf-8"))
        for row in gold:
            row.setdefault("split", "train")
        frozen = json.loads((ROOT / "fixtures" / "frozen_test.json").read_text(encoding="utf-8"))
        protected = protected_from_train(gold)
        weights = train(gold)
        self.assertEqual(score_rows(frozen, segment)["exact"], 36)
        self.assertEqual(
            score_rows(frozen, lambda token: lexicon_segment(token, protected))["exact"],
            36,
        )
        self.assertEqual(score_rows(frozen, lambda token: predict(token, weights))["exact"], 64)
        wins = lexicon_only(
            frozen,
            lambda token: lexicon_segment(token, protected),
            lambda token: predict(token, weights),
        )
        self.assertEqual(wins, ["שישן", "שנפל", "שנכנס", "שיחק", "בתיק"])
        char_score = score_rows(frozen, lambda token: predict(token, weights))
        mix = error_mix(frozen, lambda token: predict(token, weights))
        self.assertEqual(sum(mix.values()) + char_score["exact"], len(frozen))
        self.assertEqual(char_score["exact"], 64)
        share = mix_share(frozen, lambda token: predict(token, weights))
        self.assertAlmostEqual(sum(share.values()), 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
