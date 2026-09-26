import json
import unittest
from pathlib import Path

from he_segment.engine import frozen_rows, normalize_text, stem_token

ROOT = Path(__file__).resolve().parents[1]


class EngineTests(unittest.TestCase):
    def test_frozen_file_does_not_overlap_train(self) -> None:
        gold = json.loads((ROOT / "fixtures" / "v1_gold.json").read_text(encoding="utf-8"))
        frozen = {row["token"] for row in frozen_rows()}
        train = {row["token"] for row in gold if row.get("split") == "train"}
        self.assertGreaterEqual(len(frozen), 40)
        self.assertTrue(frozen.isdisjoint(train))

    def test_punctuation_stays(self) -> None:
        out = normalize_text("שלום, עולם.")
        self.assertIn(",", out)
        self.assertTrue(out.endswith("."))

    def test_v1_cases(self) -> None:
        cases = json.loads((ROOT / "fixtures" / "v1_cases.json").read_text(encoding="utf-8"))
        by_kind = {row["kind"] for row in cases}
        self.assertIn("help", by_kind)
        self.assertIn("limit", by_kind)
        self.assertEqual(stem_token("והמחשב"), "מחשב")
        self.assertEqual(stem_token("מכתב"), "מכתב")
        self.assertIn("מחשב", normalize_text("קנינו והמחשב החדש."))
        self.assertEqual(stem_token("כשהגיע"), "גיע")

    def test_stem_is_a_suffix_of_the_token(self) -> None:
        for token in ("המשרד", "מכתב", "מהבית"):
            stem = stem_token(token)
            self.assertTrue(token.endswith(stem), token)
