import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from he_segment.baseline import matches, segment


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = json.loads((ROOT / "fixtures" / "gold.json").read_text(encoding="utf-8"))

    def test_fifteen_rows(self) -> None:
        self.assertEqual(len(self.rows), 15)

    def test_article_after_mem(self) -> None:
        self.assertEqual(segment("מהעיר"), (("מ", "ה"), "עיר"))

    def test_baseline_is_not_done(self) -> None:
        hits = sum(1 for row in self.rows if matches(row["token"], row["prefixes"], row["stem"]))
        self.assertLess(hits, len(self.rows))
        self.assertGreaterEqual(hits, 8)
        missed = [row["token"] for row in self.rows if not matches(row["token"], row["prefixes"], row["stem"])]
        self.assertIn("מכתב", missed)
        self.assertIn("והילד", missed)


if __name__ == "__main__":
    unittest.main()
