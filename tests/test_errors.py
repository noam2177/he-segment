import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from he_segment.baseline import segment
from he_segment.errors import error_tag


class ErrorTagTests(unittest.TestCase):
    def test_match_is_empty(self) -> None:
        self.assertIsNone(error_tag(["ה"], "משרד", ["ה"], "משרד"))

    def test_false_friend(self) -> None:
        pred_p, pred_s = segment("מכתב")
        self.assertEqual(error_tag([], "מכתב", pred_p, pred_s), "false_friend")

    def test_compound_vav_he(self) -> None:
        pred_p, pred_s = segment("והילד")
        self.assertEqual(error_tag(["ו", "ה"], "ילד", pred_p, pred_s), "compound")

    def test_under_strip_leaves_article(self) -> None:
        self.assertEqual(error_tag(["מ", "ה"], "עיר", ["מ"], "העיר"), "under_strip")

    def test_over_strip_eats_stem(self) -> None:
        self.assertEqual(error_tag(["ה"], "משרד", ["ה", "מ"], "שרד"), "over_strip")

    def test_when_he_arrived_keeps_the_verb(self) -> None:
        rows = json.loads((ROOT / "fixtures" / "gold.json").read_text(encoding="utf-8"))
        by_token = {row["token"]: row for row in rows}
        self.assertEqual(by_token["כשהגיע"]["prefixes"], ["כש"])
        self.assertEqual(by_token["כשהגיע"]["stem"], "הגיע")

    def test_known_rows_get_a_known_tag(self) -> None:
        rows = json.loads((ROOT / "fixtures" / "gold.json").read_text(encoding="utf-8"))
        for row in rows:
            pred_p, pred_s = segment(row["token"])
            tag = error_tag(row["prefixes"], row["stem"], pred_p, pred_s)
            if tag is not None:
                self.assertIn(tag, {"over_strip", "under_strip", "compound", "false_friend"})


if __name__ == "__main__":
    unittest.main()
