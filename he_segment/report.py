"""Write fixtures/report.json from the three systems. Test rows only."""

from __future__ import annotations

import json
from pathlib import Path

from he_segment.baseline import segment
from he_segment.char_model import predict, train
from he_segment.lexicon import lexicon_segment, protected_from_train
from he_segment.score import score_rows

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "fixtures" / "gold.json"
OUT = ROOT / "fixtures" / "report.json"


def main() -> None:
    rows = json.loads(GOLD.read_text(encoding="utf-8"))
    for row in rows:
        row.setdefault("split", "train")
    test = [row for row in rows if row["split"] == "test"]
    protected = protected_from_train(rows)
    weights = train(rows)
    report = {
        "n_gold": len(rows),
        "n_test": len(test),
        "protected_train": len(protected),
        "baseline": score_rows(test, segment),
        "lexicon": score_rows(test, lambda token: lexicon_segment(token, protected)),
        "char_model": score_rows(test, lambda token: predict(token, weights)),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
