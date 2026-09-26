"""Frozen v1. Trains on the snapshot, scores the locked test, normalizes text."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from he_segment.char_model import predict, train

ROOT = Path(__file__).resolve().parents[1]
V1_GOLD = ROOT / "fixtures" / "v1_gold.json"
FROZEN_TEST = ROOT / "fixtures" / "frozen_test.json"


def _read(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _model() -> tuple[dict[str, float], frozenset[str]]:
    rows = _read(V1_GOLD)
    frozen = frozenset(row["token"] for row in _read(FROZEN_TEST))
    train_rows = [row for row in rows if row.get("split") == "train" and row["token"] not in frozen]
    return train(train_rows), frozen


def stem_token(token: str) -> str:
    weights, _frozen = _model()
    _prefixes, stem = predict(token, weights)
    return stem


def normalize_text(text: str) -> str:
    parts: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            parts.append(stem_token("".join(buf)))
            buf.clear()

    for ch in text:
        if "\u0590" <= ch <= "\u05FF":
            buf.append(ch)
        else:
            flush()
            parts.append(ch)
    flush()
    return "".join(parts)


def frozen_rows() -> list[dict]:
    return _read(FROZEN_TEST)


def train_row_count() -> int:
    _weights, frozen = _model()
    rows = _read(V1_GOLD)
    return sum(1 for row in rows if row.get("split") == "train" and row["token"] not in frozen)
