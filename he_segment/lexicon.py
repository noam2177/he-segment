"""Lexicon system: train-only tokens that must not be cut."""

from __future__ import annotations

from he_segment.baseline import segment


def protected_from_train(rows: list[dict]) -> frozenset[str]:
    kept: set[str] = set()
    for row in rows:
        if row.get("split") != "train":
            continue
        if row.get("prefixes"):
            continue
        kept.add(str(row["token"]))
    return frozenset(kept)


def lexicon_segment(token: str, protected: frozenset[str]) -> tuple[tuple[str, ...], str]:
    if token in protected:
        return (), token
    return segment(token)
