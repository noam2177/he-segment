"""Compare a predicted cut to the gold cut. One tag, or None when they match."""

from __future__ import annotations

TAGS = ("over_strip", "under_strip", "compound", "false_friend")


def _joined(prefixes: tuple[str, ...] | list[str]) -> str:
    return "".join(prefixes)


def error_tag(
    gold_prefixes: list[str] | tuple[str, ...],
    gold_stem: str,
    pred_prefixes: list[str] | tuple[str, ...],
    pred_stem: str,
) -> str | None:
    gold_p = tuple(gold_prefixes)
    pred_p = tuple(pred_prefixes)
    if pred_p == gold_p and pred_stem == gold_stem:
        return None
    gold_joined = _joined(gold_p)
    pred_joined = _joined(pred_p)
    if gold_joined == pred_joined and gold_p != pred_p:
        return "compound"
    if not gold_p and pred_p:
        return "false_friend"
    if len(pred_stem) < len(gold_stem):
        return "over_strip"
    if len(pred_stem) > len(gold_stem):
        return "under_strip"
    if len(pred_p) > len(gold_p):
        return "over_strip"
    return "under_strip"
