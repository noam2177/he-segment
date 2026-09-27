"""Exact-match counts, plus an error tag when the cut is wrong."""

from __future__ import annotations

from collections import Counter

from he_segment.errors import TAGS, error_tag


def score_rows(rows: list[dict], predict) -> dict:
    exact = 0
    tags: Counter[str] = Counter()
    n = 0
    for row in rows:
        n += 1
        pred_p, pred_s = predict(str(row["token"]))
        tag = error_tag(row["prefixes"], str(row["stem"]), pred_p, pred_s)
        if tag is None:
            exact += 1
        else:
            tags[tag] += 1
    return {"n": n, "exact": exact, "exact_rate": round(exact / n, 3) if n else 0.0, "errors": dict(tags)}


def error_mix(rows: list[dict], predict) -> dict[str, int]:
    """How many frozen mistakes carry each tag. Exact rows are not in the mix."""
    errors = score_rows(rows, predict)["errors"]
    return {tag: int(errors.get(tag, 0)) for tag in TAGS}


def lexicon_only(rows: list[dict], lexicon_predict, other_predict) -> list[str]:
    """Tokens the lexicon cuts correctly and the other system does not."""
    wins: list[str] = []
    for row in rows:
        token = str(row["token"])
        gold_p = tuple(row["prefixes"])
        gold_s = str(row["stem"])
        lex_p, lex_s = lexicon_predict(token)
        other_p, other_s = other_predict(token)
        lexicon_ok = lex_p == gold_p and lex_s == gold_s
        other_ok = tuple(other_p) == gold_p and other_s == gold_s
        if lexicon_ok and not other_ok:
            wins.append(token)
    return wins
