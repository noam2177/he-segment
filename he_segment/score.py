"""Exact-match counts, plus an error tag when the cut is wrong."""

from __future__ import annotations

from collections import Counter

from he_segment.errors import error_tag


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
