"""Character features over a closed set of cuts. Trained on train rows only."""

from __future__ import annotations

from he_segment.baseline import segment

_ONE = ("כש", "שב", "ה", "ו", "ב", "ל", "מ", "כ", "ש")


def candidates(token: str) -> list[tuple[tuple[str, ...], str]]:
    found: list[tuple[tuple[str, ...], str]] = []

    def add(prefixes: tuple[str, ...], stem: str) -> None:
        if "".join(prefixes) + stem != token:
            return
        if prefixes and len(stem) < 2:
            return
        item = (prefixes, stem)
        if item not in found:
            found.append(item)

    add((), token)
    add(*segment(token))
    if token.startswith("וה") and len(token) > 3:
        add(("ו", "ה"), token[2:])
    for pref in _ONE:
        if token.startswith(pref) and len(token) - len(pref) >= 2:
            rest = token[len(pref) :]
            add((pref,), rest)
            if "ה" not in pref and rest.startswith("ה") and len(rest) >= 3:
                add((pref, "ה"), rest[1:])
    return found


def _features(token: str, prefixes: tuple[str, ...], stem: str) -> dict[str, float]:
    joined = "".join(prefixes) or "NONE"
    feats = {
        "bias": 1.0,
        f"pref:{joined}": 1.0,
        f"n:{len(prefixes)}": 1.0,
        f"slen:{min(len(stem), 6)}": 1.0,
        f"t2:{token[:2]}": 1.0,
    }
    if stem:
        feats[f"s0:{stem[0]}"] = 1.0
    return feats


def _dot(weights: dict[str, float], feats: dict[str, float]) -> float:
    return sum(weights.get(key, 0.0) * value for key, value in feats.items())


def train(rows: list[dict], *, epochs: int = 8) -> dict[str, float]:
    weights: dict[str, float] = {}
    usable = [row for row in rows if row.get("split") == "train"]
    for _ in range(epochs):
        for row in usable:
            token = str(row["token"])
            gold = (tuple(row["prefixes"]), str(row["stem"]))
            options = candidates(token)
            if gold not in options:
                continue
            scored = sorted(options, key=lambda item: _dot(weights, _features(token, item[0], item[1])), reverse=True)
            pred = scored[0]
            if pred == gold:
                continue
            for key, value in _features(token, gold[0], gold[1]).items():
                weights[key] = weights.get(key, 0.0) + value
            for key, value in _features(token, pred[0], pred[1]).items():
                weights[key] = weights.get(key, 0.0) - value
    return weights


def predict(token: str, weights: dict[str, float]) -> tuple[tuple[str, ...], str]:
    options = candidates(token)
    return max(options, key=lambda item: _dot(weights, _features(token, item[0], item[1])))
