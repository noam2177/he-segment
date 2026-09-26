"""Naive proclitic cut: one prefix, then a following article. Not the finished segmenter."""

from __future__ import annotations

_PREFIXES = ("כש", "שב", "וה", "ה", "ו", "ב", "ל", "מ", "כ", "ש")


def segment(token: str) -> tuple[tuple[str, ...], str]:
    token = (token or "").strip()
    if len(token) < 3:
        return (), token
    matched = ""
    for pref in _PREFIXES:
        if token.startswith(pref) and len(token) - len(pref) >= 2:
            matched = pref
            break
    if not matched:
        return (), token
    rest = token[len(matched) :]
    prefixes = [matched]
    if "ה" not in matched and rest.startswith("ה") and len(rest) - 1 >= 2:
        prefixes.append("ה")
        rest = rest[1:]
    return tuple(prefixes), rest


def matches(token: str, prefixes: list[str], stem: str) -> bool:
    got_prefixes, got_stem = segment(token)
    return got_prefixes == tuple(prefixes) and got_stem == stem
