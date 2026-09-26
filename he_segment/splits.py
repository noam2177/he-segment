"""Stable split. A token never moves once it has a split."""

from __future__ import annotations

import hashlib


def assign_split(token: str) -> str:
    bucket = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket < 2:
        return "test"
    if bucket < 3:
        return "dev"
    return "train"
