"""Stable synthetic row identity without a corpus or database dependency."""
import hashlib
import json
from enum import StrEnum

class EffectiveLabelTier(StrEnum):
    GOLD = "gold"
    SILVER = "silver"
    SYNTHETIC = "synthetic"

def content_digest(value: object) -> str:
    payload = json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()
