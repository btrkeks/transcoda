"""Public sample identity contract."""
import re

def validate_sample_id(sample_id: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", sample_id) is None:
        raise ValueError(f"Invalid sample ID: {sample_id}")
