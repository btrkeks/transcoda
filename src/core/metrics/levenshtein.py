"""Levenshtein distance utilities for metrics computation."""

import warnings

# Try to import RapidFuzz for fast C++ Levenshtein, fallback to pure Python
try:
    from rapidfuzz.distance import Levenshtein as _RFLev

    levenshtein = _RFLev.distance  # C++ fast path (works on lists of ints or strings)
except Exception:
    # Fallback to pure-Python implementation if RapidFuzz unavailable
    warnings.warn(
        "RapidFuzz not available. Falling back to pure-Python Levenshtein implementation. "
        "Install rapidfuzz for significantly better performance: pip install rapidfuzz",
        RuntimeWarning,
        stacklevel=2,
    )

    def levenshtein(a, b):
        """Compute the Levenshtein distance between sequences a and b."""
        n, m = len(a), len(b)
        if n > m:
            a, b, n, m = b, a, m, n
        current = range(n + 1)
        for i in range(1, m + 1):
            previous, current = current, [i] + [0] * n
            for j in range(1, n + 1):
                add, delete = previous[j] + 1, current[j - 1] + 1
                change = previous[j - 1] + (a[j - 1] != b[i - 1])
                current[j] = min(add, delete, change)
        return current[n]
