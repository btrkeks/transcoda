"""String-based metrics for fair cross-tokenizer comparison.

These metrics operate on decoded **kern strings rather than token IDs,
enabling fair comparison between models with different tokenizers.
"""

# Try to import RapidFuzz for fast C++ Levenshtein, fallback to pure Python
try:
    from rapidfuzz.distance import Levenshtein as _RFLev

    _levenshtein = _RFLev.distance
except ImportError:
    import warnings

    warnings.warn(
        "RapidFuzz not available. Falling back to pure-Python Levenshtein implementation. "
        "Install rapidfuzz for significantly better performance: pip install rapidfuzz",
        RuntimeWarning,
        stacklevel=2,
    )

    def _levenshtein(a: list, b: list) -> int:
        """Computes the Levenshtein distance between sequences a and b."""
        n, m = len(a), len(b)
        if n > m:
            a, b, n, m = b, a, m, n
        current = list(range(n + 1))
        for i in range(1, m + 1):
            previous, current = current, [i] + [0] * n
            for j in range(1, n + 1):
                add, delete = previous[j] + 1, current[j - 1] + 1
                change = previous[j - 1] + (a[j - 1] != b[i - 1])
                current[j] = min(add, delete, change)
        return current[n]


def compute_cer(pred: str, target: str) -> float:
    """Compute Character Error Rate on decoded strings.

    CER measures the edit distance at the character level, normalized
    by the target length. This provides fine-grained accuracy measurement.

    Args:
        pred: Predicted **kern string
        target: Ground truth **kern string

    Returns:
        Character Error Rate as percentage (0-100+)
    """
    if not target:
        return 0.0

    pred_chars = list(pred)
    target_chars = list(target)

    edit_dist = _levenshtein(pred_chars, target_chars)
    return 100.0 * edit_dist / len(target_chars)


def compute_ser(pred: str, target: str) -> float:
    """Compute Symbol Error Rate on decoded strings.

    SER measures edit distance at the token/symbol level, where each
    whitespace-separated token is treated as a unit. This aligns with
    how **kern notation structures musical symbols.

    Args:
        pred: Predicted **kern string
        target: Ground truth **kern string

    Returns:
        Symbol Error Rate as percentage (0-100+)
    """
    # Split by any whitespace (spaces, tabs, newlines) to get kern tokens
    pred_tokens = pred.split()
    target_tokens = target.split()

    if not target_tokens:
        return 0.0

    edit_dist = _levenshtein(pred_tokens, target_tokens)
    return 100.0 * edit_dist / len(target_tokens)


def compute_ler(pred: str, target: str) -> float:
    """Compute Line Error Rate on decoded strings.

    LER measures edit distance at the line level, treating each newline-
    separated line as a unit. This captures structural accuracy of the
    **kern document format.

    Args:
        pred: Predicted **kern string
        target: Ground truth **kern string

    Returns:
        Line Error Rate as percentage (0-100+)
    """
    pred_lines = pred.strip().split("\n")
    target_lines = target.strip().split("\n")

    if not target_lines:
        return 0.0

    edit_dist = _levenshtein(pred_lines, target_lines)
    return 100.0 * edit_dist / len(target_lines)
