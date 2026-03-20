"""Base utilities for data generators."""

from collections.abc import Iterator
from concurrent.futures import TimeoutError
from dataclasses import dataclass

from pebble import ProcessExpired


@dataclass
class GenerationStats:
    """Statistics from a generation run."""

    successful: int = 0
    overflows: int = 0
    errors: int = 0
    timeouts: int = 0
    expired_workers: int = 0
    invalid: int = 0
    rejected_sparse: int = 0
    rejected_render_fit: int = 0

    @property
    def total_failures(self) -> int:
        """Total number of failed samples."""
        return (
            self.overflows
            + self.errors
            + self.timeouts
            + self.expired_workers
            + self.invalid
            + self.rejected_sparse
            + self.rejected_render_fit
        )

    @property
    def has_failures(self) -> bool:
        """Whether any failures occurred."""
        return self.total_failures > 0


def iterate_results(
    results: Iterator,
    counters: dict[str, int],
) -> Iterator[tuple[bytes, str, str] | tuple[None, str, str]]:
    """Iterate over pool results with unified exception handling.

    Handles TimeoutError (Verovio hangs) and ProcessExpired (worker crashes
    from OOM, segfault in Verovio/libpng, etc.) gracefully, logging and
    counting the failures instead of crashing the entire generation.

    Args:
        results: Iterator from pebble ProcessPool.map().result().
        counters: Dict with 'timeout' and 'expired' keys for tracking failures.

    Yields:
        Result tuples from successful worker tasks.
    """
    while True:
        try:
            yield next(results)
        except StopIteration:
            break
        except TimeoutError:
            counters["timeout"] += 1
        except ProcessExpired:
            counters["expired"] += 1
