"""Base classes and protocols for kern file filtering."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class FilterStatus(Enum):
    """Result status for a filter check."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class FilterResult:
    """Result of a filter check on a file."""

    status: FilterStatus
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def pass_(cls) -> "FilterResult":
        """Create a passing result."""
        return cls(status=FilterStatus.PASSED)

    @classmethod
    def fail(cls, reason: str, **details: Any) -> "FilterResult":
        """Create a failing result with a reason."""
        return cls(status=FilterStatus.FAILED, reason=reason, details=details)

    @classmethod
    def error(cls, reason: str, **details: Any) -> "FilterResult":
        """Create an error result."""
        return cls(status=FilterStatus.ERROR, reason=reason, details=details)

    @property
    def passed(self) -> bool:
        """Check if the result indicates the file passed."""
        return self.status == FilterStatus.PASSED


class FilterContext(dict[str, Any]):
    """
    Shared context for filter passes, with file content caching.

    Each filter can store data keyed by its name or a custom key.
    File content is cached on first read to avoid redundant I/O.

    Example:
        ctx = FilterContext()
        content = ctx.get_content(path)  # Reads and caches
        content = ctx.get_content(path)  # Returns cached value
    """

    _CONTENT_KEY = "_file_content"
    _ERROR_KEY = "_file_error"

    def get_content(self, path: Path) -> str | None:
        """
        Get file content, caching on first read.

        Args:
            path: Path to the file to read

        Returns:
            File content as string, or None if read fails (UTF-8 error, missing file, etc.)
        """
        if self._CONTENT_KEY in self:
            return self[self._CONTENT_KEY]

        try:
            content = path.read_text(encoding="utf-8")
            self[self._CONTENT_KEY] = content
            return content
        except UnicodeDecodeError as e:
            self[self._ERROR_KEY] = f"UTF-8 decode error: {e}"
            self[self._CONTENT_KEY] = None
            return None
        except FileNotFoundError:
            self[self._ERROR_KEY] = "File not found"
            self[self._CONTENT_KEY] = None
            return None
        except OSError as e:
            self[self._ERROR_KEY] = f"OS error: {e}"
            self[self._CONTENT_KEY] = None
            return None

    def get_content_error(self) -> str | None:
        """Get the error message if content reading failed."""
        return self.get(self._ERROR_KEY)


class FileFilter(Protocol):
    """
    Protocol defining the interface for a file filter.

    Each filter implements a single `check` method that determines
    whether a file passes the filter criteria.

    Attributes:
        name: Unique identifier for this filter

    Example:
        class MyFilter:
            name = "my_filter"

            def check(self, path: Path, ctx: FilterContext) -> FilterResult:
                content = ctx.get_content(path)
                if content is None:
                    return FilterResult.fail("Could not read file")
                if "required_string" not in content:
                    return FilterResult.fail("Missing required string")
                return FilterResult.pass_()
    """

    name: str

    def check(self, path: Path, ctx: FilterContext) -> FilterResult:
        """
        Check if a file passes this filter.

        Args:
            path: Path to the file to check
            ctx: Shared context for caching and cross-filter data

        Returns:
            FilterResult indicating pass, fail, or error
        """
        ...
