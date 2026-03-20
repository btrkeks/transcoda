"""Statistics tracking for filter pipeline execution."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FileCheckResult:
    """Result from checking a single file through all filters."""

    path_str: str  # String for clean pickling
    passed: bool
    rejecting_filter: str | None
    # filter_name -> (status_value, reason)
    filter_outcomes: dict[str, tuple[str, str | None]]

    @property
    def path(self) -> Path:
        return Path(self.path_str)


@dataclass
class FilterStats:
    """Statistics for a single filter's execution."""

    name: str
    files_checked: int = 0
    files_passed: int = 0
    files_failed: int = 0
    files_errored: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)

    def record_pass(self) -> None:
        """Record a file passing this filter."""
        self.files_checked += 1
        self.files_passed += 1

    def record_fail(self, reason: str | None) -> None:
        """Record a file failing this filter."""
        self.files_checked += 1
        self.files_failed += 1
        if reason:
            self.failure_reasons[reason] = self.failure_reasons.get(reason, 0) + 1

    def record_error(self, reason: str | None) -> None:
        """Record an error processing a file."""
        self.files_checked += 1
        self.files_errored += 1
        if reason:
            key = f"ERROR: {reason}"
            self.failure_reasons[key] = self.failure_reasons.get(key, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "files_checked": self.files_checked,
            "files_passed": self.files_passed,
            "files_failed": self.files_failed,
            "files_errored": self.files_errored,
            "failure_reasons": self.failure_reasons,
        }


@dataclass
class PipelineStats:
    """Statistics for a complete pipeline run."""

    total_input: int
    total_passed: int
    total_failed: int
    per_filter: list[FilterStats]
    duration_seconds: float
    rejection_by_filter: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_input": self.total_input,
            "total_passed": self.total_passed,
            "total_failed": self.total_failed,
            "pass_rate": self.total_passed / self.total_input if self.total_input > 0 else 0.0,
            "duration_seconds": self.duration_seconds,
            "rejection_by_filter": self.rejection_by_filter,
            "per_filter": [f.to_dict() for f in self.per_filter],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: Path) -> None:
        """Save statistics to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Total input: {self.total_input}",
            f"Passed: {self.total_passed} ({100 * self.total_passed / self.total_input:.1f}%)"
            if self.total_input > 0
            else "Passed: 0",
            f"Failed: {self.total_failed}",
            f"Duration: {self.duration_seconds:.1f}s",
            "",
            "Rejections by filter:",
        ]
        for name, count in sorted(self.rejection_by_filter.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {count}")
        return "\n".join(lines)
