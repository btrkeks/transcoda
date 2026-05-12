"""Rhythm validation filter using external rhythm_checker binary."""

import json
import subprocess
from pathlib import Path

from ..base import FilterContext, FilterResult


class RhythmFilter:
    """
    Filter that validates rhythm correctness using the rhythm_checker binary.

    This filter invokes an external Rust binary that parses kern files and
    validates that rhythmic durations sum correctly within measures.
    """

    name = "rhythm"

    def __init__(
        self,
        binary_path: str | Path = "./binaries/rhythm_checker/target/release/rhythm_checker",
        allow_anacrusis: bool = True,
        allow_incomplete_final: bool = True,
    ):
        """
        Initialize the rhythm filter.

        Args:
            binary_path: Path to the rhythm_checker binary
            allow_anacrusis: Allow incomplete first measure (pickup measure)
            allow_incomplete_final: Allow incomplete final measure
        """
        self.binary_path = Path(binary_path)
        self.allow_anacrusis = allow_anacrusis
        self.allow_incomplete_final = allow_incomplete_final

    def check(self, path: Path, ctx: FilterContext) -> FilterResult:
        """
        Check if a file has correct rhythm using the rhythm_checker binary.

        Args:
            path: Path to the file to check
            ctx: Filter context (not used by this filter)

        Returns:
            FilterResult indicating pass or fail
        """
        if not self.binary_path.exists():
            return FilterResult.error(
                f"rhythm_checker binary not found at {self.binary_path}"
            )

        cmd = [
            str(self.binary_path),
            str(path),
            "--format",
            "json",
        ]
        if self.allow_anacrusis:
            cmd.append("--allow-anacrusis")
        if self.allow_incomplete_final:
            cmd.append("--allow-incomplete-final")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return FilterResult.error("rhythm_checker timed out")
        except OSError as e:
            return FilterResult.error(f"Failed to run rhythm_checker: {e}")

        # Return code 0 = all files valid, 1 = some files have errors
        if result.returncode not in (0, 1):
            return FilterResult.error(
                f"rhythm_checker failed with code {result.returncode}",
                stderr=result.stderr,
            )

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            return FilterResult.error(f"Failed to parse rhythm_checker output: {e}")

        # Check if this file has errors
        file_results = data.get("file_results", [])
        if file_results:
            # There are errors for this file
            errors = file_results[0].get("errors", [])
            if errors:
                first_error = errors[0]
                return FilterResult.fail(
                    first_error.get("message", "Rhythm error"),
                    error_count=len(errors),
                    first_error=first_error,
                )

        return FilterResult.pass_()
