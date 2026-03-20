"""Pipeline for composing and executing file filters."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .base import FileFilter, FilterContext, FilterStatus
from .stats import FileCheckResult, FilterStats, PipelineStats


def _check_file_worker(
    path_str: str,
    filter_configs: list[tuple[str, dict]],
    fail_fast: bool,
) -> FileCheckResult:
    """
    Check a single file against all filters.

    Must be module-level for pickling in multiprocessing.

    Args:
        path_str: Path to file as string (for clean pickling)
        filter_configs: List of (filter_name, params) tuples
        fail_fast: If True, stop checking after first failure

    Returns:
        FileCheckResult with outcomes for all checked filters
    """
    from .filters import FILTER_REGISTRY

    path = Path(path_str)
    ctx = FilterContext()

    # Reconstruct filters from config (avoids pickling filter instances)
    filters = [FILTER_REGISTRY[name](**params) for name, params in filter_configs]

    passed_all = True
    rejecting_filter = None
    filter_outcomes: dict[str, tuple[str, str | None]] = {}

    for f in filters:
        result = f.check(path, ctx)
        filter_outcomes[f.name] = (result.status.value, result.reason)

        if result.status != FilterStatus.PASSED:
            passed_all = False
            if rejecting_filter is None:
                rejecting_filter = f.name
            if fail_fast:
                break

    return FileCheckResult(
        path_str=path_str,
        passed=passed_all,
        rejecting_filter=rejecting_filter,
        filter_outcomes=filter_outcomes,
    )


class FilterPipeline:
    """
    Composable pipeline for filtering files through multiple filters.

    Each file is checked against filters in sequence. By default, the pipeline
    uses fail-fast behavior: a file is rejected as soon as any filter fails.

    Example:
        from filtering import FilterPipeline
        from filtering.filters import UTF8Filter, TerminationFilter

        pipeline = FilterPipeline([UTF8Filter(), TerminationFilter()])
        passing_files = list(pipeline.filter_files(paths))
        print(pipeline.stats.summary())

    Example with stats:
        pipeline = FilterPipeline([UTF8Filter(), RhythmFilter()])
        for path in pipeline.filter_files(kern_files):
            process(path)
        pipeline.stats.save_json(Path("stats.json"))
    """

    def __init__(self, filters: Iterable[FileFilter], fail_fast: bool = True):
        """
        Initialize pipeline with a sequence of filters.

        Args:
            filters: Ordered sequence of filters to apply
            fail_fast: If True, stop checking a file after first failure
        """
        self.filters: list[FileFilter] = list(filters)
        self.fail_fast = fail_fast
        self._stats: PipelineStats | None = None
        self._filter_stats: dict[str, FilterStats] = {}

    def filter_files(self, paths: Iterable[Path]) -> Iterator[Path]:
        """
        Yield paths that pass all filters, tracking statistics.

        Args:
            paths: Iterable of file paths to filter

        Yields:
            Paths that pass all filters
        """
        start_time = time.monotonic()

        # Initialize per-filter stats
        self._filter_stats = {f.name: FilterStats(name=f.name) for f in self.filters}
        rejection_by_filter: dict[str, int] = {}
        total_input = 0
        total_passed = 0
        total_failed = 0

        for path in paths:
            total_input += 1
            ctx = FilterContext()
            passed_all = True
            rejecting_filter: str | None = None

            for filter_obj in self.filters:
                result = filter_obj.check(path, ctx)
                stats = self._filter_stats[filter_obj.name]

                if result.status == FilterStatus.PASSED:
                    stats.record_pass()
                elif result.status == FilterStatus.FAILED:
                    stats.record_fail(result.reason)
                    passed_all = False
                    if rejecting_filter is None:
                        rejecting_filter = filter_obj.name
                    if self.fail_fast:
                        break
                else:  # ERROR
                    stats.record_error(result.reason)
                    passed_all = False
                    if rejecting_filter is None:
                        rejecting_filter = filter_obj.name
                    if self.fail_fast:
                        break

            if passed_all:
                total_passed += 1
                yield path
            else:
                total_failed += 1
                if rejecting_filter:
                    rejection_by_filter[rejecting_filter] = (
                        rejection_by_filter.get(rejecting_filter, 0) + 1
                    )

        duration = time.monotonic() - start_time

        self._stats = PipelineStats(
            total_input=total_input,
            total_passed=total_passed,
            total_failed=total_failed,
            per_filter=list(self._filter_stats.values()),
            duration_seconds=duration,
            rejection_by_filter=rejection_by_filter,
        )

    @property
    def stats(self) -> PipelineStats:
        """
        Access statistics from last run.

        Raises:
            RuntimeError: If filter_files has not been called yet
        """
        if self._stats is None:
            raise RuntimeError("No statistics available. Run filter_files first.")
        return self._stats

    def get_filter_names(self) -> list[str]:
        """Get the names of all filters in execution order."""
        return [f.name for f in self.filters]

    def _get_filter_configs(self) -> list[tuple[str, dict]]:
        """
        Extract serializable config from filter instances.

        Returns:
            List of (filter_name, params) tuples for reconstructing filters in workers.
        """
        configs = []
        for f in self.filters:
            name = f.name
            params: dict = {}

            # Handle filters with constructor params
            if name == "rhythm":
                # Access RhythmFilter-specific attributes (safe at runtime)
                params = {
                    "binary_path": str(f.binary_path),  # type: ignore[attr-defined]
                    "allow_anacrusis": f.allow_anacrusis,  # type: ignore[attr-defined]
                    "allow_incomplete_final": f.allow_incomplete_final,  # type: ignore[attr-defined]
                }
            elif name == "accidentals":
                params = {
                    "max_consecutive": f.max_consecutive,  # type: ignore[attr-defined]
                }
            # Other filters have no constructor params

            configs.append((name, params))
        return configs

    def filter_files_parallel(
        self, paths: Iterable[Path], workers: int
    ) -> Iterator[Path]:
        """
        Filter files in parallel using ProcessPoolExecutor.

        Args:
            paths: Iterable of file paths to filter
            workers: Number of parallel workers

        Yields:
            Paths that pass all filters
        """
        start_time = time.monotonic()

        # Convert to list for length and reuse
        path_list = list(paths)
        filter_configs = self._get_filter_configs()

        # Initialize per-filter stats
        self._filter_stats = {f.name: FilterStats(name=f.name) for f in self.filters}
        rejection_by_filter: dict[str, int] = {}
        total_input = len(path_list)
        total_passed = 0
        total_failed = 0

        # Collect passing paths to yield after executor closes
        passing_paths: list[Path] = []

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _check_file_worker, str(path), filter_configs, self.fail_fast
                ): path
                for path in path_list
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    # Worker crashed - count as failure
                    total_failed += 1
                    # Attribute to first filter
                    if self.filters:
                        first_filter = self.filters[0].name
                        self._filter_stats[first_filter].record_error(str(e))
                        rejection_by_filter[first_filter] = (
                            rejection_by_filter.get(first_filter, 0) + 1
                        )
                    continue

                # Process result
                for filter_name, (status_value, reason) in result.filter_outcomes.items():
                    stats = self._filter_stats[filter_name]
                    if status_value == "passed":
                        stats.record_pass()
                    elif status_value == "failed":
                        stats.record_fail(reason)
                    else:  # error
                        stats.record_error(reason)

                if result.passed:
                    total_passed += 1
                    passing_paths.append(result.path)
                else:
                    total_failed += 1
                    if result.rejecting_filter:
                        rejection_by_filter[result.rejecting_filter] = (
                            rejection_by_filter.get(result.rejecting_filter, 0) + 1
                        )

        duration = time.monotonic() - start_time

        self._stats = PipelineStats(
            total_input=total_input,
            total_passed=total_passed,
            total_failed=total_failed,
            per_filter=list(self._filter_stats.values()),
            duration_seconds=duration,
            rejection_by_filter=rejection_by_filter,
        )

        # Yield passing paths after all processing is done
        yield from passing_paths

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        filter_registry: dict[str, type[FileFilter]] | None = None,
        **kwargs,
    ) -> "FilterPipeline":
        """
        Load a pipeline from a JSON configuration file.

        Config format:
            {
                "filters": [
                    {"name": "utf8"},
                    {"name": "termination"},
                    {"name": "rhythm", "params": {"allow_anacrusis": true}}
                ],
                "fail_fast": true
            }

        Args:
            config_path: Path to JSON configuration file
            filter_registry: Mapping of filter names to filter classes.
                            If None, uses default registry from filtering.filters
            **kwargs: Additional arguments passed to FilterPipeline.__init__

        Returns:
            Configured FilterPipeline instance

        Raises:
            ValueError: If config is invalid or filter names are unknown
        """
        if filter_registry is None:
            from . import filters as filters_module

            filter_registry = filters_module.FILTER_REGISTRY

        config_path = Path(config_path)
        with open(config_path) as f:
            config = json.load(f)

        if "filters" not in config:
            raise ValueError("Config must contain 'filters' key")

        filter_instances = []
        for filter_config in config["filters"]:
            name = filter_config["name"]
            params = filter_config.get("params", {})

            if name not in filter_registry:
                raise ValueError(
                    f"Unknown filter: {name}. Available: {list(filter_registry.keys())}"
                )

            filter_class = filter_registry[name]
            filter_instances.append(filter_class(**params))

        fail_fast = config.get("fail_fast", kwargs.pop("fail_fast", True))
        return cls(filter_instances, fail_fast=fail_fast, **kwargs)
