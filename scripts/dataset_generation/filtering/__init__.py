"""
Kern file filtering system.

This package provides a composable pipeline architecture for filtering
kern files based on various criteria. Filters can be composed, reordered,
and configured to accept or reject files in a predictable, maintainable way.

Quick Start:
    from filtering import FilterPipeline
    from filtering.filters import UTF8Filter, TerminationFilter

    # Create a pipeline with ordered filters
    pipeline = FilterPipeline([UTF8Filter(), TerminationFilter()])

    # Filter files
    for path in pipeline.filter_files(paths):
        process(path)

    # Access statistics
    print(pipeline.stats.summary())

Configuration-based usage:
    from filtering import FilterPipeline

    # Load pipeline from JSON config
    pipeline = FilterPipeline.from_config("config/filtering/default.json")
    passing = list(pipeline.filter_files(paths))

Implementing a new filter:
    from filtering.base import FilterContext, FilterResult

    class MyFilter:
        name = "my_filter"

        def check(self, path: Path, ctx: FilterContext) -> FilterResult:
            content = ctx.get_content(path)
            if content is None:
                return FilterResult.fail("Could not read file")
            if "required" not in content:
                return FilterResult.fail("Missing required content")
            return FilterResult.pass_()
"""

from .base import FileFilter, FilterContext, FilterResult, FilterStatus
from .pipeline import FilterPipeline
from .stats import FilterStats, PipelineStats

__all__ = [
    "FileFilter",
    "FilterContext",
    "FilterResult",
    "FilterStatus",
    "FilterPipeline",
    "FilterStats",
    "PipelineStats",
]

__version__ = "0.1.0"
