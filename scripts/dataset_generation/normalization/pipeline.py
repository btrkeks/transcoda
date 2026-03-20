"""Pipeline for composing and executing normalization passes."""

import json
from collections.abc import Iterable
from pathlib import Path

from .base import NormalizationContext, NormalizationPass


class Pipeline:
    """
    Composable pipeline for executing normalization passes in sequence.

    Each pass follows a three-phase lifecycle (prepare → transform → validate)
    and can share data through a common context.

    Example:
        from normalization import Pipeline
        from normalization.passes import OrderNotes, DedupeAccidentals

        pipeline = Pipeline([OrderNotes(), DedupeAccidentals()])
        normalized = pipeline("4c 4e 4g")

    Example with context inspection:
        pipeline = Pipeline([OrderNotes()])
        ctx = NormalizationContext()
        result = pipeline("4c 4e 4g", ctx=ctx)
        print(ctx.get("order_notes"))  # Access pass-specific data
    """

    def __init__(self, passes: Iterable[NormalizationPass]):
        """
        Initialize pipeline with a sequence of passes.

        Args:
            passes: Ordered sequence of normalization passes to apply
        """
        self.passes: list[NormalizationPass] = list(passes)

    def __call__(self, text: str, ctx: NormalizationContext | None = None) -> str:
        """
        Execute all passes in sequence on the input text.

        Args:
            text: The kern string to normalize
            ctx: Optional context to use (creates new one if not provided)

        Returns:
            The fully normalized kern string

        Raises:
            ValueError: If any pass fails validation or encounters an error
        """
        if ctx is None:
            ctx = NormalizationContext()

        for pass_obj in self.passes:
            pass_obj.prepare(text, ctx)
            text = pass_obj.transform(text, ctx)
            pass_obj.validate(text, ctx)

        return text

    def add_pass(self, pass_obj: NormalizationPass) -> None:
        """
        Add a pass to the end of the pipeline.

        Args:
            pass_obj: The normalization pass to add
        """
        self.passes.append(pass_obj)

    def remove_pass(self, name: str) -> None:
        """
        Remove a pass by name.

        Args:
            name: The name of the pass to remove

        Raises:
            ValueError: If no pass with the given name exists
        """
        for i, pass_obj in enumerate(self.passes):
            if pass_obj.name == name:
                self.passes.pop(i)
                return
        raise ValueError(f"No pass named '{name}' found in pipeline")

    def get_pass_names(self) -> list[str]:
        """
        Get the names of all passes in execution order.

        Returns:
            List of pass names
        """
        return [p.name for p in self.passes]

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        pass_registry: dict[str, type[NormalizationPass]] | None = None,
    ) -> "Pipeline":
        """
        Load a pipeline from a JSON configuration file.

        Config format:
            {
                "passes": [
                    {"name": "order_notes", "params": {...}},
                    {"name": "dedupe_accidentals", "params": {...}}
                ]
            }

        Args:
            config_path: Path to JSON configuration file
            pass_registry: Mapping of pass names to pass classes.
                          If None, uses default registry from normalization.passes

        Returns:
            Configured Pipeline instance

        Raises:
            ValueError: If config is invalid or pass names are unknown
        """
        if pass_registry is None:
            from . import passes as passes_module

            pass_registry = passes_module.PASS_REGISTRY

        config_path = Path(config_path)
        with open(config_path) as f:
            config = json.load(f)

        if "passes" not in config:
            raise ValueError("Config must contain 'passes' key")

        pass_instances = []
        for pass_config in config["passes"]:
            name = pass_config["name"]
            params = pass_config.get("params", {})

            if name not in pass_registry:
                raise ValueError(f"Unknown pass: {name}. Available: {list(pass_registry.keys())}")

            pass_class = pass_registry[name]
            pass_instances.append(pass_class(**params))

        return cls(pass_instances)

    def to_config(self, config_path: str | Path) -> None:
        """
        Save pipeline configuration to a JSON file.

        Args:
            config_path: Path where to save the configuration
        """
        config = {"passes": []}

        for pass_obj in self.passes:
            pass_config = {"name": pass_obj.name}

            # Extract params for known passes
            if pass_obj.name == "order_notes":
                pass_config["params"] = {"ascending": pass_obj.ascending}

            config["passes"].append(pass_config)

        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
