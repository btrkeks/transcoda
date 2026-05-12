"""Base classes and protocols for kern normalization passes."""

from typing import Any, Protocol


class NormalizationContext(dict[str, Any]):
    """
    Scratchpad for cross-pass data sharing.

    Each pass can store data keyed by its name or a custom key.
    This allows passes to communicate and share parsed state.

    Example:
        ctx = NormalizationContext()
        ctx["parser"] = {"tokens": [...], "metadata": {...}}
        ctx["order_notes"] = {"changes_made": 5}
    """

    pass


class NormalizationPass(Protocol):
    """
    Protocol defining the interface for a normalization pass.

    Each pass implements a three-phase lifecycle:
    1. prepare: Parse/analyze input (optional, can be no-op)
    2. transform: Modify the kern string
    3. validate: Check output is valid (optional, can be no-op)

    Attributes:
        name: Unique identifier for this pass

    Example:
        class MyPass:
            name = "my_pass"

            def prepare(self, text: str, ctx: NormalizationContext) -> None:
                # Parse or analyze text, store in ctx if needed
                pass

            def transform(self, text: str, ctx: NormalizationContext) -> str:
                # Perform transformation
                return modified_text

            def validate(self, text: str, ctx: NormalizationContext) -> None:
                # Raise exception if output is invalid
                pass
    """

    name: str

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """
        Prepare for transformation (parsing, analysis, etc.).

        Args:
            text: The kern string to analyze
            ctx: Shared context for storing parsed data

        Raises:
            ValueError: If input is malformed and cannot be processed
        """
        ...

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """
        Transform the kern string.

        Args:
            text: The kern string to transform
            ctx: Shared context with data from prepare phase

        Returns:
            The transformed kern string
        """
        ...

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """
        Validate the transformed output.

        Args:
            text: The transformed kern string
            ctx: Shared context

        Raises:
            ValueError: If output is invalid
        """
        ...
