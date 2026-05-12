"""Header clef declaration validation filter."""

from pathlib import Path

from ..base import FilterContext, FilterResult


class HeaderClefFilter:
    """
    Filter that checks if all **kern spines have header clef declarations.

    Files may contain non-kern spines (e.g. **dynam, **text). Those are
    ignored by this filter because normalization removes them later.
    Only **kern/**ekern spines must have *clef declarations before note data
    or barlines appear.
    """

    name = "header_clef"

    @staticmethod
    def _is_kern_spine_token(token: str) -> bool:
        # Accept versioned variants like **kern_1.0 / **ekern_1.0.
        return token.startswith("**kern") or token.startswith("**ekern")

    def check(self, path: Path, ctx: FilterContext) -> FilterResult:
        content = ctx.get_content(path)

        if content is None:
            return FilterResult.fail("Could not read file content")

        text = content.rstrip("\n")
        if not text:
            return FilterResult.fail("Empty file")

        lines = text.split("\n")

        # Find an exclusive interpretation line containing at least one
        # **kern/**ekern spine.
        spine_line_idx = -1
        num_spines = 0
        kern_indices: list[int] = []
        for i, line in enumerate(lines):
            tokens = line.split("\t")
            if not tokens or not all(t.startswith("**") for t in tokens):
                continue
            current_kern_indices = [j for j, token in enumerate(tokens) if self._is_kern_spine_token(token)]
            if current_kern_indices:
                spine_line_idx = i
                num_spines = len(tokens)
                kern_indices = current_kern_indices
                break

        if spine_line_idx < 0:
            return FilterResult.fail("No spine declaration found")

        clef_found: dict[int, bool] = {idx: False for idx in kern_indices}

        # Scan lines after spine declaration
        for line in lines[spine_line_idx + 1 :]:
            tokens = line.split("\t")

            if line.startswith("!"):
                continue

            # Update clef coverage for **kern columns only.
            for idx in kern_indices:
                if idx < len(tokens) and tokens[idx].startswith("*clef"):
                    clef_found[idx] = True
            if all(clef_found.values()):
                return FilterResult.pass_()

            # Hit note/bar data in a kern spine before all required clefs found.
            data_seen = False
            for idx in kern_indices:
                if idx >= len(tokens):
                    continue
                token = tokens[idx]
                if token.startswith("*") or token.startswith("!"):
                    continue
                data_seen = True
                break

            if not data_seen:
                continue

            missing = [idx for idx, found in clef_found.items() if not found]
            return FilterResult.fail(
                "Missing header clef declaration",
                missing_spines=missing,
                num_spines=num_spines,
            )

        # Reached end of file without finding clefs (or data)
        missing = [idx for idx, found in clef_found.items() if not found]
        if missing:
            return FilterResult.fail(
                "Missing header clef declaration",
                missing_spines=missing,
                num_spines=num_spines,
            )

        return FilterResult.pass_()
