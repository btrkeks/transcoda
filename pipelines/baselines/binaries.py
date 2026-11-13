"""Explicit, inspectable resolution for external baseline tools."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class BinaryResolutionError(FileNotFoundError):
    """No executable candidate could be resolved."""


@dataclass(frozen=True)
class BinaryResolution:
    path: Path
    source: str
    tried: tuple[str, ...]


def binary_version_output(path: Path) -> str:
    """Return bounded best-effort version/help output without treating it as identity."""

    try:
        result = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"[:1000]
    combined = (result.stdout + result.stderr).decode("utf-8", errors="replace").strip()
    return combined[:1000] or f"exit {result.returncode} with no version output"


def resolve_binary(
    binary_name: str,
    *,
    explicit: Path | None = None,
    env_var: str,
) -> BinaryResolution:
    """Resolve explicit path, environment override, then PATH in that order."""

    tried: list[str] = []
    if explicit is not None:
        tried.append(f"explicit={explicit}")
        resolved = _executable(explicit)
        if resolved is not None:
            return BinaryResolution(resolved, "explicit", tuple(tried))

    environment_value = os.environ.get(env_var)
    tried.append(f"{env_var}={environment_value or '<unset>'}")
    if environment_value:
        resolved = _executable(Path(environment_value))
        if resolved is not None:
            return BinaryResolution(resolved, f"environment:{env_var}", tuple(tried))

    path_match = shutil.which(binary_name)
    tried.append(f"PATH={path_match or '<not found>'}")
    if path_match:
        return BinaryResolution(Path(path_match).resolve(), "PATH", tuple(tried))
    raise BinaryResolutionError(
        f"could not resolve {binary_name}; tried " + "; ".join(tried)
    )


def _executable(path: Path) -> Path | None:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError:
        return None
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        return None
    return resolved
