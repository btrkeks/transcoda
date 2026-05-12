"""Fixtures for kern augmentation tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def simple_kern() -> str:
    """A simple two-spine kern string for testing."""
    return """**kern\t**kern
*clefF4\t*clefG2
*k[]\t*k[]
*M4/4\t*M4/4
4c\t4e
4d\t4f
=\t=
4e\t4g
4f\t4a
=\t=
*-\t*-"""


@pytest.fixture
def kern_with_beams() -> str:
    """Kern string with beamed notes."""
    return """**kern\t**kern
*clefF4\t*clefG2
*M4/4\t*M4/4
8cL\t8eL
8dJ\t8fJ
4e\t4g
=\t=
*-\t*-"""


@pytest.fixture
def kern_with_ties() -> str:
    """Kern string with tied notes."""
    return """**kern\t**kern
*clefF4\t*clefG2
*M4/4\t*M4/4
[4c\t4e
4c]\t4f
=\t=
*-\t*-"""


@pytest.fixture
def empty_kern() -> str:
    """Empty kern string."""
    return ""
