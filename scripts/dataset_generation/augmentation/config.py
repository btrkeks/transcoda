"""Configuration for kern augmentation."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["AugmentationConfig"]


@dataclass(frozen=True)
class AugmentationConfig:
    """Probabilities for each augmentation type.

    Each probability is in the range [0.0, 1.0] and controls whether
    that augmentation type is applied at all. Individual augmentation
    functions have their own per-note/per-position probabilities.

    Attributes:
        tempo: Probability of adding tempo markings.
        staccatos: Probability of adding staccato articulations.
        sforzandos: Probability of adding sforzando accents.
        fermatas: Probability of adding fermata markings.
        pedals: Probability of adding pedal markings.
        expression: Probability of adding expression markings.
        xtuplet: Probability of adding *Xtuplet markers to spines with tuplets.
    invisible_rests: Probability of adding invisible rest markers (yy)
            to rests in multi-voice regions.
        courtesy_naturals: Probability of adding redundant courtesy natural
            signs to notes that are already natural under key + measure state.
    """

    tempo: float = 0.3
    staccatos: float = 0.5
    sforzandos: float = 0.3
    fermatas: float = 0.2
    pedals: float = 0.3
    expression: float = 0.3
    xtuplet: float = 0.3
    invisible_rests: float = 0.4
    remove_beginning_time_signature: float = 0.1
    courtesy_naturals: float = 0.15

    def __post_init__(self) -> None:
        for field_name in (
            "tempo",
            "staccatos",
            "sforzandos",
            "fermatas",
            "pedals",
            "expression",
            "xtuplet",
            "invisible_rests",
            "remove_beginning_time_signature",
            "courtesy_naturals",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0.0, 1.0], got {value}")
