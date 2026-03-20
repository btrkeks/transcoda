"""Pipeline orchestrator for kern augmentation."""

from __future__ import annotations

import random

from .config import AugmentationConfig
from .courtesy_naturals import apply_courtesy_naturals
from .invisible_rests import apply_invisible_rests
from .remove_beginning_time_signature import apply_remove_beginning_time_signature
from .xtuplet import apply_xtuplet

__all__ = ["augment_transcription"]


def augment_transcription(
    krn: str,
    config: AugmentationConfig | None = None,
) -> str:
    """Apply random augmentations to a kern transcription.

    Each augmentation type is applied based on its probability in the config.
    When an augmentation is applied, it affects a subset of eligible positions
    based on the function's internal per-position probability.

    Args:
        krn: The kern transcription string to augment.
        config: Configuration controlling augmentation probabilities.
                Defaults to AugmentationConfig() if not provided.

    Returns:
        The augmented kern string.
    """
    if not krn:
        return krn

    if config is None:
        config = AugmentationConfig()

    # Apply each augmentation type based on its probability
    # if random.random() < config.tempo:
    #     krn = apply_tempo_markings(krn)

    # if random.random() < config.staccatos:
    #     krn = apply_staccatos(krn)

    # if random.random() < config.sforzandos:
    #     krn = apply_sforzandos(krn)

    # if random.random() < config.fermatas:
    #     krn = apply_fermatas(krn)

    # if random.random() < config.pedals:
    #     krn = apply_pedaling(krn)

    # if random.random() < config.expression:
    #     krn = apply_expression_markings(krn)

    if random.random() < config.xtuplet:
        krn = apply_xtuplet(krn)

    if random.random() < config.invisible_rests:
        krn = apply_invisible_rests(krn)

    if random.random() < config.remove_beginning_time_signature:
        krn = apply_remove_beginning_time_signature(krn)

    if random.random() < config.courtesy_naturals:
        krn = apply_courtesy_naturals(krn)

    return krn
