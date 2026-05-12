"""Tests for pipeline module."""

from __future__ import annotations

import pytest

from scripts.dataset_generation.augmentation import AugmentationConfig, augment_transcription


class TestAugmentTranscription:
    def test_with_default_config(self, simple_kern: str) -> None:
        # Should not raise any errors
        result = augment_transcription(simple_kern)
        assert isinstance(result, str)

    def test_with_custom_config(self, simple_kern: str) -> None:
        config = AugmentationConfig(
            tempo=0.0,
            staccatos=0.0,
            sforzandos=0.0,
            fermatas=0.0,
            pedals=0.0,
            expression=0.0,
            xtuplet=0.0,
            invisible_rests=0.0,
            remove_beginning_time_signature=0.0,
            courtesy_naturals=1.0,
        )
        result = augment_transcription(simple_kern, config=config)

        assert result != simple_kern
        assert "n" in result

    def test_all_augmentations_disabled(self, simple_kern: str) -> None:
        config = AugmentationConfig(
            tempo=0.0,
            staccatos=0.0,
            sforzandos=0.0,
            fermatas=0.0,
            pedals=0.0,
            expression=0.0,
            xtuplet=0.0,
            invisible_rests=0.0,
            remove_beginning_time_signature=0.0,
            courtesy_naturals=0.0,
        )
        result = augment_transcription(simple_kern, config=config)

        # Should be unchanged (or close to it - articulations won't be added)
        # Note: even with 0 probability, the kern should still be valid
        assert "**kern" in result

    def test_empty_string(self, empty_kern: str) -> None:
        result = augment_transcription(empty_kern)
        assert result == ""

    def test_preserves_structure(self, simple_kern: str) -> None:
        config = AugmentationConfig(
            tempo=1.0,
            staccatos=1.0,
            sforzandos=0.0,
            fermatas=0.0,
            pedals=0.0,
            expression=0.0,
            xtuplet=0.0,
            invisible_rests=0.0,
            remove_beginning_time_signature=0.0,
            courtesy_naturals=0.0,
        )
        result = augment_transcription(simple_kern, config=config)

        # Should still have the basic structure
        assert "**kern" in result
        assert "*-" in result


class TestAugmentationConfig:
    def test_default_values(self) -> None:
        config = AugmentationConfig()

        assert config.tempo == 0.3
        assert config.staccatos == 0.5
        assert config.sforzandos == 0.3
        assert config.fermatas == 0.2
        assert config.pedals == 0.3
        assert config.expression == 0.3
        assert config.courtesy_naturals == 0.15

    def test_custom_values(self) -> None:
        config = AugmentationConfig(
            tempo=0.1,
            staccatos=0.2,
        )

        assert config.tempo == 0.1
        assert config.staccatos == 0.2

    def test_validation_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            AugmentationConfig(tempo=-0.1)

    def test_validation_rejects_above_one(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            AugmentationConfig(staccatos=1.5)
