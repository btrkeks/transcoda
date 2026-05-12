from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import export_fcmae_encoder as exporter
from src.pretraining.fcmae.config import FCMAEModelConfig, FCMAETrainingConfig


class FakeSavePretrainedEncoder:
    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text('{"model_type":"fake"}')


def test_export_refuses_non_empty_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "exported"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep")

    with pytest.raises(FileExistsError, match="not empty"):
        exporter.export_fcmae_encoder(
            checkpoint_path=tmp_path / "model.ckpt",
            output_dir=output_dir,
        )


def test_export_writes_encoder_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_module = SimpleNamespace(
        model=SimpleNamespace(encoder=FakeSavePretrainedEncoder()),
        model_config=FCMAEModelConfig(encoder_model_name_or_path="fake/source"),
        training_config=FCMAETrainingConfig(batch_size=4, max_steps=12),
    )
    monkeypatch.setattr(
        exporter.FCMAEPretrainer,
        "load_from_checkpoint",
        lambda *args, **kwargs: fake_module,
    )
    monkeypatch.setattr(exporter, "_collect_git_commit", lambda: "abc123")

    output_dir = exporter.export_fcmae_encoder(
        checkpoint_path=tmp_path / "model.ckpt",
        output_dir=tmp_path / "exported",
    )

    assert (output_dir / "config.json").exists()
    metadata = json.loads((output_dir / exporter.EXPORT_METADATA_FILENAME).read_text())
    assert metadata["source_checkpoint"] == str(tmp_path / "model.ckpt")
    assert metadata["source_encoder_model_name_or_path"] == "fake/source"
    assert metadata["pretraining_config"]["model"]["encoder_model_name_or_path"] == "fake/source"
    assert metadata["pretraining_config"]["training"]["batch_size"] == 4
    assert metadata["git_commit"] == "abc123"
    assert metadata["v1_implementation_commit"] == "57b2b0e"


def test_export_validation_defaults_match_downstream_canvas() -> None:
    assert exporter.export_fcmae_encoder.__kwdefaults__["validation_image_height"] == 1485
    assert exporter.export_fcmae_encoder.__kwdefaults__["validation_image_width"] == 1050
