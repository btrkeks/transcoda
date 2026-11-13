"""Callback that persists per-sample validation predictions to JSONL files.

Activated by ``TranscodaTrainer.set_validation_prediction_recording(True)``
(the entrypoint enables it for the automatic final validation and for
validate-only runs). For each recorded validation pass it writes, under
``<output_root>/<metric_prefix>_predictions/``:

- ``predictions.rank<r>.jsonl`` — one JSON object per sample with decoded
  prediction/target kern text and exact per-sample CER values
- ``manifest.rank<r>.json`` — prediction set display metadata, checkpoint path,
  step/epoch, and record counts

Files are per-rank because under DDP each rank validates its own shard;
consumers should glob ``predictions.rank*.jsonl`` and may need to dedupe by
(val_set, sample_id) since DistributedSampler can pad with duplicates.
"""

import json
from pathlib import Path
from typing import IO, Any

from lightning.pytorch import Callback, LightningModule, Trainer

from transcoda.metrics_schema import VAL_PREFIX
from transcoda.validation.records import decoded_batch_from_validation_outputs


def predictions_output_dir(output_root: str | Path, metric_prefix: str) -> Path:
    """Directory holding prediction dumps for a given validation metric prefix."""
    return Path(output_root) / f"{metric_prefix}_predictions"


class ValidationPredictionWriter(Callback):
    """Streams per-sample validation predictions to disk when recording is enabled.

    Args:
        output_root: Run directory to write under (checkpoint dirpath).
        i2w: Index-to-word mapping for decoding token IDs to kern text.
        pad_token_id: Padding token ID terminating decoded sequences.
    """

    def __init__(
        self,
        output_root: str | Path,
        i2w: dict[int, str],
        pad_token_id: int,
        *,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        prediction_set_label: str | None = None,
        prediction_set_slug: str | None = None,
    ) -> None:
        super().__init__()
        self.output_root = Path(output_root)
        self.i2w = i2w
        self.pad_token_id = pad_token_id
        self.bos_token_id = _resolve_special_token_id(i2w, "<bos>", bos_token_id)
        self.eos_token_id = _resolve_special_token_id(i2w, "<eos>", eos_token_id)
        self.prediction_set_label = _clean_label(prediction_set_label)
        self.prediction_set_slug = _clean_slug(
            prediction_set_slug
            or self.prediction_set_label
            or self.output_root.name
        )
        self._file: IO[str] | None = None
        self._output_dir: Path | None = None
        self._records_by_set: dict[str, int] = {}

    @staticmethod
    def _module_is_recording(pl_module: LightningModule) -> bool:
        checker = getattr(pl_module, "is_recording_validation_predictions", None)
        if callable(checker):
            return bool(checker())
        return False

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self._module_is_recording(pl_module):
            return

        metric_prefix = str(getattr(pl_module, "validation_metric_prefix", VAL_PREFIX))
        self._output_dir = predictions_output_dir(self.output_root, metric_prefix)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._records_by_set = {}
        path = self._output_dir / f"predictions.rank{trainer.global_rank}.jsonl"
        self._file = path.open("w", encoding="utf-8")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self._file is None or outputs is None:
            return

        decoded_batch = decoded_batch_from_validation_outputs(
            outputs,
            i2w=self.i2w,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )
        for sample in decoded_batch:
            self._file.write(sample.to_json_line() + "\n")
            self._records_by_set[sample.val_set] = (
                self._records_by_set.get(sample.val_set, 0) + 1
            )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._file is None:
            return

        self._file.close()
        self._file = None

        manifest = {
            "metric_prefix": str(getattr(pl_module, "validation_metric_prefix", VAL_PREFIX)),
            "prediction_set_slug": self.prediction_set_slug,
            "prediction_set_label": self.prediction_set_label or self.prediction_set_slug,
            "checkpoint_path": getattr(trainer, "ckpt_path", None),
            "global_step": int(getattr(trainer, "global_step", 0)),
            "epoch": int(getattr(trainer, "current_epoch", 0)),
            "global_rank": int(getattr(trainer, "global_rank", 0)),
            "world_size": int(getattr(trainer, "world_size", 1)),
            "records_by_set": self._records_by_set,
        }
        assert self._output_dir is not None
        manifest_path = self._output_dir / f"manifest.rank{manifest['global_rank']}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        self._output_dir = None
        self._records_by_set = {}


def _clean_label(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _resolve_special_token_id(
    i2w: dict[int, str],
    token_text: str,
    explicit_token_id: int | None,
) -> int | None:
    if explicit_token_id is not None:
        return int(explicit_token_id)
    for token_id, text in i2w.items():
        if text == token_text:
            return int(token_id)
    return None


def _clean_slug(value: str | None) -> str:
    if value is None:
        return "prediction-set"
    lowered = str(value).strip().lower()
    cleaned = "".join(
        char if char.isalnum() or char in {"_", "-"} else "-"
        for char in lowered
    )
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    cleaned = cleaned.strip("-_.")
    return cleaned or "prediction-set"
