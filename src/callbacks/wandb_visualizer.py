"""Callback for logging validation examples to Weights & Biases.

This callback retains the K best and K worst validation samples (by CER) in
bounded heaps during the validation pass, then builds a W&B table at epoch end.
No re-inference is needed — predictions are captured during validation.
"""

import heapq
from dataclasses import dataclass, field
from typing import Any

import torch
import wandb
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger

from src.callbacks.wandb_visualizer_rows import ValidationTableRowFormatter
from src.core.diff_utils import generate_html_diff


@dataclass(order=False)
class _HeapEntry:
    """A heap-comparable entry holding sample data.

    Only ``sort_key`` participates in comparisons so that non-comparable
    fields (tensors, lists) never cause errors.
    """

    sort_key: tuple = field(compare=True)
    image: torch.Tensor = field(compare=False, repr=False)
    pred_ids: list[int] = field(compare=False, repr=False)
    gt_ids: list[int] = field(compare=False, repr=False)
    cer: float = field(compare=False)
    cer_no_ties_beams: float | None = field(compare=False)
    val_set_name: str = field(compare=False)
    sample_id: int = field(compare=False)
    source_name: str | None = field(default=None, compare=False)

    # Make the entry comparable via sort_key only
    def __lt__(self, other: "_HeapEntry") -> bool:
        return self.sort_key < other.sort_key

    def __le__(self, other: "_HeapEntry") -> bool:
        return self.sort_key <= other.sort_key

    def __gt__(self, other: "_HeapEntry") -> bool:
        return self.sort_key > other.sort_key

    def __ge__(self, other: "_HeapEntry") -> bool:
        return self.sort_key >= other.sort_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _HeapEntry):
            return NotImplemented
        return self.sort_key == other.sort_key


class WandbVisualizerCallback(Callback):
    """Logs best/worst validation examples to Weights & Biases.

    During validation, bounded heaps retain only the K best and K worst
    samples by CER.  At epoch end, a W&B table is built directly from the
    heap contents — no re-inference required.

    Args:
        pad_token_id: Padding token ID for the tokenizer
        bos_token_id: Beginning-of-sequence token ID
        eos_token_id: End-of-sequence token ID
        i2w: Index-to-word mapping dictionary (token ID -> string)
        n_best: Number of best (lowest CER) examples to log (default: 5)
        n_worst: Number of worst (highest CER) examples to log (default: 20)
    """

    def __init__(
        self,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        i2w: dict[int, str],
        n_best: int = 5,
        n_worst: int = 20,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.i2w = i2w
        self.n_best = n_best
        self.n_worst = n_worst
        self._row_formatter = ValidationTableRowFormatter(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            i2w=i2w,
        )
        self._best_heaps: dict[str, list[_HeapEntry]] = {}
        self._worst_heaps: dict[str, list[_HeapEntry]] = {}

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Push per-sample data into bounded best/worst heaps.

        Expects ``outputs`` to contain:
        - val_set_name: str
        - sample_ids: tensor of dataset indices
        - cers: list[float]
        - cers_no_ties_beams: Optional[list[float]]
        - pred_ids: list[list[int]]
        - gt_ids: list[list[int]]
        Uses ``batch["pixel_values"]`` to capture images only for samples retained
        in best/worst heaps.
        """
        if not self._module_should_log_examples(pl_module):
            return

        if trainer.logger is None or not isinstance(trainer.logger, WandbLogger):
            return

        if outputs is None:
            return

        val_set_name = outputs["val_set_name"]
        sample_ids = outputs["sample_ids"].tolist()
        sources = outputs.get("sources")
        cers = outputs["cers"]
        cers_no_ties_beams = outputs.get("cers_no_ties_beams")
        pred_ids = outputs["pred_ids"]
        gt_ids = outputs["gt_ids"]

        # Validate that all per-sample fields have matching lengths
        n = len(sample_ids)
        if not (len(cers) == n and len(pred_ids) == n and len(gt_ids) == n):
            raise ValueError(
                f"All per-sample fields must have matching lengths. Got "
                f"sample_ids={n}, cers={len(cers)}, pred_ids={len(pred_ids)}, "
                f"gt_ids={len(gt_ids)}."
            )
        if cers_no_ties_beams is not None and len(cers_no_ties_beams) != n:
            raise ValueError(
                "All per-sample fields must have matching lengths. "
                f"Got sample_ids={n}, cers_no_ties_beams={len(cers_no_ties_beams)}."
            )
        if sources is not None and len(sources) != n:
            raise ValueError(
                "All per-sample fields must have matching lengths. "
                f"Got sample_ids={n}, sources={len(sources)}."
            )

        name = str(val_set_name)
        for i in range(n):
            cer = float(cers[i])
            cer_no_ties_beams = (
                float(cers_no_ties_beams[i]) if cers_no_ties_beams is not None else None
            )
            sid = int(sample_ids[i])
            source_name = str(sources[i]) if sources is not None and sources[i] is not None else None
            best_key = (-cer, name, sid)
            worst_key = (cer, name, sid)
            best_heap = self._best_heaps.get(name)
            worst_heap = self._worst_heaps.get(name)
            best_accept = self.n_best > 0 and (
                best_heap is None
                or len(best_heap) < self.n_best
                or best_key > best_heap[0].sort_key
            )
            worst_accept = self.n_worst > 0 and (
                worst_heap is None
                or len(worst_heap) < self.n_worst
                or worst_key > worst_heap[0].sort_key
            )

            # Avoid copying images for samples that are not retained in either heap.
            if not (best_accept or worst_accept):
                continue

            if not isinstance(batch, dict) or "pixel_values" not in batch:
                raise ValueError("Validation batch must contain pixel_values for example image logging.")

            # Clone to avoid retaining references to larger batch tensors.
            image = batch["pixel_values"][i].detach().to(dtype=torch.float16).cpu().clone()

            # --- Best heap (keep lowest CER) ---
            # sort_key=(-cer, name, id): min-heap root is the *highest* CER
            # in the set (the eviction candidate).
            if best_accept:
                if best_heap is None:
                    best_heap = []
                    self._best_heaps[name] = best_heap
                best_entry = _HeapEntry(
                    sort_key=best_key,
                    image=image,
                    pred_ids=pred_ids[i],
                    gt_ids=gt_ids[i],
                    cer=cer,
                    cer_no_ties_beams=cer_no_ties_beams,
                    val_set_name=name,
                    sample_id=sid,
                    source_name=source_name,
                )
                if len(best_heap) < self.n_best:
                    heapq.heappush(best_heap, best_entry)
                else:
                    # new sample has lower CER → evict highest-CER entry
                    heapq.heapreplace(best_heap, best_entry)

            # --- Worst heap (keep highest CER) ---
            # sort_key=(cer, name, id): min-heap root is the *lowest* CER
            # in the set (the eviction candidate).
            if worst_accept:
                if worst_heap is None:
                    worst_heap = []
                    self._worst_heaps[name] = worst_heap
                worst_entry = _HeapEntry(
                    sort_key=worst_key,
                    image=image,
                    pred_ids=pred_ids[i],
                    gt_ids=gt_ids[i],
                    cer=cer,
                    cer_no_ties_beams=cer_no_ties_beams,
                    val_set_name=name,
                    sample_id=sid,
                    source_name=source_name,
                )
                if len(worst_heap) < self.n_worst:
                    heapq.heappush(worst_heap, worst_entry)
                else:
                    # new sample has higher CER → evict lowest-CER entry
                    heapq.heapreplace(worst_heap, worst_entry)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Build and log a W&B table from the heap contents, then clear heaps."""
        # Under DDP, heaps are populated independently on each rank from that rank's
        # validation shard. Log only from rank 0 (best/worst examples will be drawn
        # from rank 0's shard, which is representative but not globally optimal).
        if not trainer.is_global_zero:
            self._clear_heaps()
            return

        if not self._should_log_examples(trainer, pl_module):
            self._clear_heaps()
            return

        validation_table = self._build_table_from_heaps(current_epoch=pl_module.current_epoch)
        trainer.logger.experiment.log({"Validation Table": validation_table})
        self._clear_heaps()

    def _should_log_examples(self, trainer: Trainer, pl_module: LightningModule) -> bool:
        """Check if validation examples should be logged to W&B."""
        return (
            self._module_should_log_examples(pl_module)
            and trainer.logger is not None
            and isinstance(trainer.logger, WandbLogger)
            and (any(self._best_heaps.values()) or any(self._worst_heaps.values()))
        )

    @staticmethod
    def _module_should_log_examples(pl_module: LightningModule) -> bool:
        checker = getattr(pl_module, "should_log_validation_examples", None)
        if callable(checker):
            return bool(checker())
        return bool(pl_module.hparams.training.log_example_images)

    def _clear_heaps(self) -> None:
        """Clear all per-dataset best/worst heaps."""
        self._best_heaps.clear()
        self._worst_heaps.clear()

    def _build_table_from_heaps(self, current_epoch: int) -> wandb.Table:
        """Create a W&B table from the current best/worst heap contents."""
        columns = [
            "ID",
            "Category",
            "Ground Truth Image",
            "Ground Truth",
            "Prediction",
            "Prediction Rendered Image",
            "SER",
            "CER",
            "CER_no_ties_beams",
            "Diff",
        ]

        rows: list[list] = []

        set_names = sorted(set(self._best_heaps.keys()) | set(self._worst_heaps.keys()))
        for set_name in set_names:
            # Best entries sorted by CER ascending
            best_entries = sorted(
                self._best_heaps.get(set_name, []), key=lambda e: (e.cer, e.val_set_name, e.sample_id)
            )
            for entry in best_entries:
                rows.append(self._entry_to_row(entry, "best", current_epoch))

            # Worst entries sorted by CER descending
            worst_entries = sorted(
                self._worst_heaps.get(set_name, []),
                key=lambda e: (-e.cer, e.val_set_name, e.sample_id),
            )
            for entry in worst_entries:
                rows.append(self._entry_to_row(entry, "worst", current_epoch))

        return wandb.Table(columns=columns, data=rows)

    def _entry_to_row(self, entry: _HeapEntry, category: str, current_epoch: int) -> list:
        """Convert a heap entry into a table row."""
        formatted = self._row_formatter.format_entry(
            entry,
            category=category,
            current_epoch=current_epoch,
            diff_builder=generate_html_diff,
        )

        return [
            formatted.row_id,
            formatted.category,
            wandb.Image(formatted.image),
            formatted.ground_truth,
            formatted.prediction,
            (
                wandb.Image(formatted.prediction_rendered_image)
                if formatted.prediction_rendered_image is not None
                else None
            ),
            f"{formatted.ser:.4f}",
            f"{formatted.cer:.4f}",
            (
                f"{formatted.cer_no_ties_beams:.4f}"
                if formatted.cer_no_ties_beams is not None
                else None
            ),
            wandb.Html(formatted.diff_html),
        ]

    def _token_ids_to_string(self, token_ids: list[int]) -> str:
        """Converts a sequence of token IDs to a string, cleaning special tokens."""
        return self._row_formatter.token_ids_to_string(token_ids)
