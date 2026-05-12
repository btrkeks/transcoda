"""Symbol Error Rate metric for token-level evaluation."""

import torch
from torchmetrics import Metric

from .levenshtein import levenshtein


class SymbolErrorRate(Metric):
    """TorchMetrics-compatible Symbol Error Rate tracker."""

    full_state_update = False
    higher_is_better = False

    def __init__(
        self,
        pad_id: int,
        bos_id: int,
        eos_id: int,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.add_state("edit_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("target_length", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore[override]
        # Work on CPU lists once; use C++ Levenshtein on integer sequences (no strings)
        batch_edit_distance = 0.0
        batch_target_length = 0.0

        for pred_seq, tgt_seq in zip(
            preds.detach().cpu().tolist(), target.detach().cpu().tolist(), strict=False
        ):
            # Trim at <pad>/<eos> and drop <bos>
            pred_ids = self._trim_ids(pred_seq)
            tgt_ids = self._trim_ids(tgt_seq)
            if not tgt_ids:
                continue

            batch_edit_distance += levenshtein(pred_ids, tgt_ids)
            batch_target_length += len(tgt_ids)

        if batch_target_length == 0:
            return

        device = self.edit_distance.device
        dtype = self.edit_distance.dtype
        self.edit_distance += torch.tensor(batch_edit_distance, device=device, dtype=dtype)
        self.target_length += torch.tensor(batch_target_length, device=device, dtype=dtype)

    def compute(self) -> torch.Tensor:  # type: ignore[override]
        if self.target_length.item() == 0:
            return torch.tensor(
                0.0, device=self.edit_distance.device, dtype=self.edit_distance.dtype
            )
        return 100.0 * self.edit_distance / self.target_length

    def _trim_ids(self, seq: list[int]) -> list[int]:
        """Trim sequence at padding/eos and remove bos tokens."""
        out = []
        for t in seq:
            if t == self.pad_id:
                break
            if t == self.bos_id:
                continue
            if t == self.eos_id:
                break
            out.append(t)
        return out

    @staticmethod
    def compute_single(
        pred: list[int], target: list[int], pad_id: int, bos_id: int, eos_id: int
    ) -> float:
        """Calculate SER for a single sequence without metric state.

        Args:
            pred: Predicted token IDs
            target: Target token IDs
            pad_id: Padding token ID
            bos_id: Beginning-of-sequence token ID
            eos_id: End-of-sequence token ID

        Returns:
            Symbol Error Rate as percentage (0-100+)
        """
        # Trim sequences
        pred_ids = []
        for t in pred:
            if t == pad_id:
                break
            if t == bos_id:
                continue
            if t == eos_id:
                break
            pred_ids.append(t)

        tgt_ids = []
        for t in target:
            if t == pad_id:
                break
            if t == bos_id:
                continue
            if t == eos_id:
                break
            tgt_ids.append(t)

        if not tgt_ids:
            return 0.0

        return levenshtein(pred_ids, tgt_ids) / len(tgt_ids) * 100.0
