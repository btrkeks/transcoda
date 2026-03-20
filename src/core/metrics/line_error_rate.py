"""Line Error Rate metric for line-level evaluation."""

import torch
from torchmetrics import Metric

from .levenshtein import levenshtein


class LineErrorRate(Metric):
    """TorchMetrics-compatible Line Error Rate tracker."""

    full_state_update = False
    higher_is_better = False

    def __init__(
        self,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        i2w: dict[int, str],
        compute_on_step: bool = False,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.i2w = i2w
        self.add_state("edit_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("target_length", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore[override]
        batch_edit_distance = 0.0
        batch_target_length = 0.0

        for pred_seq, tgt_seq in zip(
            preds.detach().cpu().tolist(), target.detach().cpu().tolist(), strict=False
        ):
            pred_str = self._token_ids_to_string(pred_seq)
            target_str = self._token_ids_to_string(tgt_seq)

            if not target_str:
                continue

            # Parse as lines
            pred_lines = pred_str.split("\n")
            target_lines = target_str.split("\n")

            batch_edit_distance += levenshtein(pred_lines, target_lines)
            batch_target_length += len(target_lines)

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

    def _token_ids_to_string(self, sequence: list[int]) -> str:
        """Convert token IDs to string representation."""
        tokens = []
        for token_id in sequence:
            if token_id == self.pad_id:
                break
            token = self.i2w.get(token_id, "")
            if token_id == self.bos_id:
                continue
            if token_id == self.eos_id:
                break
            tokens.append(token)

        # Convert to string
        result = "".join(tokens)
        return result
