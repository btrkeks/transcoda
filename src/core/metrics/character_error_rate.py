"""Character Error Rate metric for character-level evaluation."""

from collections.abc import Callable, Iterable

import torch
from torchmetrics import Metric

from .levenshtein import levenshtein


class CharacterErrorRate(Metric):
    """TorchMetrics-compatible Character Error Rate tracker."""

    full_state_update = False
    higher_is_better = False

    def __init__(
        self,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        i2w: dict[int, str],
        text_normalizer: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.i2w = i2w
        self.text_normalizer = text_normalizer
        self.add_state("edit_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("target_length", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore[override]
        batch_edit_distance = 0.0
        batch_target_length = 0.0

        for pred_seq, tgt_seq in zip(
            preds.detach().cpu().tolist(), target.detach().cpu().tolist(), strict=False
        ):
            pred_chars = CharacterErrorRate.token_ids_to_codepoints(
                pred_seq,
                pad_id=self.pad_id,
                i2w=self.i2w,
                text_normalizer=self.text_normalizer,
            )
            tgt_chars = CharacterErrorRate.token_ids_to_codepoints(
                tgt_seq,
                pad_id=self.pad_id,
                i2w=self.i2w,
                text_normalizer=self.text_normalizer,
            )

            if not tgt_chars:
                continue

            batch_edit_distance += levenshtein(pred_chars, tgt_chars)
            batch_target_length += len(tgt_chars)

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

    @staticmethod
    def token_ids_to_codepoints(
        seq: Iterable[int],
        pad_id: int,
        i2w: dict[int, str],
        text_normalizer: Callable[[str], str] | None = None,
    ) -> list[int]:
        """Expand token IDs to Unicode codepoints."""
        text = CharacterErrorRate.token_ids_to_text(seq=seq, pad_id=pad_id, i2w=i2w)
        if text_normalizer is not None:
            text = text_normalizer(text)
        return list(map(ord, text))

    @staticmethod
    def token_ids_to_text(seq: Iterable[int], pad_id: int, i2w: dict[int, str]) -> str:
        """Decode token IDs to plain text while skipping BOS/EOS."""
        skip = {"<bos>", "<eos>"}
        parts: list[str] = []
        for t in seq:
            if t == pad_id:
                break
            token = i2w.get(t)
            if not token or token in skip:
                continue
            parts.append(token)
        return "".join(parts)

    @staticmethod
    def compute_single(
        pred: list[int],
        target: list[int],
        pad_id: int,
        i2w: dict[int, str],
        text_normalizer: Callable[[str], str] | None = None,
    ) -> float:
        """Calculate CER for a single sequence without metric state.

        Args:
            pred: Predicted token IDs
            target: Target token IDs
            pad_id: Padding token ID
            i2w: Index-to-word vocabulary mapping

        Returns:
            Character Error Rate as percentage (0-100+)
        """
        pred_text = CharacterErrorRate.token_ids_to_text(pred, pad_id=pad_id, i2w=i2w)
        tgt_text = CharacterErrorRate.token_ids_to_text(target, pad_id=pad_id, i2w=i2w)
        if text_normalizer is not None:
            pred_text = text_normalizer(pred_text)
            tgt_text = text_normalizer(tgt_text)
        pred_chars = list(map(ord, pred_text))
        tgt_chars = list(map(ord, tgt_text))

        if not tgt_chars:
            return 0.0

        return levenshtein(pred_chars, tgt_chars) / len(tgt_chars) * 100.0
