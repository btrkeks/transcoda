"""Formatting helpers for validation example rows logged to W&B tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import torch

from src.core.metrics import SymbolErrorRate
from src.core.text_processing import token_ids_to_string


class HeapEntryLike(Protocol):
    image: torch.Tensor
    pred_ids: list[int]
    gt_ids: list[int]
    cer: float
    cer_no_ties_beams: float | None
    val_set_name: str
    sample_id: int
    source_name: str | None


@dataclass(frozen=True)
class FormattedValidationRow:
    row_id: str
    category: str
    image: torch.Tensor
    ground_truth: str
    prediction: str
    prediction_rendered_image: torch.Tensor | None
    ser: float
    cer: float
    cer_no_ties_beams: float | None
    diff_html: str


class ValidationTableRowFormatter:
    """Converts retained heap entries into display-ready row payloads."""

    def __init__(
        self,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        i2w: dict[int, str],
    ) -> None:
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.i2w = i2w

    def token_ids_to_string(self, token_ids: list[int]) -> str:
        return token_ids_to_string(token_ids, self.i2w, self.pad_token_id, add_header=False)

    def format_entry(
        self,
        entry: HeapEntryLike,
        *,
        category: str,
        current_epoch: int,
        diff_builder: Callable[[str, str], str],
    ) -> FormattedValidationRow:
        pred_text = self.token_ids_to_string(entry.pred_ids)
        gt_text = self.token_ids_to_string(entry.gt_ids)
        # Disable predicted-score rendering for now: malformed model outputs can
        # trigger an uncaught native Verovio/Humdrum abort and crash validation.
        pred_rendered_image = None

        ser = SymbolErrorRate.compute_single(
            pred=entry.pred_ids,
            target=entry.gt_ids,
            pad_id=self.pad_token_id,
            bos_id=self.bos_token_id,
            eos_id=self.eos_token_id,
        )

        diff_html = diff_builder(gt_text, pred_text)

        # Convert to float32 for wandb compatibility
        image_f32 = entry.image.float() if entry.image.dtype != torch.float32 else entry.image
        fallback_row_id = f"epoch_{current_epoch}_{entry.val_set_name}_ex_{entry.sample_id}"
        if entry.source_name is not None and entry.source_name.strip():
            row_id = Path(entry.source_name).name
        else:
            row_id = fallback_row_id

        return FormattedValidationRow(
            row_id=row_id,
            category=category,
            image=image_f32,
            ground_truth=gt_text,
            prediction=pred_text,
            prediction_rendered_image=pred_rendered_image,
            ser=ser,
            cer=entry.cer,
            cer_no_ties_beams=entry.cer_no_ties_beams,
            diff_html=diff_html,
        )
