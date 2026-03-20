from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn


@dataclass
class VisionFrontendOutput:
    """Frontend output consumed by the decoder."""

    encoder_tokens_raw: torch.Tensor
    encoder_tokens_pos: torch.Tensor
    encoder_attention_mask: torch.Tensor | None = None


class VisionFrontend(Protocol):
    """Protocol for vision frontends that produce decoder-ready tokens."""

    encoder: nn.Module
    projector: nn.Module

    def forward(
        self,
        images: torch.Tensor,
        image_sizes: torch.Tensor | None = None,
    ) -> VisionFrontendOutput:
        ...


class ProjectorMLP(nn.Module):
    """Token-space MLP projector: Linear -> GELU -> Linear."""

    def __init__(self, in_dim: int, out_dim: int, hidden_mult: float = 4.0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = max(out_dim, int(round(out_dim * hidden_mult)))

        self.fc1 = nn.Linear(in_dim, self.hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class PositionalEncoding2D(nn.Module):
    def __init__(self, dim: int, h_max: int, w_max: int):
        assert dim % 4 == 0, "PositionalEncoding2D expects dim divisible by 4."
        super().__init__()
        self.h_max = h_max
        self.w_max = w_max
        self.dim = dim

        self.pe: torch.Tensor
        self.register_buffer(
            "pe",
            torch.zeros((dim, h_max, w_max), requires_grad=False),
            persistent=False,
        )

        div = torch.exp(
            -torch.arange(0.0, dim // 2, 2) / dim * torch.log(torch.tensor(1e4))
        ).unsqueeze(1)
        w_pos = torch.arange(0.0, w_max) * div
        h_pos = torch.arange(0.0, h_max) * div
        self.pe[: dim // 2 : 2] = torch.sin(h_pos).unsqueeze(2).repeat(1, 1, w_max)
        self.pe[1 : dim // 2 : 2] = torch.cos(h_pos).unsqueeze(2).repeat(1, 1, w_max)
        self.pe[dim // 2 :: 2] = torch.sin(w_pos).unsqueeze(1).repeat(1, h_max, 1)
        self.pe[dim // 2 + 1 :: 2] = torch.cos(w_pos).unsqueeze(1).repeat(1, h_max, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.get_pe_by_size(x.size(-2), x.size(-1)).to(dtype=x.dtype, device=x.device)

    def get_pe_by_size(self, h: int, w: int) -> torch.Tensor:
        return self.pe[:, :h, :w]
