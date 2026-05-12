"""Dense SimMIM-style masked image modeling inspired by ConvNeXt V2 FCMAE.

The package keeps the `fcmae` name for plan/history continuity and decoder
lineage, but it intentionally does not implement Meta's sparse FCMAE path.
"""

from src.pretraining.fcmae.config import FCMAEConfig
from src.pretraining.fcmae.model import DenseMaskedImageModelingConvNeXtV2

__all__ = ["DenseMaskedImageModelingConvNeXtV2", "FCMAEConfig"]
