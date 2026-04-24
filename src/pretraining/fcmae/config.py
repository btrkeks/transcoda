from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FCMAEDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_dir: str | None = None
    manifest_path: str | None = None
    image_height: int = 768
    image_width: int = 544
    extensions: list[str] = Field(default_factory=lambda: [".png", ".jpg", ".jpeg", ".webp"])


class FCMAEModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoder_model_name_or_path: str = "facebook/convnextv2-base-22k-224"
    patch_size: int = 32
    mask_ratio: float = 0.6
    decoder_dim: int = 512
    decoder_depth: int = 2
    norm_pix_loss: bool = True


class FCMAETrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = 2
    num_workers: int = 4
    max_steps: int = 1000
    accumulate_grad_batches: int = 4
    base_learning_rate: float = 1.5e-4
    weight_decay: float = 0.05
    warmup_steps: int = 100
    precision: str = "bf16-mixed"
    seed: int = 0


class FCMAECheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dirpath: str = "weights/fcmae"
    filename: str = "fcmae-{step:06d}"
    save_last: bool = True
    save_top_k: int = 0


class FCMAEConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: FCMAEDataConfig = Field(default_factory=FCMAEDataConfig)
    model: FCMAEModelConfig = Field(default_factory=FCMAEModelConfig)
    training: FCMAETrainingConfig = Field(default_factory=FCMAETrainingConfig)
    checkpoint: FCMAECheckpointConfig = Field(default_factory=FCMAECheckpointConfig)

    @model_validator(mode="after")
    def validate_config(self) -> FCMAEConfig:
        if (self.data.image_dir is None) == (self.data.manifest_path is None):
            raise ValueError("exactly one of data.image_dir and data.manifest_path must be set")

        if self.data.image_height <= 0 or self.data.image_width <= 0:
            raise ValueError("data.image_height and data.image_width must be positive")
        if self.model.patch_size <= 0:
            raise ValueError("model.patch_size must be positive")
        if self.data.image_height % self.model.patch_size != 0:
            raise ValueError("data.image_height must be divisible by model.patch_size")
        if self.data.image_width % self.model.patch_size != 0:
            raise ValueError("data.image_width must be divisible by model.patch_size")
        if not 0 < self.model.mask_ratio < 1:
            raise ValueError("model.mask_ratio must satisfy 0 < mask_ratio < 1")

        if self.training.batch_size <= 0:
            raise ValueError("training.batch_size must be positive")
        if self.training.num_workers < 0:
            raise ValueError("training.num_workers must be >= 0")
        if self.training.max_steps <= 0:
            raise ValueError("training.max_steps must be positive")
        if self.training.accumulate_grad_batches <= 0:
            raise ValueError("training.accumulate_grad_batches must be positive")
        if self.training.base_learning_rate < 0:
            raise ValueError("training.base_learning_rate must be >= 0")
        if self.training.weight_decay < 0:
            raise ValueError("training.weight_decay must be >= 0")
        if self.training.warmup_steps < 0:
            raise ValueError("training.warmup_steps must be >= 0")

        normalized_extensions = []
        for extension in self.data.extensions:
            suffix = extension.lower()
            if not suffix.startswith("."):
                suffix = f".{suffix}"
            normalized_extensions.append(suffix)
        self.data.extensions = normalized_extensions
        return self


def load_fcmae_config(path: str | Path) -> FCMAEConfig:
    return FCMAEConfig.model_validate_json(Path(path).read_text())

