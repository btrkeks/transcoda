from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FCMAEDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_dir: str | None = None
    manifest_path: str | None = None
    image_height: int = 1485
    image_width: int = 1050
    extensions: list[str] = Field(default_factory=lambda: [".png", ".jpg", ".jpeg", ".webp"])


class FCMAEModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoder_model_name_or_path: str = "facebook/convnextv2-base-22k-224"
    patch_size: int = 32
    mask_ratio: float = 0.6
    decoder_dim: int = 512
    decoder_depth: int = 2
    norm_pix_loss: bool = True
    ink_bias_strength: float = 0.3


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
    resume_from_checkpoint: str | None = None
    devices: int = 1
    strategy: str = "auto"
    num_nodes: int = 1

    @model_validator(mode="after")
    def validate_ddp(self) -> FCMAETrainingConfig:
        if self.devices < 1:
            raise ValueError("training.devices must be >= 1")
        if self.num_nodes < 1:
            raise ValueError("training.num_nodes must be >= 1")
        if self.strategy == "ddp_spawn":
            raise ValueError(
                "training.strategy='ddp_spawn' is not supported; use 'ddp' instead. "
                "ddp_spawn re-pickles the datamodule per subprocess and is fragile "
                "with this setup."
            )
        if self.devices * self.num_nodes > 1 and self.strategy == "auto":
            self.strategy = "ddp"
        return self


class FCMAECheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dirpath: str = "weights/fcmae"
    filename: str = "fcmae-{step:06d}"
    save_last: bool = True
    save_top_k: int = -1
    every_n_train_steps: int | None = 5000


class FCMAELoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wandb_enabled: bool = False
    project: str = "SMT-FCMAE"
    run_name: str | None = None
    group: str | None = None
    tags: list[str] = Field(default_factory=list)
    log_model: bool = False
    log_reconstructions: bool = True
    log_reconstruction_every_n_steps: int = 500
    log_reconstruction_max_batches: int | None = 20
    log_reconstruction_max_images: int = 4


class FCMAEExportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str | None = None
    export_on_train_end: bool = False
    overwrite: bool = False
    validate_export: bool = True


class FCMAEConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: FCMAEDataConfig = Field(default_factory=FCMAEDataConfig)
    model: FCMAEModelConfig = Field(default_factory=FCMAEModelConfig)
    training: FCMAETrainingConfig = Field(default_factory=FCMAETrainingConfig)
    checkpoint: FCMAECheckpointConfig = Field(default_factory=FCMAECheckpointConfig)
    logging: FCMAELoggingConfig = Field(default_factory=FCMAELoggingConfig)
    export: FCMAEExportConfig = Field(default_factory=FCMAEExportConfig)

    @model_validator(mode="after")
    def validate_config(self) -> FCMAEConfig:
        if (self.data.image_dir is None) == (self.data.manifest_path is None):
            raise ValueError("exactly one of data.image_dir and data.manifest_path must be set")

        if self.data.image_height <= 0 or self.data.image_width <= 0:
            raise ValueError("data.image_height and data.image_width must be positive")
        if self.model.patch_size <= 0:
            raise ValueError("model.patch_size must be positive")
        if not 0 < self.model.mask_ratio < 1:
            raise ValueError("model.mask_ratio must satisfy 0 < mask_ratio < 1")
        if self.model.ink_bias_strength < 0:
            raise ValueError("model.ink_bias_strength must be >= 0")

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
        if self.checkpoint.save_top_k < -1:
            raise ValueError("checkpoint.save_top_k must be >= -1")
        if (
            self.checkpoint.every_n_train_steps is not None
            and self.checkpoint.every_n_train_steps < 1
        ):
            raise ValueError("checkpoint.every_n_train_steps must be null or >= 1")
        if self.logging.log_reconstruction_every_n_steps < 1:
            raise ValueError("logging.log_reconstruction_every_n_steps must be >= 1")
        if (
            self.logging.log_reconstruction_max_batches is not None
            and self.logging.log_reconstruction_max_batches < 1
        ):
            raise ValueError("logging.log_reconstruction_max_batches must be null or >= 1")
        if self.logging.log_reconstruction_max_images < 1:
            raise ValueError("logging.log_reconstruction_max_images must be >= 1")
        if self.export.export_on_train_end and self.export.output_dir is None:
            raise ValueError("export.output_dir must be set when export.export_on_train_end=true")

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
