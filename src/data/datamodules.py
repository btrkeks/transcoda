import hashlib
import random

import torch
from lightning import LightningDataModule

from src.config import Data, Training
from src.data.collators import Img2SeqCollator
from src.data.datasets import load_dataset_direct
from src.metrics_schema import base_val_set_name, subset_val_set_name
from src.utils.repro import worker_init_fn


class PregeneratedSyntheticGrandStaffDM(LightningDataModule):
    def __init__(
        self,
        tokenizer,
        data_config: Data,
        training_config: Training,
        max_decoder_len: int,
    ) -> None:
        super().__init__()
        self.data_cfg = data_config
        self.training_cfg = training_config
        self.tokenizer = tokenizer
        self.max_decoder_len = max_decoder_len
        self.train_set = None
        self.val_sets: dict[str, torch.utils.data.Dataset] = {}
        self._empty_val_subsets: dict[str, torch.utils.data.Dataset] = {}
        self._val_subset_indices_by_name: dict[str, list[int]] = {}
        self._val_subset_sets: dict[str, torch.utils.data.Dataset] = {}
        self._setup_stage: str | None = None
        # Keep dataloader index -> name mapping stable across setup stages.
        self._val_set_name_order = list(self.data_cfg.validation_paths.keys())
        self._val_subset_name_order = [
            subset_val_set_name(name)
            for name in self._val_set_name_order
            if name in self.training_cfg.frequent_validation_subset_sizes
        ]
        self.collator = Img2SeqCollator(
            tokenizer=self.tokenizer,
            fixed_height=self.data_cfg.fixed_image_height,
            fixed_width=self.data_cfg.fixed_image_width,
            enforce_fixed_size=True,
            fixed_decoder_length=self.max_decoder_len,
        )

    @staticmethod
    def _normalize_stage(stage: str | object | None) -> str | None:
        """Normalize Lightning stage names from strings/enums."""
        if stage is None:
            return None
        raw = getattr(stage, "value", stage)
        normalized = str(raw).strip().lower()
        if "." in normalized:
            normalized = normalized.split(".")[-1]
        aliases = {
            "fit": "fit",
            "fitting": "fit",
            "validate": "validate",
            "validating": "validate",
            "test": "test",
            "testing": "test",
            "predict": "predict",
            "predicting": "predict",
        }
        return aliases.get(normalized, normalized)

    def setup(self, stage: str | None = None) -> None:
        normalized_stage = self._normalize_stage(stage)
        self._setup_stage = normalized_stage
        load_train = normalized_stage in (None, "fit")
        load_val = normalized_stage in (None, "fit", "validate")

        # Train: single dataset loaded directly from path
        if load_train and self.train_set is None:
            self.train_set = load_dataset_direct(
                dataset_path=self.data_cfg.train_path,
                tokenizer=self.tokenizer,
            )

        # Validation: multiple named datasets
        if load_val and not self.val_sets:
            for name, path in self.data_cfg.validation_paths.items():
                self.val_sets[name] = load_dataset_direct(
                    dataset_path=path,
                    tokenizer=self.tokenizer,
                )
        if load_val and not self._val_subset_sets:
            for name in self.training_cfg.frequent_validation_subset_sizes:
                if name not in self.val_sets:
                    continue
                subset_indices = self._build_validation_subset_indices(name)
                self._val_subset_indices_by_name[name] = subset_indices
                self._val_subset_sets[name] = torch.utils.data.Subset(
                    self.val_sets[name], subset_indices
                )

    def _build_validation_subset_indices(self, set_name: str) -> list[int]:
        subset_size = self.training_cfg.frequent_validation_subset_sizes.get(set_name)
        if subset_size is None:
            return []

        dataset_len = len(self.val_sets[set_name])
        subset_size = min(int(subset_size), dataset_len)
        if subset_size >= dataset_len:
            return list(range(dataset_len))

        seed_material = (
            f"{self.training_cfg.frequent_validation_subset_seed}:{set_name}".encode("utf-8")
        )
        derived_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        indices = list(range(dataset_len))
        random.Random(derived_seed).shuffle(indices)
        return sorted(indices[:subset_size])

    def _empty_subset_for_set(self, set_name: str) -> torch.utils.data.Dataset:
        subset = self._empty_val_subsets.get(set_name)
        if subset is not None:
            return subset

        subset = torch.utils.data.Subset(self.val_sets[set_name], [])
        self._empty_val_subsets[set_name] = subset
        return subset

    def _is_full_validation_pass(self) -> bool:
        if not self.training_cfg.tiered_validation_enabled:
            return True
        if self._setup_stage not in (None, "fit"):
            return True

        trainer = getattr(self, "trainer", None)
        global_step = int(getattr(trainer, "global_step", 0) or 0)
        full_every = max(1, self.training_cfg.full_validation_every_n_steps)
        return global_step > 0 and global_step % full_every == 0

    def _active_validation_set_names(self) -> set[str]:
        all_names = set(self.val_set_names)
        if self._is_full_validation_pass():
            return all_names

        frequent_names = {
            name for name in self.training_cfg.frequent_validation_set_names if name in all_names
        }
        if not frequent_names:
            # Fail safe: if configured names are missing, run full validation.
            return all_names
        return frequent_names

    @property
    def val_set_names(self) -> list[str]:
        """Return ordered list of validation set names (for dataloader_idx mapping)."""
        return list(self._val_set_name_order)

    @property
    def val_loader_names(self) -> list[str]:
        """Return ordered list of validation loader names, including subset loaders."""
        return [*self._val_set_name_order, *self._val_subset_name_order]

    def _current_val_loader_names(self) -> list[str]:
        if self._setup_stage == "validate":
            return self.val_set_names
        return self.val_loader_names

    def _train_loader_kwargs(self) -> dict:
        workers = self.training_cfg.num_workers
        kw = {
            "num_workers": workers,
            "pin_memory": self.training_cfg.train_pin_memory,
        }
        if workers > 0:
            kw["persistent_workers"] = self.training_cfg.train_persistent_workers
            kw["prefetch_factor"] = self.training_cfg.train_prefetch_factor
            kw["worker_init_fn"] = worker_init_fn  # Ensure reproducible augmentation per worker
        return kw

    def _effective_val_num_workers(self) -> int:
        if self.training_cfg.val_num_workers is not None:
            return self.training_cfg.val_num_workers
        return min(2, self.training_cfg.num_workers) if self.training_cfg.num_workers > 0 else 0

    @staticmethod
    def _format_gib(num_bytes: int) -> str:
        return f"{num_bytes / (1024**3):.2f} GiB"

    def _available_host_memory_bytes(self) -> int | None:
        """Best-effort available host RAM for runtime loader-safety decisions."""
        try:
            import psutil

            return int(psutil.virtual_memory().available)
        except Exception:
            return None

    def _estimate_validation_queue_bytes(
        self,
        *,
        loader_count: int,
        val_batch_size: int,
        workers: int,
        prefetch_factor: int,
        pin_memory: bool,
    ) -> int:
        # Validation images are normalized float32 tensors with fixed CxHxW geometry.
        bytes_per_sample = int(3 * self.data_cfg.fixed_image_height * self.data_cfg.fixed_image_width * 4)
        bytes_per_batch = int(val_batch_size) * bytes_per_sample
        in_flight_batches = 1 + (int(workers) * max(1, int(prefetch_factor)) if workers > 0 else 0)
        queued_bytes = int(loader_count) * bytes_per_batch * in_flight_batches
        if pin_memory:
            # Pinned-memory staging can roughly double in-flight host allocations.
            queued_bytes *= 2
        return queued_bytes

    def _resolve_validate_loader_policy(
        self,
        *,
        val_batch_size: int,
        loader_count: int,
    ) -> dict:
        workers = self._effective_val_num_workers()
        pin_memory = bool(self.training_cfg.val_pin_memory)
        persistent_workers = bool(self.training_cfg.val_persistent_workers)
        prefetch_factor = int(self.training_cfg.val_prefetch_factor)

        if workers <= 0:
            return {
                "num_workers": 0,
                "pin_memory": pin_memory,
            }

        # Safety clamp is applied only for standalone validation runs, where
        # memory-heavy full-set evals are commonly used and OOMs are costly.
        if self._setup_stage == "validate":
            available_bytes = self._available_host_memory_bytes()
            if available_bytes is not None and available_bytes > 0:
                # Keep a conservative headroom budget for model/runtime allocations.
                memory_budget = max(1, int(available_bytes * 0.30))
                original = (workers, prefetch_factor, pin_memory)
                estimated = self._estimate_validation_queue_bytes(
                    loader_count=loader_count,
                    val_batch_size=val_batch_size,
                    workers=workers,
                    prefetch_factor=prefetch_factor,
                    pin_memory=pin_memory,
                )

                while estimated > memory_budget and prefetch_factor > 1:
                    prefetch_factor -= 1
                    estimated = self._estimate_validation_queue_bytes(
                        loader_count=loader_count,
                        val_batch_size=val_batch_size,
                        workers=workers,
                        prefetch_factor=prefetch_factor,
                        pin_memory=pin_memory,
                    )

                if estimated > memory_budget and pin_memory:
                    pin_memory = False
                    estimated = self._estimate_validation_queue_bytes(
                        loader_count=loader_count,
                        val_batch_size=val_batch_size,
                        workers=workers,
                        prefetch_factor=prefetch_factor,
                        pin_memory=pin_memory,
                    )

                while estimated > memory_budget and workers > 0:
                    workers -= 1
                    estimated = self._estimate_validation_queue_bytes(
                        loader_count=loader_count,
                        val_batch_size=val_batch_size,
                        workers=workers,
                        prefetch_factor=prefetch_factor,
                        pin_memory=pin_memory,
                    )

                adjusted = (workers, prefetch_factor, pin_memory)
                if adjusted != original:
                    print(
                        "[val] memory safety: adjusted validate dataloader "
                        f"(workers {original[0]}->{workers}, "
                        f"prefetch_factor {original[1]}->{prefetch_factor}, "
                        f"pin_memory {original[2]}->{pin_memory}) "
                        f"for {loader_count} loader(s), batch_size={val_batch_size}; "
                        f"estimated in-flight host queue "
                        f"{self._format_gib(estimated)} "
                        f"(budget {self._format_gib(memory_budget)}).",
                        flush=True,
                    )

        kw = {
            "num_workers": workers,
            "pin_memory": pin_memory,
        }
        if workers > 0:
            kw["persistent_workers"] = persistent_workers
            kw["prefetch_factor"] = prefetch_factor
            kw["worker_init_fn"] = worker_init_fn  # Ensure reproducible augmentation per worker
        return kw

    def _val_loader_kwargs(self, *, val_batch_size: int, loader_count: int) -> dict:
        return self._resolve_validate_loader_policy(
            val_batch_size=val_batch_size,
            loader_count=loader_count,
        )

    def train_dataloader(self):
        if self.train_set is None:
            raise RuntimeError("Train dataset is not loaded. Call setup(stage='fit') before train_dataloader().")
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.training_cfg.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collator,
            **self._train_loader_kwargs(),
        )

    def val_dataloader(self):
        if not self.val_sets:
            raise RuntimeError(
                "Validation datasets are not loaded. Call setup(stage='fit' or 'validate') before val_dataloader()."
            )
        # Return list of dataloaders - Lightning handles multiple validation sets
        val_batch_size = self.training_cfg.val_batch_size or self.training_cfg.batch_size
        active_set_names = self._active_validation_set_names()
        loader_names = self._current_val_loader_names()
        val_loader_kwargs = self._val_loader_kwargs(
            val_batch_size=val_batch_size,
            loader_count=len(loader_names),
        )
        return [
            torch.utils.data.DataLoader(
                self._validation_dataset_for_loader(name, active_set_names),
                batch_size=val_batch_size,
                collate_fn=self.collator,
                **val_loader_kwargs,
            )
            for name in loader_names
        ]

    def _validation_dataset_for_loader(
        self,
        loader_name: str,
        active_set_names: set[str],
    ) -> torch.utils.data.Dataset:
        if loader_name in self._val_subset_name_order:
            return self._val_subset_sets[base_val_set_name(loader_name)]
        if loader_name in active_set_names:
            return self.val_sets[loader_name]
        return self._empty_subset_for_set(loader_name)
