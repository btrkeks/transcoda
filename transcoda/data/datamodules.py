import hashlib
import json
import random
import re
import time

import torch
from lightning import LightningDataModule

from transcoda.config import Data, Training
from transcoda.data.collators import Img2SeqCollator
from transcoda.data.datasets import load_dataset_direct
from transcoda.data.mixture import (
    CappedMixtureSampler,
    LoadedMixture,
    compile_mixture_columns,
    load_mixture,
)
from transcoda.data.train_sources import (
    MultiSourceDataset,
    check_sample_fields,
    dataset_column,
    source_dataset_path,
    train_sources_mixture,
)
from transcoda.metrics_schema import base_val_set_name, subset_val_set_name
from transcoda.preprocessing import image_to_uint8_chw
from transcoda.utils.repro import worker_init_fn

# Duration digits of a kern note/rest token: not preceded by a digit or "%"
# (so rational durations like 16%9 count once) and followed by an optional
# rational part, augmentation dots, then a pitch or rest letter.
# Interpretation/barline/comment lines are skipped by the caller, which keeps
# meters like *M12/8 out of the counts.
_KERN_RECIP_RE = re.compile(r"(?<![\d%])(\d+)(%\d+)?\.*(?=[a-gA-Gr])")

# A transcription counts as tuplet-rich when non-power-of-two durations are at
# least this share of its duration-bearing tokens and at least this many
# overall. Thresholds follow experiments/analysis/05roh5s9.md: tuplet-heavy
# Polish validation pages sit above a 5% non-binary duration share.
_TUPLET_RICH_MIN_FRACTION = 0.05
_TUPLET_RICH_MIN_COUNT = 8


def transcription_is_tuplet_rich(text: str) -> bool:
    """Whether a kern transcription has a meaningful share of tuplet durations."""
    total = 0
    tuplet = 0
    for line in text.split("\n"):
        if line.startswith(("*", "=", "!")):
            continue
        for match in _KERN_RECIP_RE.finditer(line):
            duration = int(match.group(1))
            is_rational = match.group(2) is not None
            if duration <= 0 and not is_rational:
                # "0" (breve) and "00" (longa) are binary durations.
                continue
            total += 1
            # Rational durations (e.g. 16%9) are tuplets regardless of base.
            if is_rational or duration & (duration - 1):
                tuplet += 1
    if tuplet < _TUPLET_RICH_MIN_COUNT:
        return False
    return tuplet >= _TUPLET_RICH_MIN_FRACTION * total


class FetchTimingSink:
    """Mutable epoch-local sink for train DataLoader fetch timing."""

    def __init__(self) -> None:
        self.total_s = 0.0

    def reset(self) -> None:
        self.total_s = 0.0

    def add(self, elapsed_s: float) -> None:
        self.total_s += max(0.0, float(elapsed_s))


class TimedDataLoader(torch.utils.data.DataLoader):
    """DataLoader subclass that records wall time spent in iterator next()."""

    def __init__(self, *args, fetch_timing_sink: FetchTimingSink | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fetch_timing_sink = fetch_timing_sink

    def __iter__(self):
        iterator = super().__iter__()
        while True:
            start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            elapsed_s = time.perf_counter() - start
            if self.fetch_timing_sink is not None:
                self.fetch_timing_sink.add(elapsed_s)
            yield batch


class PregeneratedSyntheticGrandStaffDM(LightningDataModule):
    def __init__(
        self,
        tokenizer,
        data_config: Data,
        training_config: Training,
        max_decoder_len: int,
        training_seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_cfg = data_config
        self.training_cfg = training_config
        self.tokenizer = tokenizer
        self.max_decoder_len = max_decoder_len
        self.training_seed = int(training_seed)
        self.train_set = None
        self.val_sets: dict[str, torch.utils.data.Dataset] = {}
        self._empty_val_subsets: dict[str, torch.utils.data.Dataset] = {}
        self._val_subset_indices_by_name: dict[str, list[int]] = {}
        self._val_subset_sets: dict[str, torch.utils.data.Dataset] = {}
        self._setup_stage: str | None = None
        self._mixture_sampler: CappedMixtureSampler | None = None
        self._restored_mixture_sampler_state: dict[str, int | str] | None = None
        self.train_fetch_sink = FetchTimingSink()
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
            decoder_length_buckets=self.training_cfg.decoder_length_buckets,
        )
        # Validation keeps the single fixed decoder length: its batch sizes
        # already vary (last partial batch), and bucketing there would multiply
        # torch.compile shape variants for little gain.
        self.val_collator = Img2SeqCollator(
            tokenizer=self.tokenizer,
            fixed_height=self.data_cfg.fixed_image_height,
            fixed_width=self.data_cfg.fixed_image_width,
            enforce_fixed_size=True,
            fixed_decoder_length=self.max_decoder_len,
        )

    def state_dict(self) -> dict[str, object]:
        state = (
            self._mixture_sampler.state_dict()
            if self._mixture_sampler is not None
            else self._restored_mixture_sampler_state
        )
        return {} if state is None else {"mixture_sampler": dict(state)}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        sampler_state = state_dict.get("mixture_sampler")
        if sampler_state is None:
            self._restored_mixture_sampler_state = None
            return
        if not isinstance(sampler_state, dict):
            raise ValueError("mixture_sampler checkpoint state must be an object")
        if self.data_cfg.mixture_path is None and self.data_cfg.train_sources is None:
            raise ValueError(
                "mixture sampler checkpoint state requires data.mixture_path or data.train_sources"
            )
        epoch = sampler_state.get("epoch")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("mixture sampler state requires a non-negative integer epoch")
        mixture_digest = sampler_state.get("mixture_digest")
        if not isinstance(mixture_digest, str):
            raise ValueError("mixture sampler state requires a mixture_digest")
        seed = sampler_state.get("seed")
        if type(seed) is not int:
            raise ValueError("mixture sampler state requires an integer seed")
        restored: dict[str, int | str] = {
            "epoch": epoch,
            "mixture_digest": mixture_digest,
            "seed": seed,
        }
        self._restored_mixture_sampler_state = restored
        if self._mixture_sampler is not None:
            self._mixture_sampler.load_state_dict(restored)

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

        # With normalize_on_gpu, workers emit uint8 CHW and TranscodaTrainer
        # normalizes after device transfer; otherwise workers emit normalized
        # float32 (4x the IPC/pin bytes).
        pixel_transform = image_to_uint8_chw if self.training_cfg.normalize_on_gpu else None

        # Train: one dataset from train_path, or one per train source
        if load_train and self.train_set is None and self.data_cfg.train_sources is not None:
            self.train_set = self._load_train_sources(pixel_transform)
        elif load_train and self.train_set is None:
            self.train_set = load_dataset_direct(
                dataset_path=self.data_cfg.train_path,
                tokenizer=self.tokenizer,
                custom_transform=pixel_transform,
                max_target_len=self.max_decoder_len,
                sample_limit=self.training_cfg.limit_train_samples,
            )
            self._log_sequence_length_filter("Train", self.train_set)

        # Validation: multiple named datasets
        if load_val and not self.val_sets:
            for name, path in self.data_cfg.validation_paths.items():
                self.val_sets[name] = load_dataset_direct(
                    dataset_path=path,
                    tokenizer=self.tokenizer,
                    custom_transform=pixel_transform,
                )
        if load_val and not self._val_subset_sets:
            self._val_subset_name_order = []
            for name in self.training_cfg.frequent_validation_subset_sizes:
                if name not in self.val_sets:
                    continue
                subset_indices = self._build_validation_subset_indices(name)
                if not subset_indices:
                    continue
                self._val_subset_indices_by_name[name] = subset_indices
                self._val_subset_sets[name] = torch.utils.data.Subset(
                    self.val_sets[name], subset_indices
                )
                self._val_subset_name_order.append(subset_val_set_name(name))

    def _load_train_sources(self, pixel_transform) -> MultiSourceDataset:
        """Each release loaded and tokenized alone; limit_train_samples applies per release."""
        parts = []
        for source in self.data_cfg.train_sources or ():
            dataset = load_dataset_direct(
                dataset_path=source_dataset_path(source),
                tokenizer=self.tokenizer,
                custom_transform=pixel_transform,
                max_target_len=self.max_decoder_len,
                sample_limit=self.training_cfg.limit_train_samples,
            )
            self._log_sequence_length_filter(f"Train {source.release}", dataset)
            parts.append((source.release, dataset))
        check_sample_fields(parts)
        train_set = MultiSourceDataset(parts)
        weights = {source.release: source.weight for source in self.data_cfg.train_sources or ()}
        print(
            "Train sources: "
            + ", ".join(
                f"{name} ({rows:,} rows, weight {weights[name]})"
                for name, rows in train_set.source_lengths().items()
            ),
            flush=True,
        )
        return train_set

    def _build_validation_subset_indices(self, set_name: str) -> list[int]:
        subset_size = self.training_cfg.frequent_validation_subset_sizes.get(set_name)
        if subset_size is None:
            return []

        dataset_len = len(self.val_sets[set_name])
        if subset_size >= dataset_len:
            return []
        subset_size = int(subset_size)

        seed_material = f"{self.training_cfg.frequent_validation_subset_seed}:{set_name}".encode()
        derived_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        indices = list(range(dataset_len))
        random.Random(derived_seed).shuffle(indices)
        return sorted(indices[:subset_size])

    @staticmethod
    def _log_sequence_length_filter(label: str, dataset: torch.utils.data.Dataset) -> None:
        stats = getattr(dataset, "sequence_length_filter", None)
        if stats is None or stats.removed_count == 0:
            return
        print(
            f"{label} sequence filter: kept {stats.after_count:,} / "
            f"{stats.before_count:,}, removed {stats.removed_count:,} "
            f"over max_target_len={stats.max_target_len} "
            f"({stats.removed_rate:.4%}).",
            flush=True,
        )

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
        # Validation images have fixed CxHxW geometry: uint8 when normalization
        # is deferred to the GPU, normalized float32 otherwise.
        bytes_per_pixel = 1 if self.training_cfg.normalize_on_gpu else 4
        bytes_per_sample = int(
            3 * self.data_cfg.fixed_image_height * self.data_cfg.fixed_image_width * bytes_per_pixel
        )
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

    def _train_column(self, column: str) -> list | None:
        if self.train_set is None:
            return None
        return dataset_column(self.train_set, column)

    def _train_string_column(self, column: str) -> list[str] | None:
        values = self._train_column(column)
        return None if values is None else [str(value) for value in values]

    def _train_mixture(self) -> LoadedMixture | None:
        """data.mixture_path's policy, or one source per release in data.train_sources."""
        if self.data_cfg.train_sources is not None:
            row_count = len(self.train_set)
            # An epoch is every loaded row (zero-weight releases included, so
            # turning a release off keeps the epoch), cut to whole batches.
            epoch_size = row_count - row_count % self.training_cfg.batch_size
            if epoch_size < 1:
                raise RuntimeError(
                    f"data.train_sources hold {row_count} rows, fewer than one batch"
                )
            return train_sources_mixture(self.data_cfg.train_sources, epoch_size=epoch_size)
        if self.data_cfg.mixture_path is not None:
            return load_mixture(self.data_cfg.mixture_path)
        return None

    def _build_mixture_sampler(self) -> CappedMixtureSampler | None:
        if self.train_set is None:
            return None
        policy = (
            "data.train_sources" if self.data_cfg.train_sources is not None else "data.mixture_path"
        )
        if self.data_cfg.train_sources is None and self.data_cfg.mixture_path is None:
            return None
        configured_legacy = [
            name
            for name, value in (
                (
                    "real_scan_sample_probability",
                    self.training_cfg.real_scan_sample_probability,
                ),
                (
                    "real_scan_no_music_sample_probability",
                    self.training_cfg.real_scan_no_music_sample_probability,
                ),
                ("tuplet_oversample_factor", self.training_cfg.tuplet_oversample_factor),
            )
            if value is not None
        ]
        if configured_legacy:
            raise RuntimeError(
                f"{policy} cannot be combined with legacy sampling flags: {configured_legacy}"
            )
        if self.training_cfg.devices * self.training_cfg.num_nodes > 1:
            raise RuntimeError(
                f"{policy} currently supports single-process training only. "
                "Add a distributed capped sampler before enabling multiple devices or nodes."
            )
        loaded = self._train_mixture()
        assert loaded is not None
        row_count = len(self.train_set)
        requested_epoch_size = (
            row_count
            if loaded.spec.epoch_size == "dataset"
            else int(loaded.spec.epoch_size)
        )
        epoch_size = requested_epoch_size
        if epoch_size < 1:
            raise RuntimeError("resolved mixture epoch is empty")
        if epoch_size % self.training_cfg.batch_size:
            raise RuntimeError(
                "resolved mixture epoch_size must be divisible by training.batch_size "
                "so drop_last cannot discard configured source quotas"
            )
        columns = {}
        for field in loaded.spec.required_fields:
            values = self._train_column(field)
            if values is None:
                raise RuntimeError(f"{policy} requires the training dataset to expose {field!r}")
            columns[field] = values
        try:
            resolved = compile_mixture_columns(
                loaded,
                columns=columns,
                row_count=row_count,
                epoch_size_override=epoch_size,
            )
        except ValueError as exc:
            raise RuntimeError(f"unable to resolve training mixture: {exc}") from exc
        print(f"Training mixture: {json.dumps(resolved.summary(), sort_keys=True)}")
        sampler = CappedMixtureSampler(resolved, seed=self.training_seed)
        if self._restored_mixture_sampler_state is not None:
            sampler.load_state_dict(self._restored_mixture_sampler_state)
        self._mixture_sampler = sampler
        return sampler

    def _build_train_sampler(self) -> torch.utils.data.Sampler | None:
        mixture_sampler = self._build_mixture_sampler()
        if mixture_sampler is not None:
            return mixture_sampler
        real_probability = self.training_cfg.real_scan_sample_probability
        tuplet_factor = self.training_cfg.tuplet_oversample_factor
        if real_probability is None and tuplet_factor is None:
            return None
        if self.train_set is None:
            return None

        if self.training_cfg.devices * self.training_cfg.num_nodes > 1:
            raise RuntimeError(
                "training.real_scan_sample_probability and training.tuplet_oversample_factor "
                "currently support single-process training only. Add a distributed weighted "
                "sampler before enabling them with multiple devices or nodes."
            )

        num_rows = len(self.train_set)
        weights = torch.ones(num_rows, dtype=torch.double)

        if tuplet_factor is not None:
            transcriptions = self._train_string_column("transcription")
            if transcriptions is None:
                raise RuntimeError(
                    "training.tuplet_oversample_factor requires the training dataset to "
                    "expose a transcription column with kern text."
                )
            if len(transcriptions) != num_rows:
                raise RuntimeError(
                    "transcription metadata length does not match the training dataset "
                    f"length ({len(transcriptions)} != {num_rows})."
                )
            tuplet_indices = [
                idx
                for idx, text in enumerate(transcriptions)
                if transcription_is_tuplet_rich(text)
            ]
            if not tuplet_indices:
                raise RuntimeError(
                    "training.tuplet_oversample_factor was set, but the training dataset "
                    "contains no tuplet-rich transcriptions."
                )
            weights[torch.tensor(tuplet_indices, dtype=torch.long)] *= tuplet_factor
            print(
                f"Tuplet oversampling: {len(tuplet_indices)}/{num_rows} tuplet-rich rows "
                f"weighted x{tuplet_factor}."
            )

        if real_probability is not None:
            no_music_probability = self.training_cfg.real_scan_no_music_sample_probability
            source_domains = self._train_string_column("source_domain")
            if source_domains is None:
                raise RuntimeError(
                    "training.real_scan_sample_probability requires the training dataset to "
                    "expose a source_domain column with real/synth values."
                )
            if len(source_domains) != num_rows:
                raise RuntimeError(
                    "source_domain metadata length does not match the training dataset length "
                    f"({len(source_domains)} != {num_rows})."
                )

            real_indices = [idx for idx, domain in enumerate(source_domains) if domain == "real"]
            non_real_indices = [
                idx for idx, domain in enumerate(source_domains) if domain != "real"
            ]
            if not real_indices:
                raise RuntimeError(
                    "training.real_scan_sample_probability was set, but the training dataset "
                    "contains no source_domain='real' rows."
                )
            if not non_real_indices:
                raise RuntimeError(
                    "training.real_scan_sample_probability was set, but the training dataset "
                    "contains only real rows."
                )

            if no_music_probability is not None:
                curation_stages = self._train_string_column("curation_stage")
                if curation_stages is None:
                    raise RuntimeError(
                        "training.real_scan_no_music_sample_probability requires the training "
                        "dataset to expose a curation_stage column."
                    )
                if len(curation_stages) != num_rows:
                    raise RuntimeError(
                        "curation_stage metadata length does not match the training dataset "
                        f"length ({len(curation_stages)} != {num_rows})."
                    )
                no_music_indices = [
                    idx
                    for idx in real_indices
                    if curation_stages[idx] == "manual_review_no_music"
                ]
                music_indices = [
                    idx
                    for idx in real_indices
                    if curation_stages[idx] != "manual_review_no_music"
                ]
                if not no_music_indices:
                    raise RuntimeError(
                        "training.real_scan_no_music_sample_probability was set, but the "
                        "training dataset contains no real rows with "
                        "curation_stage='manual_review_no_music'."
                    )
                if not music_indices:
                    raise RuntimeError(
                        "training.real_scan_no_music_sample_probability was set, but the "
                        "training dataset contains no real rows besides no-music ones."
                    )
                groups = (
                    (music_indices, real_probability),
                    (no_music_indices, no_music_probability),
                    (non_real_indices, 1.0 - real_probability - no_music_probability),
                )
                print(
                    f"Real-scan split: {len(music_indices)} music rows mass "
                    f"{real_probability}, {len(no_music_indices)} no-music rows mass "
                    f"{no_music_probability}."
                )
            else:
                groups = (
                    (real_indices, real_probability),
                    (non_real_indices, 1.0 - real_probability),
                )

            # Rescale each source-domain group to its configured total mass so the
            # real/synth split stays exact; tuplet boosting only redistributes
            # probability inside each group.
            for indices, mass in groups:
                index_tensor = torch.tensor(indices, dtype=torch.long)
                weights[index_tensor] *= mass / float(weights[index_tensor].sum())
        else:
            weights /= float(weights.sum())

        return torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=num_rows,
            replacement=True,
        )

    def train_dataloader(self):
        if self.train_set is None:
            raise RuntimeError(
                "Train dataset is not loaded. Call setup(stage='fit') before train_dataloader()."
            )
        train_sampler = self._build_train_sampler()
        return TimedDataLoader(
            self.train_set,
            batch_size=self.training_cfg.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=True,
            collate_fn=self.collator,
            fetch_timing_sink=self.train_fetch_sink,
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
                collate_fn=self.val_collator,
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
