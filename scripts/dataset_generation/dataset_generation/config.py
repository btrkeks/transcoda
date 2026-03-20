"""Configuration models and resolution helpers for dataset generation runs."""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from dataclasses import MISSING, dataclass, fields
from pathlib import Path

from scripts.dataset_generation.data_spec import (
    DEFAULT_DATA_SPEC_PATH,
    resolve_image_size_from_spec,
)
from scripts.dataset_generation.dataset_generation.worker_models import WorkerInitConfig


def coerce_boolish(value: object) -> bool:
    """Normalize CLI bool-like values, especially Fire string arguments."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean string: {value!r}")
    if isinstance(value, (int, float)):
        return bool(value)
    return bool(value)


@dataclass(frozen=True)
class FailurePolicySettings:
    """Resolved task timeout/retry settings for a run."""

    name: str
    task_timeout_seconds: int
    max_task_retries_timeout: int
    max_task_retries_expired: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "task_timeout_seconds": self.task_timeout_seconds,
            "max_task_retries_timeout": self.max_task_retries_timeout,
            "max_task_retries_expired": self.max_task_retries_expired,
        }


@dataclass(frozen=True)
class GenerationRunConfig:
    """User-facing run configuration for synthetic dataset generation."""

    kern_dirs: tuple[str, ...]
    dataset_preset: str | None = None
    output_dir: str = "data/datasets/train_medium"
    num_samples: int = 10_000_000
    target_accepted_samples: int | None = None
    max_scheduled_tasks: int | None = None
    image_width: int | None = None
    image_height: int | None = None
    data_spec_path: str = str(DEFAULT_DATA_SPEC_PATH)
    strict_data_spec: bool = True
    num_workers: int | None = None
    variants_per_file: int = 3
    adaptive_variants_enabled: bool = False
    overflow_truncation_enabled: bool = True
    overflow_truncation_max_trials: int = 24
    augment_seed: int | None = None
    render_pedals_enabled: bool = True
    render_pedals_probability: float = 0.20
    render_pedals_measures_probability: float = 0.3
    render_instrument_piano_enabled: bool = True
    render_instrument_piano_probability: float = 0.15
    render_sforzando_enabled: bool = True
    render_sforzando_probability: float = 0.20
    render_sforzando_per_note_probability: float = 0.03
    render_accent_enabled: bool = True
    render_accent_probability: float = 0.10
    render_accent_per_note_probability: float = 0.015
    render_tempo_enabled: bool = True
    render_tempo_probability: float = 0.12
    render_tempo_include_mm_probability: float = 0.35
    render_hairpins_enabled: bool = True
    render_hairpins_probability: float = 0.25
    render_hairpins_max_spans: int = 2
    render_dynamic_marks_enabled: bool = True
    render_dynamic_marks_probability: float = 0.15
    render_dynamic_marks_min_count: int = 1
    render_dynamic_marks_max_count: int = 2
    courtesy_naturals_probability: float = 0.15
    disable_offline_image_augmentations: bool = False
    geom_x_squeeze_prob: float = 0.45
    geom_x_squeeze_min_scale: float = 0.70
    geom_x_squeeze_max_scale: float = 0.95
    geom_x_squeeze_apply_in_conservative: bool = True
    geom_x_squeeze_preview_force_scale: float | None = None
    target_min_systems: int | None = None
    target_max_systems: int | None = None
    render_layout_profile: str = "default"
    prefilter_min_non_empty_lines: int | None = None
    prefilter_max_non_empty_lines: int | None = None
    prefilter_min_measure_count: int | None = None
    prefilter_max_measure_count: int | None = None
    progress_enabled: bool = True
    progress_update_interval_seconds: int = 30
    profile_enabled: bool = False
    profile_out_dir: str | None = None
    profile_log_every: int = 100
    profile_sample_limit: int = 500
    profile_capture_per_sample: bool = False
    failure_policy: str = "balanced"
    resume_mode: str = "auto"
    quarantine_in: str | None = None
    quarantine_out: str | None = None
    artifacts_out_dir: str | None = None
    start_method: str = "auto"
    quiet: bool = False

    @property
    def requested_num_samples(self) -> int:
        return int(self.num_samples)

    @property
    def effective_num_samples(self) -> int:
        if not self.profile_enabled:
            return int(self.num_samples)
        return min(int(self.num_samples), max(1, int(self.profile_sample_limit)))

    @property
    def effective_target_accepted_samples(self) -> int | None:
        if self.target_accepted_samples is None:
            return None
        if not self.profile_enabled:
            return int(self.target_accepted_samples)
        return min(
            int(self.target_accepted_samples),
            max(1, int(self.profile_sample_limit)),
        )

    @property
    def effective_max_scheduled_tasks(self) -> int | None:
        if self.max_scheduled_tasks is None:
            return None
        if not self.profile_enabled:
            return int(self.max_scheduled_tasks)
        return min(
            int(self.max_scheduled_tasks),
            max(1, int(self.profile_sample_limit)),
        )


DATASET_PRESET_NAMES: tuple[str, ...] = (
    "legacy_cloud_default",
    "target_system_polish_clean",
    "target_system_polish_full_aug",
    "target_system_legacy_5_6",
    "ablation_no_render_only_aug",
    "ablation_no_render_or_gt_aug",
    "ablation_no_offline_image_aug",
)

_RENDER_AUGMENTATIONS_DISABLED = {
    "render_pedals_enabled": False,
    "render_instrument_piano_enabled": False,
    "render_sforzando_enabled": False,
    "render_accent_enabled": False,
    "render_tempo_enabled": False,
    "render_hairpins_enabled": False,
    "render_dynamic_marks_enabled": False,
}

_RENDER_AUGMENTATIONS_ENABLED = {
    "render_pedals_enabled": True,
    "render_instrument_piano_enabled": True,
    "render_sforzando_enabled": True,
    "render_accent_enabled": True,
    "render_tempo_enabled": True,
    "render_hairpins_enabled": True,
    "render_dynamic_marks_enabled": True,
}

_LEGACY_CLOUD_DEFAULT_PRESET = {
    "adaptive_variants_enabled": True,
    "overflow_truncation_max_trials": 24,
    "failure_policy": "balanced",
    "courtesy_naturals_probability": 0.15,
    "disable_offline_image_augmentations": False,
    "render_pedals_probability": 0.20,
    "render_pedals_measures_probability": 0.3,
    "render_instrument_piano_probability": 0.15,
    "render_sforzando_probability": 0.20,
    "render_sforzando_per_note_probability": 0.03,
    "render_accent_probability": 0.10,
    "render_accent_per_note_probability": 0.015,
    "render_tempo_probability": 0.12,
    "render_tempo_include_mm_probability": 0.35,
    "render_hairpins_probability": 0.25,
    "render_hairpins_max_spans": 2,
    "render_dynamic_marks_probability": 0.15,
    "render_dynamic_marks_min_count": 1,
    "render_dynamic_marks_max_count": 2,
    **_RENDER_AUGMENTATIONS_ENABLED,
}

_TARGET_SYSTEM_POLISH_COMMON = {
    "target_accepted_samples": 50_000,
    "variants_per_file": 5,
    "adaptive_variants_enabled": False,
    "target_min_systems": 5,
    "target_max_systems": 7,
    "render_layout_profile": "polish_5_6_systems",
    "prefilter_min_measure_count": 20,
    "prefilter_max_measure_count": 32,
    "failure_policy": "balanced",
    "overflow_truncation_max_trials": 48,
    "courtesy_naturals_probability": 0.15,
}

_DATASET_PRESET_OVERRIDES: dict[str, dict[str, object]] = {
    "legacy_cloud_default": _LEGACY_CLOUD_DEFAULT_PRESET,
    "target_system_polish_clean": {
        "output_dir": "data/datasets/train_target_system_measureband_20_32_5_7_clean",
        "disable_offline_image_augmentations": True,
        **_TARGET_SYSTEM_POLISH_COMMON,
        **_RENDER_AUGMENTATIONS_DISABLED,
    },
    "target_system_polish_full_aug": {
        "output_dir": "data/datasets/train_target_system_measureband_20_32_5_7_full_augmented",
        "disable_offline_image_augmentations": False,
        **_TARGET_SYSTEM_POLISH_COMMON,
        **_RENDER_AUGMENTATIONS_ENABLED,
    },
    "target_system_legacy_5_6": {
        "output_dir": "data/datasets/train_target_system_5_6",
        "target_accepted_samples": 50_000,
        "variants_per_file": 3,
        "adaptive_variants_enabled": True,
        "target_min_systems": 5,
        "target_max_systems": 6,
        "render_layout_profile": "target_5_6_systems",
        "prefilter_min_non_empty_lines": 360,
        "prefilter_max_non_empty_lines": 390,
        "failure_policy": "balanced",
        "overflow_truncation_max_trials": 48,
        "disable_offline_image_augmentations": False,
        "courtesy_naturals_probability": 0.15,
        **_RENDER_AUGMENTATIONS_DISABLED,
    },
    "ablation_no_render_only_aug": {
        **_RENDER_AUGMENTATIONS_DISABLED,
    },
    "ablation_no_render_or_gt_aug": {
        "courtesy_naturals_probability": 0.0,
        **_RENDER_AUGMENTATIONS_DISABLED,
    },
    "ablation_no_offline_image_aug": {
        "disable_offline_image_augmentations": True,
    },
}


def _generation_run_default_values() -> dict[str, object]:
    defaults: dict[str, object] = {}
    for field_def in fields(GenerationRunConfig):
        if field_def.name == "kern_dirs":
            continue
        if field_def.default is not MISSING:
            defaults[field_def.name] = field_def.default
            continue
        if field_def.default_factory is not MISSING:
            defaults[field_def.name] = field_def.default_factory()
    return defaults


def resolve_dataset_preset_overrides(dataset_preset: str | None) -> dict[str, object]:
    """Return a copy of the preset defaults for a named dataset preset."""
    if dataset_preset is None:
        return {}
    normalized = str(dataset_preset).strip()
    if normalized not in _DATASET_PRESET_OVERRIDES:
        allowed = ", ".join(DATASET_PRESET_NAMES)
        raise ValueError(f"dataset_preset must be one of: {allowed}")
    return dict(_DATASET_PRESET_OVERRIDES[normalized])


def build_generation_run_config(
    *,
    kern_dirs: tuple[str, ...],
    **overrides: object,
) -> GenerationRunConfig:
    """Construct a run config by applying preset defaults then explicit overrides."""
    dataset_preset = overrides.get("dataset_preset")
    resolved_values = _generation_run_default_values()
    resolved_values["kern_dirs"] = tuple(kern_dirs)
    resolved_values.update(resolve_dataset_preset_overrides(dataset_preset if isinstance(dataset_preset, str) else None))
    if dataset_preset is not None:
        resolved_values["dataset_preset"] = dataset_preset
    for key, value in overrides.items():
        if key == "kern_dirs":
            continue
        if value is not None:
            resolved_values[key] = value
    return GenerationRunConfig(**resolved_values)


def resolve_failure_policy(failure_policy: str) -> FailurePolicySettings:
    """Resolve named failure policy into concrete timeout/retry settings."""
    normalized = str(failure_policy).strip().lower()
    policy_map = {
        "throughput": FailurePolicySettings(
            name="throughput",
            task_timeout_seconds=5,
            max_task_retries_timeout=0,
            max_task_retries_expired=0,
        ),
        "balanced": FailurePolicySettings(
            name="balanced",
            task_timeout_seconds=30,
            max_task_retries_timeout=1,
            max_task_retries_expired=0,
        ),
        "coverage": FailurePolicySettings(
            name="coverage",
            task_timeout_seconds=60,
            max_task_retries_timeout=1,
            max_task_retries_expired=1,
        ),
    }
    if normalized not in policy_map:
        raise ValueError("failure_policy must be one of: throughput, balanced, coverage")
    return policy_map[normalized]


def load_quarantine_list(quarantine_in: str | None) -> list[str]:
    """Load a quarantine file list from JSON payload."""
    if not quarantine_in:
        return []
    source_path = Path(quarantine_in)
    if not source_path.exists():
        raise ValueError(f"quarantine input file does not exist: {source_path}")
    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        files = payload.get("files")
        if not isinstance(files, list):
            raise ValueError("quarantine input JSON object must include a 'files' list")
        return [str(item) for item in files]
    if isinstance(payload, list):
        return [str(item) for item in payload]
    raise ValueError("quarantine input must be a JSON list or object with a 'files' list")


def resolve_start_method(start_method: str) -> str:
    normalized = str(start_method).strip().lower()
    if normalized == "auto":
        return "fork" if sys.platform.startswith("linux") else "spawn"
    if normalized not in {"spawn", "fork", "forkserver"}:
        raise ValueError("start_method must be one of: spawn, fork, forkserver, auto")
    if normalized == "fork" and not sys.platform.startswith("linux"):
        raise ValueError("start_method='fork' is only supported on Linux in this project")
    return normalized


def configure_start_method(start_method: str) -> str:
    resolved = resolve_start_method(start_method)
    current = mp.get_start_method(allow_none=True)
    if current is None:
        mp.set_start_method(resolved)
        return resolved
    if current != resolved:
        raise RuntimeError(
            f"multiprocessing start method already set to '{current}', requested '{resolved}'"
        )
    return current


def resolve_kern_dirs(kern_dirs: tuple[str, ...]) -> list[Path]:
    """Resolve input directory list, auto-discovering defaults when omitted."""
    if not kern_dirs:
        kern_dir_paths = sorted(Path("data/interim/train").glob("*/3_normalized"))
        if not kern_dir_paths:
            raise ValueError("No normalized directories found in data/interim/train/*/3_normalized")
        return kern_dir_paths
    return [Path(path) for path in kern_dirs]


def resolve_worker_config(config: GenerationRunConfig) -> WorkerInitConfig:
    """Build and validate worker initialization config."""
    image_width, image_height = resolve_image_size_from_spec(
        image_width=config.image_width,
        image_height=config.image_height,
        data_spec_path=config.data_spec_path,
        strict_data_spec=config.strict_data_spec,
    )
    worker_config = WorkerInitConfig(
        image_width=image_width,
        image_height=image_height,
        augment_seed=config.augment_seed,
        render_pedals_enabled=config.render_pedals_enabled,
        render_pedals_probability=config.render_pedals_probability,
        render_pedals_measures_probability=config.render_pedals_measures_probability,
        render_instrument_piano_enabled=config.render_instrument_piano_enabled,
        render_instrument_piano_probability=config.render_instrument_piano_probability,
        render_sforzando_enabled=config.render_sforzando_enabled,
        render_sforzando_probability=config.render_sforzando_probability,
        render_sforzando_per_note_probability=config.render_sforzando_per_note_probability,
        render_accent_enabled=config.render_accent_enabled,
        render_accent_probability=config.render_accent_probability,
        render_accent_per_note_probability=config.render_accent_per_note_probability,
        render_tempo_enabled=config.render_tempo_enabled,
        render_tempo_probability=config.render_tempo_probability,
        render_tempo_include_mm_probability=config.render_tempo_include_mm_probability,
        render_hairpins_enabled=config.render_hairpins_enabled,
        render_hairpins_probability=config.render_hairpins_probability,
        render_hairpins_max_spans=config.render_hairpins_max_spans,
        render_dynamic_marks_enabled=config.render_dynamic_marks_enabled,
        render_dynamic_marks_probability=config.render_dynamic_marks_probability,
        render_dynamic_marks_min_count=config.render_dynamic_marks_min_count,
        render_dynamic_marks_max_count=config.render_dynamic_marks_max_count,
        courtesy_naturals_probability=config.courtesy_naturals_probability,
        disable_offline_image_augmentations=config.disable_offline_image_augmentations,
        geom_x_squeeze_prob=config.geom_x_squeeze_prob,
        geom_x_squeeze_min_scale=config.geom_x_squeeze_min_scale,
        geom_x_squeeze_max_scale=config.geom_x_squeeze_max_scale,
        geom_x_squeeze_apply_in_conservative=config.geom_x_squeeze_apply_in_conservative,
        geom_x_squeeze_preview_force_scale=config.geom_x_squeeze_preview_force_scale,
        target_min_systems=config.target_min_systems,
        target_max_systems=config.target_max_systems,
        render_layout_profile=config.render_layout_profile,
        overflow_truncation_enabled=config.overflow_truncation_enabled,
        overflow_truncation_max_trials=config.overflow_truncation_max_trials,
        profile_enabled=config.profile_enabled,
    )
    worker_config.validate()
    return worker_config
