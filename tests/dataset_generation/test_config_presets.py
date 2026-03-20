from pathlib import Path

from scripts.dataset_generation.dataset_generation.config import (
    build_generation_run_config,
)
from scripts.dataset_generation.dataset_generation.resumable_dataset import (
    compute_generation_config_fingerprint,
)


def test_build_generation_run_config_applies_legacy_cloud_default_preset():
    config = build_generation_run_config(
        kern_dirs=("data/interim/train/pdmx/3_normalized",),
        dataset_preset="legacy_cloud_default",
    )

    assert config.dataset_preset == "legacy_cloud_default"
    assert config.adaptive_variants_enabled is True
    assert config.failure_policy == "balanced"
    assert config.overflow_truncation_max_trials == 24
    assert config.disable_offline_image_augmentations is False
    assert config.render_pedals_enabled is True
    assert config.render_hairpins_enabled is True
    assert config.render_dynamic_marks_enabled is True
    assert config.courtesy_naturals_probability == 0.15


def test_build_generation_run_config_applies_target_system_polish_clean_preset():
    config = build_generation_run_config(
        kern_dirs=("data/interim/train/pdmx/3_normalized",),
        dataset_preset="target_system_polish_clean",
    )

    assert config.output_dir == "data/datasets/train_target_system_measureband_20_32_5_7_clean"
    assert config.target_accepted_samples == 50_000
    assert config.variants_per_file == 5
    assert config.adaptive_variants_enabled is False
    assert config.target_min_systems == 5
    assert config.target_max_systems == 7
    assert config.render_layout_profile == "polish_5_6_systems"
    assert config.prefilter_min_measure_count == 20
    assert config.prefilter_max_measure_count == 32
    assert config.disable_offline_image_augmentations is True
    assert config.render_pedals_enabled is False
    assert config.render_tempo_enabled is False


def test_build_generation_run_config_applies_render_and_gt_ablation_presets():
    render_only = build_generation_run_config(
        kern_dirs=("data/interim/train/pdmx/3_normalized",),
        dataset_preset="ablation_no_render_only_aug",
    )
    render_and_gt = build_generation_run_config(
        kern_dirs=("data/interim/train/pdmx/3_normalized",),
        dataset_preset="ablation_no_render_or_gt_aug",
    )

    assert render_only.render_pedals_enabled is False
    assert render_only.render_hairpins_enabled is False
    assert render_only.render_dynamic_marks_enabled is False
    assert render_only.courtesy_naturals_probability == 0.15

    assert render_and_gt.render_pedals_enabled is False
    assert render_and_gt.render_hairpins_enabled is False
    assert render_and_gt.render_dynamic_marks_enabled is False
    assert render_and_gt.courtesy_naturals_probability == 0.0


def test_build_generation_run_config_applies_offline_ablation_preset():
    config = build_generation_run_config(
        kern_dirs=("data/interim/train/pdmx/3_normalized",),
        dataset_preset="ablation_no_offline_image_aug",
    )

    assert config.disable_offline_image_augmentations is True
    assert config.render_pedals_enabled is True
    assert config.courtesy_naturals_probability == 0.15


def test_build_generation_run_config_explicit_overrides_win_over_preset():
    config = build_generation_run_config(
        kern_dirs=("data/interim/train/pdmx/3_normalized",),
        dataset_preset="ablation_no_render_only_aug",
        output_dir="data/datasets/custom_override",
        render_pedals_enabled=True,
        courtesy_naturals_probability=0.55,
    )

    assert config.output_dir == "data/datasets/custom_override"
    assert config.render_pedals_enabled is True
    assert config.courtesy_naturals_probability == 0.55


def test_generation_config_fingerprint_changes_with_dataset_preset_and_gt_knobs():
    kern_dirs = [Path("data/interim/train/pdmx/3_normalized")]
    base = build_generation_run_config(
        kern_dirs=("data/interim/train/pdmx/3_normalized",),
        dataset_preset="ablation_no_render_only_aug",
    )
    gt_variant = build_generation_run_config(
        kern_dirs=("data/interim/train/pdmx/3_normalized",),
        dataset_preset="ablation_no_render_or_gt_aug",
    )

    base_fingerprint = compute_generation_config_fingerprint(
        config=base,
        resolved_kern_dirs=kern_dirs,
    )
    gt_variant_fingerprint = compute_generation_config_fingerprint(
        config=gt_variant,
        resolved_kern_dirs=kern_dirs,
    )

    assert base_fingerprint != gt_variant_fingerprint
