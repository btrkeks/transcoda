from scripts.dataset_generation.dataset_generation import cli


def test_cli_coerces_fire_style_boolean_strings(monkeypatch):
    captured = {}

    def fake_run_generation(config):
        captured["config"] = config
        return None

    monkeypatch.setattr(cli, "run_generation", fake_run_generation)

    cli.main(
        "data/interim/train/pdmx/3_normalized",
        adaptive_variants_enabled="false",
        disable_offline_image_augmentations="false",
        render_pedals_enabled="false",
        render_pedals_probability=0.4,
        progress_enabled="true",
        quiet="true",
    )

    config = captured["config"]
    assert config.adaptive_variants_enabled is False
    assert config.disable_offline_image_augmentations is False
    assert config.render_pedals_enabled is False
    assert config.render_pedals_probability == 0.4
    assert config.progress_enabled is True
    assert config.quiet is True


def test_cli_default_runtime_and_pedal_defaults(monkeypatch):
    captured = {}

    def fake_run_generation(config):
        captured["config"] = config
        return None

    monkeypatch.setattr(cli, "run_generation", fake_run_generation)

    cli.main("data/interim/train/pdmx/3_normalized", quiet="true")

    config = captured["config"]
    assert config.render_pedals_probability == 0.20
    assert config.failure_policy == "balanced"
    assert config.resume_mode == "auto"
    assert config.overflow_truncation_max_trials == 24


def test_cli_applies_dataset_preset_and_explicit_overrides(monkeypatch):
    captured = {}

    def fake_run_generation(config):
        captured["config"] = config
        return None

    monkeypatch.setattr(cli, "run_generation", fake_run_generation)

    cli.main(
        "data/interim/train/pdmx/3_normalized",
        dataset_preset="ablation_no_render_or_gt_aug",
        render_pedals_enabled="true",
        quiet="true",
    )

    config = captured["config"]
    assert config.dataset_preset == "ablation_no_render_or_gt_aug"
    assert config.render_pedals_enabled is True
    assert config.render_tempo_enabled is False
    assert config.courtesy_naturals_probability == 0.0
