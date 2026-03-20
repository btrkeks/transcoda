import scripts.dataset_generation.dataset_generation.worker as worker
from scripts.dataset_generation.dataset_generation.config import build_generation_run_config
from scripts.dataset_generation.dataset_generation.config import resolve_worker_config


def test_render_only_ablation_skips_render_side_augmentations(monkeypatch):
    config = resolve_worker_config(
        build_generation_run_config(
            kern_dirs=("data/interim/train/pdmx/3_normalized",),
            dataset_preset="ablation_no_render_only_aug",
        )
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError("render-only augmentation should be disabled by preset")

    monkeypatch.setattr(worker, "apply_pedaling", _unexpected)
    monkeypatch.setattr(worker, "apply_instrument_label_piano", _unexpected)
    monkeypatch.setattr(worker, "apply_sforzandos", _unexpected)
    monkeypatch.setattr(worker, "apply_accents", _unexpected)
    monkeypatch.setattr(worker, "apply_tempo_markings", _unexpected)
    monkeypatch.setattr(worker, "apply_render_hairpins", _unexpected)
    monkeypatch.setattr(worker, "apply_render_dynamic_marks", _unexpected)

    render_transcription = worker._build_render_transcription("4c\n*-\n", config)

    assert render_transcription.startswith("**kern\n")
    assert render_transcription.endswith("4c\n*-\n")


def test_render_and_gt_ablation_skips_courtesy_naturals_label_mutation():
    config = resolve_worker_config(
        build_generation_run_config(
            kern_dirs=("data/interim/train/pdmx/3_normalized",),
            dataset_preset="ablation_no_render_or_gt_aug",
        )
    )

    assert worker._build_label_transcription("4cn\n*-\n", config) == "4cn\n*-\n"
