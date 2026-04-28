import pytest

from src.config import experiment_config_from_dict


def test_training_limit_val_batches_accepts_fraction():
    config = experiment_config_from_dict(
        {
            "data": {
                "train_path": "./data/datasets/train_full",
                "validation_paths": {"synth": "./data/datasets/validation/synth"},
                "vocab_dir": "./vocab/bpe4k",
            },
            "checkpoint": {},
            "training": {"limit_val_batches": 0.25},
        }
    )

    assert config.training.limit_val_batches == 0.25


def test_progress_config_accepts_valid_values():
    config = experiment_config_from_dict(
        {
            "data": {
                "train_path": "./data/datasets/train_full",
                "validation_paths": {"synth": "./data/datasets/validation/synth"},
                "vocab_dir": "./vocab/bpe4k",
            },
            "checkpoint": {},
            "training": {
                "progress_train_interval_seconds": 15.0,
                "progress_train_every_n_steps": 5,
                "progress_val_percent_interval": 20,
            },
        }
    )

    assert config.training.progress_train_interval_seconds == 15.0
    assert config.training.progress_train_every_n_steps == 5
    assert config.training.progress_val_percent_interval == 20


def test_freeze_encoder_steps_defaults_to_zero():
    config = experiment_config_from_dict(
        {
            "data": {
                "train_path": "./data/datasets/train_full",
                "validation_paths": {"synth": "./data/datasets/validation/synth"},
                "vocab_dir": "./vocab/bpe4k",
            },
            "checkpoint": {},
        }
    )

    assert config.training.freeze_encoder_steps == 0


def test_freeze_encoder_steps_accepts_positive_value():
    config = experiment_config_from_dict(
        {
            "data": {
                "train_path": "./data/datasets/train_full",
                "validation_paths": {"synth": "./data/datasets/validation/synth"},
                "vocab_dir": "./vocab/bpe4k",
            },
            "checkpoint": {},
            "training": {"freeze_encoder_steps": 123},
        }
    )

    assert config.training.freeze_encoder_steps == 123


def test_freeze_encoder_steps_rejects_negative_value():
    with pytest.raises(ValueError, match="freeze_encoder_steps"):
        experiment_config_from_dict(
            {
                "data": {
                    "train_path": "./data/datasets/train_full",
                    "validation_paths": {"synth": "./data/datasets/validation/synth"},
                    "vocab_dir": "./vocab/bpe4k",
                },
                "checkpoint": {},
                "training": {"freeze_encoder_steps": -1},
            }
        )


def test_semantic_constraints_require_grammar_constraints():
    with pytest.raises(ValueError, match="require training.use_grammar_constraints=true"):
        experiment_config_from_dict(
            {
                "data": {
                    "train_path": "./data/datasets/train_full",
                    "validation_paths": {"synth": "./data/datasets/validation/synth"},
                    "vocab_dir": "./vocab/bpe4k",
                },
                "checkpoint": {},
                "training": {"use_spine_structure_constraints": True},
            }
        )

    with pytest.raises(ValueError, match="require training.use_grammar_constraints=true"):
        experiment_config_from_dict(
            {
                "data": {
                    "train_path": "./data/datasets/train_full",
                    "validation_paths": {"synth": "./data/datasets/validation/synth"},
                    "vocab_dir": "./vocab/bpe4k",
                },
                "checkpoint": {},
                "training": {"use_interpretation_transition_constraints": True},
            }
        )

    with pytest.raises(ValueError, match="require training.use_grammar_constraints=true"):
        experiment_config_from_dict(
            {
                "data": {
                    "train_path": "./data/datasets/train_full",
                    "validation_paths": {"synth": "./data/datasets/validation/synth"},
                    "vocab_dir": "./vocab/bpe4k",
                },
                "checkpoint": {},
                "training": {
                    "use_spine_structure_constraints": False,
                    "use_rhythm_constraints": True,
                },
            }
        )


def test_interpretation_transition_config_rejects_invalid_values():
    base = {
        "data": {
            "train_path": "./data/datasets/train_full",
            "validation_paths": {"synth": "./data/datasets/validation/synth"},
            "vocab_dir": "./vocab/bpe4k",
        },
        "checkpoint": {},
    }

    with pytest.raises(ValueError, match="non_spine_bonus"):
        experiment_config_from_dict(
            {
                **base,
                "training": {
                    "use_grammar_constraints": True,
                    "interpretation_transition_non_spine_bonus": -0.1,
                },
            }
        )

    with pytest.raises(ValueError, match="null_interp_bonus"):
        experiment_config_from_dict(
            {
                **base,
                "training": {
                    "use_grammar_constraints": True,
                    "interpretation_transition_null_interp_bonus": -0.1,
                },
            }
        )

    with pytest.raises(ValueError, match="data_start_penalty"):
        experiment_config_from_dict(
            {
                **base,
                "training": {
                    "use_grammar_constraints": True,
                    "interpretation_transition_data_start_penalty": 0.1,
                },
            }
        )

    with pytest.raises(ValueError, match="barline_start_penalty"):
        experiment_config_from_dict(
            {
                **base,
                "training": {
                    "use_grammar_constraints": True,
                    "interpretation_transition_barline_start_penalty": 0.1,
                },
            }
        )


def test_progress_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="progress_train_interval_seconds"):
        experiment_config_from_dict(
            {
                "data": {
                    "train_path": "./data/datasets/train_full",
                    "validation_paths": {"synth": "./data/datasets/validation/synth"},
                    "vocab_dir": "./vocab/bpe4k",
                },
                "checkpoint": {},
                "training": {"progress_train_interval_seconds": 0},
            }
        )


def test_runaway_monitor_config_accepts_valid_values():
    config = experiment_config_from_dict(
        {
            "data": {
                "train_path": "./data/datasets/train_full",
                "validation_paths": {"synth": "./data/datasets/validation/synth"},
                "vocab_dir": "./vocab/bpe4k",
            },
            "checkpoint": {},
            "training": {
                "runaway_monitor_strictness": "strict",
                "runaway_monitor_max_len_ratio": 1.5,
                "runaway_monitor_repeat_ngram_size": 3,
                "runaway_monitor_repeat_ngram_max_occurrences": 4,
                "runaway_monitor_max_identical_line_run": 5,
                "runaway_monitor_flag_no_eos_at_max_length": False,
            },
        }
    )

    assert config.training.runaway_monitor_strictness == "strict"
    assert config.training.runaway_monitor_max_len_ratio == 1.5
    assert config.training.runaway_monitor_repeat_ngram_size == 3
    assert config.training.runaway_monitor_repeat_ngram_max_occurrences == 4
    assert config.training.runaway_monitor_max_identical_line_run == 5
    assert config.training.runaway_monitor_flag_no_eos_at_max_length is False


def test_runaway_monitor_config_rejects_invalid_values():
    base = {
        "data": {
            "train_path": "./data/datasets/train_full",
            "validation_paths": {"synth": "./data/datasets/validation/synth"},
            "vocab_dir": "./vocab/bpe4k",
        },
        "checkpoint": {},
    }

    with pytest.raises(ValueError, match="runaway_monitor_strictness"):
        experiment_config_from_dict(
            {
                **base,
                "training": {"runaway_monitor_strictness": "aggressive"},
            }
        )

    with pytest.raises(ValueError, match="runaway_monitor_max_len_ratio"):
        experiment_config_from_dict(
            {
                **base,
                "training": {"runaway_monitor_max_len_ratio": 1.0},
            }
        )

    with pytest.raises(ValueError, match="runaway_monitor_repeat_ngram_size"):
        experiment_config_from_dict(
            {
                **base,
                "training": {"runaway_monitor_repeat_ngram_size": 1},
            }
        )

    with pytest.raises(ValueError, match="runaway_monitor_repeat_ngram_max_occurrences"):
        experiment_config_from_dict(
            {
                **base,
                "training": {"runaway_monitor_repeat_ngram_max_occurrences": 1},
            }
        )

    with pytest.raises(ValueError, match="runaway_monitor_max_identical_line_run"):
        experiment_config_from_dict(
            {
                **base,
                "training": {"runaway_monitor_max_identical_line_run": 1},
            }
        )


def test_tiered_validation_rejects_invalid_full_interval():
    with pytest.raises(ValueError, match="full_validation_every_n_steps"):
        experiment_config_from_dict(
            {
                "data": {
                    "train_path": "./data/datasets/train_full",
                    "validation_paths": {"synth": "./data/datasets/validation/synth"},
                    "vocab_dir": "./vocab/bpe4k",
                },
                "checkpoint": {},
                "training": {"full_validation_every_n_steps": 0},
            }
        )


def test_frequent_validation_subset_config_accepts_valid_values():
    config = experiment_config_from_dict(
        {
            "data": {
                "train_path": "./data/datasets/train_full",
                "validation_paths": {
                    "synth": "./data/datasets/validation/synth",
                    "polish": "./data/datasets/validation/polish",
                },
                "vocab_dir": "./vocab/bpe4k",
            },
            "checkpoint": {},
            "training": {
                "frequent_validation_subset_sizes": {"synth": 256},
                "frequent_validation_subset_seed": 7,
            },
        }
    )

    assert config.training.frequent_validation_subset_sizes == {"synth": 256}
    assert config.training.frequent_validation_subset_seed == 7


def test_frequent_validation_subset_config_rejects_unknown_set_names():
    with pytest.raises(ValueError, match="frequent_validation_subset_sizes keys"):
        experiment_config_from_dict(
            {
                "data": {
                    "train_path": "./data/datasets/train_full",
                    "validation_paths": {"synth": "./data/datasets/validation/synth"},
                    "vocab_dir": "./vocab/bpe4k",
                },
                "checkpoint": {},
                "training": {
                    "frequent_validation_subset_sizes": {"polish": 32},
                },
            }
        )


def test_frequent_validation_subset_config_rejects_invalid_subset_sizes():
    with pytest.raises(ValueError, match="frequent_validation_subset_sizes values"):
        experiment_config_from_dict(
            {
                "data": {
                    "train_path": "./data/datasets/train_full",
                    "validation_paths": {"synth": "./data/datasets/validation/synth"},
                    "vocab_dir": "./vocab/bpe4k",
                },
                "checkpoint": {},
                "training": {
                    "frequent_validation_subset_sizes": {"synth": 0},
                },
            }
        )


def test_early_stopping_rejects_negative_min_delta():
    with pytest.raises(ValueError, match="early_stopping_min_delta"):
        experiment_config_from_dict(
            {
                "data": {
                    "train_path": "./data/datasets/train_full",
                    "validation_paths": {"synth": "./data/datasets/validation/synth"},
                    "vocab_dir": "./vocab/bpe4k",
                },
                "checkpoint": {},
                "training": {"early_stopping_min_delta": -0.1},
            }
        )


def test_config_list_defaults_are_not_shared_between_instances():
    cfg_a = experiment_config_from_dict(
        {
            "data": {
                "train_path": "./data/datasets/train_full",
                "validation_paths": {"synth": "./data/datasets/validation/synth"},
                "vocab_dir": "./vocab/bpe4k",
            },
            "checkpoint": {},
        }
    )
    cfg_b = experiment_config_from_dict(
        {
            "data": {
                "train_path": "./data/datasets/train_full",
                "validation_paths": {"synth": "./data/datasets/validation/synth"},
                "vocab_dir": "./vocab/bpe4k",
            },
            "checkpoint": {},
        }
    )

    cfg_a.training.frequent_validation_set_names.append("synth")
    cfg_a.checkpoint.tags.append("scale-study")

    assert cfg_b.training.frequent_validation_set_names == ["polish"]
    assert cfg_b.checkpoint.tags == []
