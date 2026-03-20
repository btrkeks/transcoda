from src.config import Checkpoint
from src.metrics_schema import (
    DEFAULT_CHECKPOINT_MONITOR,
    FINAL_VAL_PREFIX,
    TRAIN_LOSS,
    TRAIN_STAGE,
    base_val_set_name,
    build_test_metric_key,
    final_val_aggregate_metric,
    final_val_set_metric,
    is_subset_val_set_name,
    subset_val_set_name,
    validation_set_metric,
    val_aggregate_metric,
    val_set_metric,
    val_subset_metric,
)


def test_metric_schema_builders():
    assert TRAIN_LOSS == "train/loss"
    assert TRAIN_STAGE == "train/stage"
    assert FINAL_VAL_PREFIX == "final_val"
    assert validation_set_metric("final_val", "synth", "CER") == "final_val/synth/CER"
    assert val_set_metric("polish", "CER") == "val/polish/CER"
    assert final_val_set_metric("polish", "CER") == "final_val/polish/CER"
    assert subset_val_set_name("synth") == "synth_subset"
    assert is_subset_val_set_name("synth_subset") is True
    assert base_val_set_name("synth_subset") == "synth"
    assert val_subset_metric("synth", "CER") == "val/synth_subset/CER"
    assert val_aggregate_metric("SER") == "val/aggregate/SER"
    assert final_val_aggregate_metric("SER") == "final_val/aggregate/SER"
    assert build_test_metric_key("runaway_rate") == "test/runaway_rate"


def test_checkpoint_default_monitor_uses_schema_constant():
    checkpoint = Checkpoint()
    assert checkpoint.monitor == DEFAULT_CHECKPOINT_MONITOR
    assert checkpoint.monitor == "val/aggregate/SER"
