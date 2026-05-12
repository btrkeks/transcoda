"""Centralized metric naming schema for training, validation, and test logging."""

from __future__ import annotations

TRAIN_PREFIX = "train"
VAL_PREFIX = "val"
VAL_AGGREGATE_PREFIX = "val/aggregate"
FINAL_VAL_PREFIX = "final_val"
FINAL_VAL_AGGREGATE_PREFIX = "final_val/aggregate"
TEST_PREFIX = "test"
VAL_SUBSET_SUFFIX = "_subset"

METRIC_SER = "SER"
METRIC_CER = "CER"
METRIC_LER = "LER"
METRIC_LOSS = "loss"

TRAIN_LOSS = f"{TRAIN_PREFIX}/{METRIC_LOSS}"
TRAIN_STAGE = f"{TRAIN_PREFIX}/stage"
VAL_AGGREGATE_SER = f"{VAL_AGGREGATE_PREFIX}/{METRIC_SER}"
DEFAULT_CHECKPOINT_MONITOR = VAL_AGGREGATE_SER


def train_metric(metric_name: str) -> str:
    return f"{TRAIN_PREFIX}/{metric_name}"


def validation_set_metric(prefix: str, set_name: str, metric_name: str) -> str:
    return f"{prefix}/{set_name}/{metric_name}"


def val_set_metric(set_name: str, metric_name: str) -> str:
    return validation_set_metric(VAL_PREFIX, set_name, metric_name)


def subset_val_set_name(set_name: str) -> str:
    return f"{set_name}{VAL_SUBSET_SUFFIX}"


def is_subset_val_set_name(set_name: str) -> bool:
    return set_name.endswith(VAL_SUBSET_SUFFIX)


def base_val_set_name(set_name: str) -> str:
    if is_subset_val_set_name(set_name):
        return set_name[: -len(VAL_SUBSET_SUFFIX)]
    return set_name


def val_subset_metric(set_name: str, metric_name: str) -> str:
    return val_set_metric(subset_val_set_name(set_name), metric_name)


def final_val_set_metric(set_name: str, metric_name: str) -> str:
    return validation_set_metric(FINAL_VAL_PREFIX, set_name, metric_name)


def final_val_aggregate_metric(metric_name: str) -> str:
    return f"{FINAL_VAL_AGGREGATE_PREFIX}/{metric_name}"


def val_aggregate_metric(metric_name: str) -> str:
    return f"{VAL_AGGREGATE_PREFIX}/{metric_name}"


def build_test_metric_key(metric_name: str) -> str:
    return f"{TEST_PREFIX}/{metric_name}"
