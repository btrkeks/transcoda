import torch

from src.training.lightning_module import SMTTrainer


class _MetricStub:
    def __init__(self, values: dict[str, torch.Tensor]):
        self.values = values
        self.compute_calls = 0
        self.reset_calls = 0

    def compute(self):
        self.compute_calls += 1
        return self.values

    def reset(self):
        self.reset_calls += 1


class _ValidationEndStub:
    def __init__(self):
        self._val_set_names = ["synth", "polish"]
        self._validation_metric_prefix = "val"
        self.val_metrics_by_set = {
            "synth": _MetricStub({"SER": torch.tensor(0.9), "CER": torch.tensor(1.2)}),
            "polish": _MetricStub(
                {
                    "SER": torch.tensor(0.5),
                    "CER": torch.tensor(0.8),
                    "CER_no_ties_beams": torch.tensor(0.7),
                }
            ),
        }
        self._val_batches_seen_by_set = {"synth": 0, "polish": 3}
        self._compute_omr_ned = False
        self._omr_ned_tracker = None
        self.logged = []

    def log(self, name, value, **kwargs):
        self.logged.append((name, value, kwargs))

    def _validation_set_metric_name(self, set_name, metric_name):
        return SMTTrainer._validation_set_metric_name(self, set_name, metric_name)

    def _validation_aggregate_metric_name(self, metric_name):
        return SMTTrainer._validation_aggregate_metric_name(self, metric_name)


def test_on_validation_epoch_end_skips_inactive_sets_and_logs_pass_flags():
    stub = _ValidationEndStub()
    SMTTrainer.on_validation_epoch_end(stub)

    logged_names = [name for name, _, _ in stub.logged]
    assert "val/polish/SER" in logged_names
    assert "val/polish/CER" in logged_names
    assert "val/polish/CER_no_ties_beams" in logged_names
    assert "val/synth/SER" not in logged_names
    assert "val/synth/CER" not in logged_names
    assert "val/aggregate/SER" in logged_names
    assert "val/aggregate/active_set_count" in logged_names
    assert "val/aggregate/is_full_pass" in logged_names

    synth_metric = stub.val_metrics_by_set["synth"]
    polish_metric = stub.val_metrics_by_set["polish"]
    assert synth_metric.compute_calls == 0
    assert polish_metric.compute_calls == 1
    assert synth_metric.reset_calls == 1
    assert polish_metric.reset_calls == 1


def test_on_validation_epoch_end_can_log_to_final_validation_namespace():
    stub = _ValidationEndStub()
    stub._validation_metric_prefix = "final_val"

    SMTTrainer.on_validation_epoch_end(stub)

    logged_names = [name for name, _, _ in stub.logged]
    assert "final_val/polish/SER" in logged_names
    assert "final_val/polish/CER" in logged_names
    assert "final_val/aggregate/SER" in logged_names
    assert "final_val/aggregate/active_set_count" in logged_names
    assert "val/polish/SER" not in logged_names
