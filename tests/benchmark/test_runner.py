from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from PIL import Image
from rich.console import Console

from src.artifacts import DecodingSpec
from src.data.preprocessing import LayoutNormalizationConfig
from src.benchmark.conversion import ConversionResult
from src.benchmark.results import DatasetModelSummary, SampleMetric, summarize_overall
from src.benchmark.runner import (
    BenchmarkRunner,
    BenchmarkRunnerConfig,
    BenchmarkSample,
    MetricJob,
    compute_metric_job,
    discover_datasets,
    resolve_batch_size,
    resolve_metric_workers,
)
from src.evaluation.omr_ned import OMRNEDResult, is_musicdiff_available


def test_discover_datasets_ignores_runs(tmp_path: Path) -> None:
    (tmp_path / "_runs").mkdir()
    (tmp_path / "synth").mkdir()
    (tmp_path / "polish").mkdir()

    assert discover_datasets(str(tmp_path), ()) == ["polish", "synth"]


def test_summarize_overall_is_weighted() -> None:
    rows = [
        DatasetModelSummary("synth", "ours", "kern", 2, 0, 0, 100.0, 10.0, 20.0, cer=5.0),
        DatasetModelSummary("polish", "ours", "kern", 1, 1, 1, 0.0, 100.0, 100.0, cer=25.0),
    ]
    overall = summarize_overall(rows)
    assert len(overall) == 1
    assert overall[0].cer == pytest.approx((2 * 5.0 + 1 * 25.0) / 3)
    assert overall[0].tedn == pytest.approx((2 * 10.0 + 1 * 100.0) / 3)
    assert overall[0].omr_ned == pytest.approx((2 * 20.0 + 1 * 100.0) / 3)
    assert overall[0].num_metric_worker_failures == 1
    assert overall[0].num_conversion_failures == 1


def test_gold_conversion_failure_aborts(monkeypatch, tmp_path: Path) -> None:
    config = BenchmarkRunnerConfig(output_root=str(tmp_path))
    runner = BenchmarkRunner(config)
    samples = [BenchmarkSample(0, "000000", "", "4c")]
    monkeypatch.setattr(
        "src.benchmark.runner.convert_kern_to_musicxml",
        lambda *_args, **_kwargs: ConversionResult(musicxml=None, error="broken gold"),
    )

    with pytest.raises(RuntimeError, match="broken gold"):
        runner._convert_gold("synth", samples)


def test_gold_conversion_failure_can_be_skipped(monkeypatch, tmp_path: Path) -> None:
    config = BenchmarkRunnerConfig(output_root=str(tmp_path), skip_invalid_gold=True)
    runner = BenchmarkRunner(config)
    samples = [
        BenchmarkSample(0, "000000", "", "4c"),
        BenchmarkSample(1, "000001", "", "4d"),
    ]

    def fake_convert(text, *_args, **_kwargs):
        if text == "4c":
            return ConversionResult(musicxml="<score-partwise/>")
        return ConversionResult(
            musicxml=None,
            error="broken gold",
            stdout="line 12",
        )

    monkeypatch.setattr("src.benchmark.runner.convert_kern_to_musicxml", fake_convert)

    gold_xml, valid_samples, gold_xml_dir = runner._convert_gold("synth", samples)

    assert list(gold_xml) == ["000000"]
    assert [sample.sample_key for sample in valid_samples] == ["000000"]
    assert len(runner.invalid_gold_samples) == 1
    assert runner.invalid_gold_samples[0].sample_key == "000001"
    assert runner.invalid_gold_samples[0].diagnostics == "line 12"
    assert (gold_xml_dir / "000000.xml").exists()


def test_prediction_conversion_failure_maps_to_100(monkeypatch, tmp_path: Path) -> None:
    config = BenchmarkRunnerConfig(
        output_root=str(tmp_path),
        models=("legato",),
        skip_inference=True,
        metric_workers=1,
    )
    runner = BenchmarkRunner(config)
    dataset_name = "synth"
    model_dir = runner.run_dir / dataset_name / "legato"
    model_dir.mkdir(parents=True, exist_ok=True)
    raw_predictions_path = model_dir / "raw_predictions.jsonl"
    raw_predictions_path.write_text(
        json.dumps(
            {
                "sample_index": 0,
                "sample_key": "000000",
                "source": "",
                "raw_format": "abc",
                "prediction": "X:1\nbroken",
            }
        )
        + "\n"
    )
    samples = [BenchmarkSample(0, "000000", "", "4c")]
    gold_xml = {"000000": "<score-partwise><part id='P1'/></score-partwise>"}
    monkeypatch.setattr(
        runner,
        "_convert_gold",
        lambda *_args, **_kwargs: (gold_xml, samples, tmp_path / "gold_xml"),
    )
    monkeypatch.setattr(
        "src.benchmark.runner.convert_abc_to_musicxml",
        lambda *_args, **_kwargs: ConversionResult(musicxml=None, error="abc failed"),
    )

    summary = runner._evaluate_model(
        dataset_name=dataset_name,
        dataset=[],
        samples=samples,
        model_name="legato",
    )

    assert summary.num_conversion_failures == 1
    assert summary.tedn == 100.0
    assert summary.omr_ned == 100.0


def test_runner_does_not_resolve_abc2xml_for_ours_only(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "src.benchmark.runner.resolve_abc2xml_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not resolve abc2xml")),
    )

    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            output_root=str(tmp_path),
            models=("ours",),
        )
    )

    assert runner.abc2xml_command is None


def test_resolve_batch_size_auto_uses_cuda_default() -> None:
    assert resolve_batch_size("auto", torch.device("cuda")) == 4


def test_resolve_batch_size_auto_uses_cpu_default() -> None:
    assert resolve_batch_size("auto", torch.device("cpu")) == 1


def test_resolve_metric_workers_prefers_slurm(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "12")
    monkeypatch.setattr("os.cpu_count", lambda: 2)
    assert resolve_metric_workers("auto") == 8


def test_resolve_metric_workers_falls_back_to_cpu_count(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setattr("os.cpu_count", lambda: 6)
    assert resolve_metric_workers("auto") == 6


def test_run_inference_logs_progress(monkeypatch, tmp_path: Path) -> None:
    messages: list[str] = []
    monkeypatch.setattr("src.benchmark.runner.console.log", lambda message: messages.append(str(message)))

    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            output_root=str(tmp_path),
            models=("ours",),
            device="cpu",
            batch_size=2,
        )
    )
    samples = [
        BenchmarkSample(0, "000000", "", "4c"),
        BenchmarkSample(1, "000001", "", "4d"),
        BenchmarkSample(2, "000002", "", "4e"),
    ]

    class FakeAdapter:
        raw_format = "kern"

        def predict_batch(self, images):
            return [f"pred-{idx}" for idx, _ in enumerate(images, start=1)]

    dataset = [{"image": Image.new("RGB", (4, 4), "white")} for _ in samples]
    predictions = runner._run_inference(
        dataset,
        samples,
        FakeAdapter(),
        dataset_name="synth",
        model_name="ours",
    )

    assert len(predictions) == 3
    assert any("inference-start dataset=synth model=ours" in message for message in messages)
    batch_logs = [message for message in messages if "inference target=synth/ours" in message]
    assert len(batch_logs) == 2
    assert any("progress=2/3" in message and "batch=1/2" in message for message in batch_logs)
    assert any("progress=3/3" in message and "batch=2/2" in message for message in batch_logs)


def test_run_inference_uses_resolved_batch_size(monkeypatch, tmp_path: Path) -> None:
    runner = BenchmarkRunner(BenchmarkRunnerConfig(output_root=str(tmp_path), models=("ours",), batch_size=2))
    samples = [
        BenchmarkSample(0, "000000", "", "4c"),
        BenchmarkSample(1, "000001", "", "4d"),
        BenchmarkSample(2, "000002", "", "4e"),
    ]

    batch_sizes: list[int] = []

    class FakeAdapter:
        raw_format = "kern"

        def predict_batch(self, images):
            batch_sizes.append(len(images))
            return [f"pred-{idx}" for idx, _ in enumerate(images)]

    dataset = [{"image": Image.new("RGB", (4, 4), "white")} for _ in samples]
    runner._run_inference(
        dataset,
        samples,
        FakeAdapter(),
        dataset_name="synth",
        model_name="ours",
    )

    assert batch_sizes == [2, 1]


def test_run_inference_writes_profile_artifacts(tmp_path: Path) -> None:
    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            output_root=str(tmp_path / "outputs"),
            models=("ours",),
            batch_size=2,
            profile=True,
            profile_warmup_batches=1,
            profile_max_batches=1,
        )
    )
    samples = [
        BenchmarkSample(0, "000000", "", "4c"),
        BenchmarkSample(1, "000001", "", "4d"),
        BenchmarkSample(2, "000002", "", "4e"),
    ]

    class FakeAdapter:
        raw_format = "kern"

        def __init__(self) -> None:
            self._collect_profile = False
            self._trace_path = None
            self._last_profile = None

        def prepare_profile(self, *, collect_profile, trace_path=None):
            self._collect_profile = collect_profile
            self._trace_path = trace_path

        def predict_batch(self, images):
            self._last_profile = {
                "preprocess_ms": 1.0,
                "tensor_setup_ms": 2.0,
                "constraint_bundle_ms": 3.0,
                "generate_ms": 10.0,
                "finalize_ms": 4.0,
                "generated_tokens": 12,
                "generated_tokens_per_sample": [6] * len(images),
                "generate_tokens_per_second": 1200.0,
                "constraint_stats": {
                    "grammar": {
                        "total_ms": 5.0,
                        "matcher_state_advance_ms": 2.0,
                        "bitmask_fill_ms": 1.0,
                        "bitmask_apply_ms": 2.0,
                        "calls": 7,
                        "rows_processed": len(images),
                        "externally_finished_rows": 0,
                    },
                    "semantic": {
                        "total_ms": 3.0,
                        "advance_row_ms": 1.0,
                        "mask_row_ms": 2.0,
                        "calls": 7,
                        "rows_processed": len(images),
                        "inactive_rows": 0,
                        "terminated_rows": 0,
                    },
                },
                "trace_path": None if self._trace_path is None else str(self._trace_path),
            }
            return [f"pred-{idx}" for idx, _ in enumerate(images, start=1)]

        def consume_last_profile(self):
            return self._last_profile

    dataset = [{"image": Image.new("RGB", (4, 4), "white")} for _ in samples]
    runner._run_inference(
        dataset,
        samples,
        FakeAdapter(),
        dataset_name="synth",
        model_name="ours",
    )

    profile_dir = runner.run_dir / "synth" / "ours" / "profile"
    batches_path = profile_dir / "profile_batches.jsonl"
    summary_path = profile_dir / "profile_summary.json"

    assert batches_path.exists()
    assert summary_path.exists()

    rows = [json.loads(line) for line in batches_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())

    assert len(rows) == 1
    assert rows[0]["batch_index"] == 1
    assert rows[0]["generated_tokens"] == 12
    assert summary["metadata"]["profiled_batches"] == 1
    assert summary["throughput"]["tokens_per_second"] > 0.0
    assert summary["constraint_processors"]["grammar"]["total_ms"]["mean"] == 5.0
    assert summary["comparison"] is None


def test_run_inference_profile_summary_compares_to_opposite_constraint_run(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            output_root=str(output_root),
            models=("ours",),
            batch_size=2,
            limit=3,
            profile=True,
            profile_warmup_batches=0,
            profile_max_batches=1,
        )
    )
    counterpart_dir = output_root / "20240101T000000Z" / "synth" / "ours" / "profile"
    counterpart_dir.mkdir(parents=True, exist_ok=True)
    counterpart_summary = {
        "metadata": {
            "created_at": "2024-01-01T00:00:00+00:00",
            "dataset_name": "synth",
            "model_name": "ours",
            "constraints_enabled": False,
            "checkpoint_path": "weights/GrandStaff/smt-model.ckpt",
            "resolved_batch_size": 2,
            "limit": 3,
        },
        "throughput": {
            "samples_per_second": 8.0,
            "tokens_per_second": 80.0,
        },
        "batch_latency_ms": {"mean": 100.0, "median": 100.0, "p95": 100.0, "total": 100.0},
        "stages_ms": {
            "generate": {"mean": 80.0, "median": 80.0, "p95": 80.0, "total": 80.0},
        },
        "constraint_processors": {
            "grammar": {"total_ms": {"mean": 0.0, "median": 0.0, "p95": 0.0, "total": 0.0}},
            "semantic": {"total_ms": {"mean": 0.0, "median": 0.0, "p95": 0.0, "total": 0.0}},
        },
    }
    (counterpart_dir / "profile_summary.json").write_text(json.dumps(counterpart_summary))

    samples = [
        BenchmarkSample(0, "000000", "", "4c"),
        BenchmarkSample(1, "000001", "", "4d"),
    ]

    class FakeAdapter:
        raw_format = "kern"

        def prepare_profile(self, *, collect_profile, trace_path=None):
            self._trace_path = trace_path

        def predict_batch(self, images):
            self._last_profile = {
                "preprocess_ms": 5.0,
                "tensor_setup_ms": 5.0,
                "constraint_bundle_ms": 5.0,
                "generate_ms": 180.0,
                "finalize_ms": 5.0,
                "generated_tokens": 40,
                "generated_tokens_per_sample": [20] * len(images),
                "generate_tokens_per_second": 222.0,
                "constraint_stats": {
                    "grammar": {
                        "total_ms": 20.0,
                        "matcher_state_advance_ms": 5.0,
                        "bitmask_fill_ms": 5.0,
                        "bitmask_apply_ms": 10.0,
                        "calls": 10,
                        "rows_processed": len(images),
                        "externally_finished_rows": 0,
                    },
                    "semantic": {
                        "total_ms": 80.0,
                        "advance_row_ms": 20.0,
                        "mask_row_ms": 60.0,
                        "calls": 10,
                        "rows_processed": len(images),
                        "inactive_rows": 0,
                        "terminated_rows": 0,
                    },
                },
                "trace_path": None if self._trace_path is None else str(self._trace_path),
            }
            return ["pred-a", "pred-b"]

        def consume_last_profile(self):
            return self._last_profile

    dataset = [{"image": Image.new("RGB", (4, 4), "white")} for _ in samples]
    runner._run_inference(
        dataset,
        samples,
        FakeAdapter(),
        dataset_name="synth",
        model_name="ours",
    )

    summary_path = runner.run_dir / "synth" / "ours" / "profile" / "profile_summary.json"
    summary = json.loads(summary_path.read_text())

    assert summary["comparison"] is not None
    assert summary["comparison"]["generate_ms_delta"] == 100.0
    assert summary["conclusion"] == "constraint overhead dominated by semantic processor"


def test_run_inference_logs_sample_keys_on_batch_exception(monkeypatch, tmp_path: Path) -> None:
    messages: list[str] = []
    monkeypatch.setattr("src.benchmark.runner.console.log", lambda message: messages.append(str(message)))

    runner = BenchmarkRunner(BenchmarkRunnerConfig(output_root=str(tmp_path), models=("ours",), batch_size=2))
    samples = [
        BenchmarkSample(0, "000000", "", "4c"),
        BenchmarkSample(1, "000001", "", "4d"),
        BenchmarkSample(2, "000002", "", "4e"),
    ]

    class FailingAdapter:
        raw_format = "kern"

        def predict_batch(self, images):
            raise RuntimeError("decode exploded")

    dataset = [{"image": Image.new("RGB", (4, 4), "white")} for _ in samples]

    with pytest.raises(RuntimeError, match="decode exploded"):
        runner._run_inference(
            dataset,
            samples,
            FailingAdapter(),
            dataset_name="synth",
            model_name="ours",
        )

    assert any("inference-crash dataset=synth model=ours" in message for message in messages)
    assert any("sample_keys=['000000', '000001']" in message for message in messages)


def test_our_adapter_uses_grammar_constrained_generate(monkeypatch) -> None:
    from src.benchmark import adapters as adapters_module
    from src.grammar.constraint_factory import ConstraintBundle
    from src.model.generation_policy import enforce_constraint_safe_settings

    sentinel_processor = object()
    captured_kwargs = {}

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(maxlen=32, out_categories=7, bos_token_id=1, eos_token_id=99)

        def generate(self, **kwargs):
            captured_kwargs.update(kwargs)
            return torch.tensor([[1, 2, 0]], dtype=torch.long)

    class FakeGrammarProvider:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def create_logits_processor(self, *, pad_token_id: int | None = None):
            assert pad_token_id == 0
            return sentinel_processor

    fake_loaded = SimpleNamespace(
        model=FakeModel(),
        i2w={1: "<bos>", 2: "4c", 0: "<pad>"},
        pad_token_id=0,
        artifact=SimpleNamespace(
            decoding=DecodingSpec(strategy="beam", num_beams=4, max_len=16),
        ),
        image_width=1050,
        fixed_size=(1485, 1050),
        vocab_dir="./vocab/test",
    )

    monkeypatch.setattr("src.model.checkpoint_loader.load_model_from_checkpoint", lambda *_args: fake_loaded)
    monkeypatch.setattr("src.grammar.provider.GrammarProvider", FakeGrammarProvider)
    monkeypatch.setattr(
        "src.grammar.constraint_factory.ConstrainedDecodingFactory",
        lambda **kwargs: SimpleNamespace(
            build=lambda settings: ConstraintBundle(
                logits_processors=[sentinel_processor],
                stopping_criteria=None,
                generation_settings=enforce_constraint_safe_settings(settings, has_constraints=True),
                semantic_rule_factories=(),
            )
        ),
    )
    preprocess_calls = []

    def fake_preprocess_pil_image(**kwargs):
        preprocess_calls.append(kwargs)
        return torch.zeros((3, 4, 4), dtype=torch.float32), (4, 4)

    monkeypatch.setattr("src.data.preprocessing.preprocess_pil_image", fake_preprocess_pil_image)

    layout_normalization = LayoutNormalizationConfig(enabled=True, top_margin_px=9, side_margin_px=7)
    with pytest.warns(UserWarning, match="not beam-safe"):
        adapter = adapters_module.OurCheckpointAdapter(
            "unused.ckpt",
            torch.device("cpu"),
            layout_normalization=layout_normalization,
        )
    result = adapter.predict_batch([Image.new("RGB", (8, 8), "white")])

    assert result == ["4c"]
    assert captured_kwargs["num_beams"] == 1
    assert captured_kwargs["logits_processor"] == [sentinel_processor]
    assert captured_kwargs["repetition_penalty"] == pytest.approx(1.3)
    assert preprocess_calls[0]["layout_normalization"] == layout_normalization


def test_our_adapter_applies_repetition_penalty_override(monkeypatch) -> None:
    from src.benchmark import adapters as adapters_module
    from src.grammar.constraint_factory import ConstraintBundle

    captured_kwargs = {}

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(maxlen=32, out_categories=7, bos_token_id=1, eos_token_id=99)

        def generate(self, **kwargs):
            captured_kwargs.update(kwargs)
            return torch.tensor([[1, 2, 0]], dtype=torch.long)

    fake_loaded = SimpleNamespace(
        model=FakeModel(),
        i2w={1: "<bos>", 2: "4c", 0: "<pad>"},
        pad_token_id=0,
        artifact=SimpleNamespace(
            decoding=DecodingSpec(strategy="greedy", num_beams=1, max_len=16),
        ),
        image_width=1050,
        fixed_size=(1485, 1050),
        vocab_dir="./vocab/test",
    )

    monkeypatch.setattr("src.model.checkpoint_loader.load_model_from_checkpoint", lambda *_args: fake_loaded)
    monkeypatch.setattr("src.grammar.provider.GrammarProvider", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "src.grammar.constraint_factory.ConstrainedDecodingFactory",
        lambda **kwargs: SimpleNamespace(
            build=lambda settings: ConstraintBundle(
                logits_processors=None,
                stopping_criteria=None,
                generation_settings=settings,
                semantic_rule_factories=(),
            )
        ),
    )
    monkeypatch.setattr(
        "src.data.preprocessing.preprocess_pil_image",
        lambda **kwargs: (torch.zeros((3, 4, 4), dtype=torch.float32), (4, 4)),
    )
    monkeypatch.setattr(
        "src.grammar.semantic_sequence_finalizer.finalize_generated_kern_sequence",
        lambda **_kwargs: SimpleNamespace(text="finalized"),
    )

    adapter = adapters_module.OurCheckpointAdapter(
        "unused.ckpt",
        torch.device("cpu"),
        repetition_penalty=1.35,
    )

    assert adapter.predict_batch([Image.new("RGB", (8, 8), "white")]) == ["finalized"]
    assert captured_kwargs["repetition_penalty"] == pytest.approx(1.35)


def test_our_adapter_disables_rhythm_constraints_by_default(monkeypatch) -> None:
    from src.benchmark import adapters as adapters_module

    captured = {}

    fake_loaded = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(maxlen=32, out_categories=7, bos_token_id=1, eos_token_id=2)),
        i2w={0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "4c"},
        pad_token_id=0,
        artifact=SimpleNamespace(
            decoding=DecodingSpec(strategy="greedy", num_beams=1, max_len=16),
        ),
        image_width=1050,
        fixed_size=(1485, 1050),
        vocab_dir="./vocab/test",
    )

    monkeypatch.setattr("src.model.checkpoint_loader.load_model_from_checkpoint", lambda *_args: fake_loaded)
    monkeypatch.setattr("src.grammar.provider.GrammarProvider", lambda *args, **kwargs: object())

    def fake_factory(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(build=lambda _settings: None)

    monkeypatch.setattr("src.grammar.constraint_factory.ConstrainedDecodingFactory", fake_factory)

    adapters_module.OurCheckpointAdapter("unused.ckpt", torch.device("cpu"))

    assert captured["kwargs"]["use_interpretation_transition_constraints"] is True
    assert captured["kwargs"]["use_rhythm_constraints"] is False
    assert captured["kwargs"]["use_spine_structure_constraints"] is True
    assert captured["kwargs"]["interpretation_transition_config"] is not None
    assert captured["kwargs"]["grammar_provider"] is not None


def test_our_adapter_can_disable_all_constraints(monkeypatch) -> None:
    from src.benchmark import adapters as adapters_module

    captured = {}

    fake_loaded = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(maxlen=32, out_categories=7, bos_token_id=1, eos_token_id=2)),
        i2w={0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "4c"},
        pad_token_id=0,
        artifact=SimpleNamespace(
            decoding=DecodingSpec(strategy="greedy", num_beams=1, max_len=16),
        ),
        image_width=1050,
        fixed_size=(1485, 1050),
        vocab_dir="./vocab/test",
    )

    monkeypatch.setattr("src.model.checkpoint_loader.load_model_from_checkpoint", lambda *_args: fake_loaded)
    monkeypatch.setattr("src.grammar.provider.GrammarProvider", lambda *args, **kwargs: object())

    def fake_factory(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(build=lambda _settings: None)

    monkeypatch.setattr("src.grammar.constraint_factory.ConstrainedDecodingFactory", fake_factory)

    adapters_module.OurCheckpointAdapter("unused.ckpt", torch.device("cpu"), use_constraints=False)

    assert captured["kwargs"]["use_interpretation_transition_constraints"] is False
    assert captured["kwargs"]["use_rhythm_constraints"] is False
    assert captured["kwargs"]["use_spine_structure_constraints"] is False
    assert captured["kwargs"]["grammar_provider"] is None


def test_our_adapter_allows_unconstrained_beam_overrides(monkeypatch) -> None:
    from src.benchmark import adapters as adapters_module
    from src.grammar.constraint_factory import ConstraintBundle

    captured_kwargs = {}

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(maxlen=32, out_categories=7, bos_token_id=1, eos_token_id=2)

        def generate(self, **kwargs):
            captured_kwargs.update(kwargs)
            return torch.tensor([[1, 3, 2, 0]], dtype=torch.long)

    fake_loaded = SimpleNamespace(
        model=FakeModel(),
        i2w={0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "4c"},
        pad_token_id=0,
        artifact=SimpleNamespace(
            decoding=DecodingSpec(strategy="greedy", num_beams=1, max_len=16),
        ),
        image_width=1050,
        fixed_size=(1485, 1050),
        vocab_dir="./vocab/test",
    )

    monkeypatch.setattr("src.model.checkpoint_loader.load_model_from_checkpoint", lambda *_args: fake_loaded)
    monkeypatch.setattr("src.grammar.provider.GrammarProvider", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "src.grammar.constraint_factory.ConstrainedDecodingFactory",
        lambda **kwargs: SimpleNamespace(
            build=lambda settings: ConstraintBundle(
                logits_processors=None,
                stopping_criteria=None,
                generation_settings=settings,
                semantic_rule_factories=(),
            )
        ),
    )
    monkeypatch.setattr(
        "src.data.preprocessing.preprocess_pil_image",
        lambda **kwargs: (torch.zeros((3, 4, 4), dtype=torch.float32), (4, 4)),
    )
    monkeypatch.setattr(
        "src.grammar.semantic_sequence_finalizer.finalize_generated_kern_sequence",
        lambda **_kwargs: SimpleNamespace(text="finalized"),
    )

    adapter = adapters_module.OurCheckpointAdapter(
        "unused.ckpt",
        torch.device("cpu"),
        use_constraints=False,
        strategy="beam",
        num_beams=10,
    )

    assert adapter.predict_batch([Image.new("RGB", (8, 8), "white")]) == ["finalized"]
    assert captured_kwargs["num_beams"] == 10
    assert adapter.decode_policy["requested"]["resolved"]["strategy"] == "beam"
    assert adapter.decode_policy["effective"]["settings"]["strategy"] == "beam"
    assert adapter.decode_policy["effective"]["settings"]["num_beams"] == 10


def test_our_adapter_warns_and_downgrades_constrained_beam_override(monkeypatch) -> None:
    from src.benchmark import adapters as adapters_module
    from src.grammar.constraint_factory import ConstraintBundle
    from src.model.generation_policy import enforce_constraint_safe_settings

    captured_kwargs = {}

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(maxlen=32, out_categories=7, bos_token_id=1, eos_token_id=2)

        def generate(self, **kwargs):
            captured_kwargs.update(kwargs)
            return torch.tensor([[1, 3, 2, 0]], dtype=torch.long)

    fake_loaded = SimpleNamespace(
        model=FakeModel(),
        i2w={0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "4c"},
        pad_token_id=0,
        artifact=SimpleNamespace(
            decoding=DecodingSpec(strategy="greedy", num_beams=1, max_len=16),
        ),
        image_width=1050,
        fixed_size=(1485, 1050),
        vocab_dir="./vocab/test",
    )

    monkeypatch.setattr("src.model.checkpoint_loader.load_model_from_checkpoint", lambda *_args: fake_loaded)
    monkeypatch.setattr("src.grammar.provider.GrammarProvider", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "src.grammar.constraint_factory.ConstrainedDecodingFactory",
        lambda **kwargs: SimpleNamespace(
            build=lambda settings: ConstraintBundle(
                logits_processors=None,
                stopping_criteria=None,
                generation_settings=enforce_constraint_safe_settings(settings, has_constraints=True),
                semantic_rule_factories=(),
            )
        ),
    )
    monkeypatch.setattr(
        "src.data.preprocessing.preprocess_pil_image",
        lambda **kwargs: (torch.zeros((3, 4, 4), dtype=torch.float32), (4, 4)),
    )
    monkeypatch.setattr(
        "src.grammar.semantic_sequence_finalizer.finalize_generated_kern_sequence",
        lambda **_kwargs: SimpleNamespace(text="finalized"),
    )

    with pytest.warns(UserWarning, match="not beam-safe"):
        adapter = adapters_module.OurCheckpointAdapter(
            "unused.ckpt",
            torch.device("cpu"),
            use_constraints=True,
            strategy="beam",
            num_beams=10,
        )

    assert adapter.predict_batch([Image.new("RGB", (8, 8), "white")]) == ["finalized"]
    assert captured_kwargs["num_beams"] == 1
    assert adapter.decode_policy["requested"]["resolved"]["strategy"] == "beam"
    assert adapter.decode_policy["requested"]["resolved"]["num_beams"] == 10
    assert adapter.decode_policy["effective"]["settings"]["strategy"] == "greedy"
    assert adapter.decode_policy["effective"]["settings"]["num_beams"] == 1
    assert adapter.decode_policy["effective"]["downgraded_to_greedy"] is True


def test_our_adapter_finalizes_generated_text(monkeypatch) -> None:
    from src.benchmark import adapters as adapters_module
    from src.grammar.constraint_factory import ConstraintBundle

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(maxlen=32, out_categories=7, bos_token_id=1, eos_token_id=2)

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 0]], dtype=torch.long)

    fake_loaded = SimpleNamespace(
        model=FakeModel(),
        i2w={1: "<bos>", 2: "4c", 0: "<pad>"},
        pad_token_id=0,
        artifact=SimpleNamespace(
            decoding=DecodingSpec(strategy="greedy", num_beams=1, max_len=16),
        ),
        image_width=1050,
        fixed_size=(1485, 1050),
        vocab_dir="./vocab/test",
    )

    monkeypatch.setattr("src.model.checkpoint_loader.load_model_from_checkpoint", lambda *_args: fake_loaded)
    monkeypatch.setattr("src.grammar.provider.GrammarProvider", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "src.grammar.constraint_factory.ConstrainedDecodingFactory",
        lambda **kwargs: SimpleNamespace(
            build=lambda settings: ConstraintBundle(
                logits_processors=None,
                stopping_criteria=None,
                generation_settings=settings,
                semantic_rule_factories=(),
            )
        ),
    )
    monkeypatch.setattr("src.data.preprocessing.preprocess_pil_image", lambda **kwargs: (torch.zeros((3, 4, 4), dtype=torch.float32), (4, 4)))
    monkeypatch.setattr("src.grammar.semantic_sequence_finalizer.finalize_generated_kern_sequence", lambda **_kwargs: SimpleNamespace(text="finalized"))

    adapter = adapters_module.OurCheckpointAdapter("unused.ckpt", torch.device("cpu"))

    assert adapter.predict_batch([Image.new("RGB", (8, 8), "white")]) == ["finalized"]


def test_runner_passes_layout_normalization_to_ours_adapter(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class FakeAdapter:
        raw_format = "kern"

        def predict_batch(self, images):
            return []

    def fake_adapter(
        checkpoint_path,
        device,
        *,
        layout_normalization=None,
        strategy=None,
        num_beams=None,
        repetition_penalty=None,
        use_constraints=True,
        profiling_enabled=False,
        loop_recovery_enabled=False,
        loop_recovery_repetition_penalty=1.35,
    ):
        captured["checkpoint_path"] = checkpoint_path
        captured["device"] = device
        captured["layout_normalization"] = layout_normalization
        captured["strategy"] = strategy
        captured["num_beams"] = num_beams
        captured["repetition_penalty"] = repetition_penalty
        captured["use_constraints"] = use_constraints
        captured["profiling_enabled"] = profiling_enabled
        captured["loop_recovery_enabled"] = loop_recovery_enabled
        captured["loop_recovery_repetition_penalty"] = loop_recovery_repetition_penalty
        return FakeAdapter()

    monkeypatch.setattr("src.benchmark.runner.OurCheckpointAdapter", fake_adapter)

    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            output_root=str(tmp_path),
            models=("ours",),
            ours_normalize_layout=True,
        )
    )

    adapter = runner._get_adapter("ours")

    assert isinstance(adapter, FakeAdapter)
    assert captured["checkpoint_path"] == "weights/GrandStaff/smt-model.ckpt"
    assert isinstance(captured["layout_normalization"], LayoutNormalizationConfig)
    assert captured["repetition_penalty"] is None
    assert captured["use_constraints"] is True
    assert captured["profiling_enabled"] is False
    assert captured["loop_recovery_enabled"] is False


def test_runner_passes_disable_constraints_to_ours_adapter(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class FakeAdapter:
        raw_format = "kern"

        def predict_batch(self, images):
            return []

    def fake_adapter(
        checkpoint_path,
        device,
        *,
        layout_normalization=None,
        strategy=None,
        num_beams=None,
        repetition_penalty=None,
        use_constraints=True,
        profiling_enabled=False,
        loop_recovery_enabled=False,
        loop_recovery_repetition_penalty=1.35,
    ):
        captured["checkpoint_path"] = checkpoint_path
        captured["device"] = device
        captured["layout_normalization"] = layout_normalization
        captured["strategy"] = strategy
        captured["num_beams"] = num_beams
        captured["repetition_penalty"] = repetition_penalty
        captured["use_constraints"] = use_constraints
        captured["profiling_enabled"] = profiling_enabled
        captured["loop_recovery_enabled"] = loop_recovery_enabled
        captured["loop_recovery_repetition_penalty"] = loop_recovery_repetition_penalty
        return FakeAdapter()

    monkeypatch.setattr("src.benchmark.runner.OurCheckpointAdapter", fake_adapter)

    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            output_root=str(tmp_path),
            models=("ours",),
            disable_constraints=True,
        )
    )

    adapter = runner._get_adapter("ours")

    assert isinstance(adapter, FakeAdapter)
    assert captured["checkpoint_path"] == "weights/GrandStaff/smt-model.ckpt"
    assert captured["repetition_penalty"] is None
    assert captured["use_constraints"] is False
    assert captured["profiling_enabled"] is False
    assert captured["loop_recovery_enabled"] is False


def test_runner_passes_profiling_flag_to_ours_adapter(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class FakeAdapter:
        raw_format = "kern"

        def predict_batch(self, images):
            return []

    def fake_adapter(
        checkpoint_path,
        device,
        *,
        layout_normalization=None,
        strategy=None,
        num_beams=None,
        repetition_penalty=None,
        use_constraints=True,
        profiling_enabled=False,
        loop_recovery_enabled=False,
        loop_recovery_repetition_penalty=1.35,
    ):
        captured["checkpoint_path"] = checkpoint_path
        captured["device"] = device
        captured["layout_normalization"] = layout_normalization
        captured["strategy"] = strategy
        captured["num_beams"] = num_beams
        captured["repetition_penalty"] = repetition_penalty
        captured["use_constraints"] = use_constraints
        captured["profiling_enabled"] = profiling_enabled
        captured["loop_recovery_enabled"] = loop_recovery_enabled
        captured["loop_recovery_repetition_penalty"] = loop_recovery_repetition_penalty
        return FakeAdapter()

    monkeypatch.setattr("src.benchmark.runner.OurCheckpointAdapter", fake_adapter)

    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            output_root=str(tmp_path),
            models=("ours",),
            profile=True,
        )
    )

    adapter = runner._get_adapter("ours")

    assert isinstance(adapter, FakeAdapter)
    assert captured["repetition_penalty"] is None
    assert captured["profiling_enabled"] is True
    assert captured["loop_recovery_enabled"] is False


def test_runner_passes_repetition_penalty_to_ours_adapter(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class FakeAdapter:
        raw_format = "kern"

        def predict_batch(self, images):
            return []

    def fake_adapter(
        checkpoint_path,
        device,
        *,
        layout_normalization=None,
        strategy=None,
        num_beams=None,
        repetition_penalty=None,
        use_constraints=True,
        profiling_enabled=False,
        loop_recovery_enabled=False,
        loop_recovery_repetition_penalty=1.35,
    ):
        captured["checkpoint_path"] = checkpoint_path
        captured["device"] = device
        captured["layout_normalization"] = layout_normalization
        captured["strategy"] = strategy
        captured["num_beams"] = num_beams
        captured["repetition_penalty"] = repetition_penalty
        captured["use_constraints"] = use_constraints
        captured["profiling_enabled"] = profiling_enabled
        captured["loop_recovery_enabled"] = loop_recovery_enabled
        captured["loop_recovery_repetition_penalty"] = loop_recovery_repetition_penalty
        return FakeAdapter()

    monkeypatch.setattr("src.benchmark.runner.OurCheckpointAdapter", fake_adapter)

    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            output_root=str(tmp_path),
            models=("ours",),
            ours_repetition_penalty=1.25,
        )
    )

    adapter = runner._get_adapter("ours")

    assert isinstance(adapter, FakeAdapter)
    assert captured["repetition_penalty"] == pytest.approx(1.25)


def test_runner_passes_loop_recovery_settings_to_ours_adapter(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class FakeAdapter:
        raw_format = "kern"

        def predict_batch(self, images):
            return []

    def fake_adapter(
        checkpoint_path,
        device,
        *,
        layout_normalization=None,
        strategy=None,
        num_beams=None,
        repetition_penalty=None,
        use_constraints=True,
        profiling_enabled=False,
        loop_recovery_enabled=False,
        loop_recovery_repetition_penalty=1.35,
    ):
        captured["strategy"] = strategy
        captured["num_beams"] = num_beams
        captured["loop_recovery_enabled"] = loop_recovery_enabled
        captured["loop_recovery_repetition_penalty"] = loop_recovery_repetition_penalty
        return FakeAdapter()

    monkeypatch.setattr("src.benchmark.runner.OurCheckpointAdapter", fake_adapter)

    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            output_root=str(tmp_path),
            models=("ours",),
            ours_loop_recovery=True,
            ours_loop_recovery_repetition_penalty=1.5,
        )
    )

    runner._get_adapter("ours")

    assert captured["loop_recovery_enabled"] is True
    assert captured["loop_recovery_repetition_penalty"] == pytest.approx(1.5)


def test_runner_passes_decode_overrides_to_ours_adapter(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class FakeAdapter:
        raw_format = "kern"

        def predict_batch(self, images):
            return []

    def fake_adapter(
        checkpoint_path,
        device,
        *,
        layout_normalization=None,
        strategy=None,
        num_beams=None,
        repetition_penalty=None,
        use_constraints=True,
        profiling_enabled=False,
        loop_recovery_enabled=False,
        loop_recovery_repetition_penalty=1.35,
    ):
        captured["checkpoint_path"] = checkpoint_path
        captured["device"] = device
        captured["strategy"] = strategy
        captured["num_beams"] = num_beams
        captured["repetition_penalty"] = repetition_penalty
        captured["use_constraints"] = use_constraints
        return FakeAdapter()

    monkeypatch.setattr("src.benchmark.runner.OurCheckpointAdapter", fake_adapter)

    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            output_root=str(tmp_path),
            models=("ours",),
            ours_strategy="beam",
            ours_num_beams=10,
        )
    )

    adapter = runner._get_adapter("ours")

    assert isinstance(adapter, FakeAdapter)
    assert captured["strategy"] == "beam"
    assert captured["num_beams"] == 10
    assert captured["repetition_penalty"] is None
    assert captured["use_constraints"] is True


def test_run_inference_preserves_prediction_row_metadata(tmp_path: Path) -> None:
    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            output_root=str(tmp_path),
            models=("ours",),
            batch_size=2,
        )
    )
    samples = [
        BenchmarkSample(0, "000000", "", "4c"),
        BenchmarkSample(1, "000001", "", "4d"),
    ]

    class FakeAdapter:
        raw_format = "kern"

        def predict_batch_rows(self, images):
            assert len(images) == 2
            return [
                {
                    "prediction": "pred-0",
                    "loop_recovery": {
                        "primary_detected": True,
                        "rerun_attempted": True,
                        "replaced_prediction": True,
                    },
                },
                {"prediction": "pred-1"},
            ]

    dataset = [{"image": Image.new("RGB", (4, 4), "white")} for _ in samples]
    predictions = runner._run_inference(
        dataset,
        samples,
        FakeAdapter(),
        dataset_name="synth",
        model_name="ours",
    )

    assert predictions[0]["prediction"] == "pred-0"
    assert predictions[0]["loop_recovery"]["primary_detected"] is True
    assert "loop_recovery" not in predictions[1]


def test_our_adapter_loop_recovery_reruns_only_flagged_samples(monkeypatch) -> None:
    from src.benchmark import adapters as adapters_module
    from src.grammar.constraint_factory import ConstraintBundle

    captured_generate_kwargs: list[dict[str, object]] = []

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(maxlen=32, out_categories=7, bos_token_id=1, eos_token_id=2)
            self.calls = 0

        def generate(self, **kwargs):
            captured_generate_kwargs.append(dict(kwargs))
            self.calls += 1
            token_id = 3 if self.calls == 1 else 4
            batch_size = kwargs["pixel_values"].shape[0]
            return torch.tensor([[1, token_id, 2, 0]] * batch_size, dtype=torch.long)

    fake_loaded = SimpleNamespace(
        model=FakeModel(),
        i2w={0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "4c", 4: "4d"},
        pad_token_id=0,
        artifact=SimpleNamespace(decoding=DecodingSpec(strategy="greedy", num_beams=1, max_len=16)),
        image_width=1050,
        fixed_size=(1485, 1050),
        vocab_dir="./vocab/test",
    )

    monkeypatch.setattr("src.model.checkpoint_loader.load_model_from_checkpoint", lambda *_args: fake_loaded)
    monkeypatch.setattr("src.grammar.provider.GrammarProvider", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "src.grammar.constraint_factory.ConstrainedDecodingFactory",
        lambda **kwargs: SimpleNamespace(
            build=lambda settings: ConstraintBundle(
                logits_processors=None,
                stopping_criteria=None,
                generation_settings=settings,
                semantic_rule_factories=(),
            )
        ),
    )
    monkeypatch.setattr(
        "src.data.preprocessing.preprocess_pil_image",
        lambda **kwargs: (torch.zeros((3, 4, 4), dtype=torch.float32), (4, 4)),
    )
    monkeypatch.setattr(
        "src.grammar.semantic_sequence_finalizer.finalize_generated_kern_sequence",
        lambda **kwargs: SimpleNamespace(text="looped" if kwargs["token_ids"][1] == 3 else "fixed"),
    )

    def fake_catastrophic(text, config=None):
        if text == "looped":
            return SimpleNamespace(
                repeat_loop=True,
                repeat_loop_reason="identical_line_run",
                max_identical_line_run=20,
                max_repeated_ngram_occurrences=0,
                repeated_ngram_size=4,
                repeated_ngram_line_coverage=0.0,
            )
        return SimpleNamespace(
            repeat_loop=False,
            repeat_loop_reason=None,
            max_identical_line_run=1,
            max_repeated_ngram_occurrences=0,
            repeated_ngram_size=4,
            repeated_ngram_line_coverage=0.0,
        )

    monkeypatch.setattr("src.benchmark.adapters.analyze_catastrophic_repetition", fake_catastrophic)

    adapter = adapters_module.OurCheckpointAdapter(
        "unused.ckpt",
        torch.device("cpu"),
        repetition_penalty=1.1,
        loop_recovery_enabled=True,
        loop_recovery_repetition_penalty=1.35,
    )

    rows = adapter.predict_batch_rows([Image.new("RGB", (8, 8), "white")])

    assert rows[0]["prediction"] == "fixed"
    assert rows[0]["loop_recovery"]["primary_detected"] is True
    assert rows[0]["loop_recovery"]["rerun_attempted"] is True
    assert rows[0]["loop_recovery"]["replaced_prediction"] is True
    assert len(captured_generate_kwargs) == 2
    assert captured_generate_kwargs[0]["repetition_penalty"] == pytest.approx(1.1)
    assert captured_generate_kwargs[1]["repetition_penalty"] == pytest.approx(1.35)


def test_our_adapter_loop_recovery_skips_rerun_for_unflagged_output(monkeypatch) -> None:
    from src.benchmark import adapters as adapters_module
    from src.grammar.constraint_factory import ConstraintBundle

    captured_generate_kwargs: list[dict[str, object]] = []

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(maxlen=32, out_categories=7, bos_token_id=1, eos_token_id=2)

        def generate(self, **kwargs):
            captured_generate_kwargs.append(dict(kwargs))
            return torch.tensor([[1, 3, 2, 0]], dtype=torch.long)

    fake_loaded = SimpleNamespace(
        model=FakeModel(),
        i2w={0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "4c"},
        pad_token_id=0,
        artifact=SimpleNamespace(decoding=DecodingSpec(strategy="greedy", num_beams=1, max_len=16)),
        image_width=1050,
        fixed_size=(1485, 1050),
        vocab_dir="./vocab/test",
    )

    monkeypatch.setattr("src.model.checkpoint_loader.load_model_from_checkpoint", lambda *_args: fake_loaded)
    monkeypatch.setattr("src.grammar.provider.GrammarProvider", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "src.grammar.constraint_factory.ConstrainedDecodingFactory",
        lambda **kwargs: SimpleNamespace(
            build=lambda settings: ConstraintBundle(
                logits_processors=None,
                stopping_criteria=None,
                generation_settings=settings,
                semantic_rule_factories=(),
            )
        ),
    )
    monkeypatch.setattr(
        "src.data.preprocessing.preprocess_pil_image",
        lambda **kwargs: (torch.zeros((3, 4, 4), dtype=torch.float32), (4, 4)),
    )
    monkeypatch.setattr(
        "src.grammar.semantic_sequence_finalizer.finalize_generated_kern_sequence",
        lambda **_kwargs: SimpleNamespace(text="clean"),
    )
    monkeypatch.setattr(
        "src.benchmark.adapters.analyze_catastrophic_repetition",
        lambda *_args, **_kwargs: SimpleNamespace(
            repeat_loop=False,
            repeat_loop_reason=None,
            max_identical_line_run=1,
            max_repeated_ngram_occurrences=0,
            repeated_ngram_size=4,
            repeated_ngram_line_coverage=0.0,
        ),
    )

    adapter = adapters_module.OurCheckpointAdapter(
        "unused.ckpt",
        torch.device("cpu"),
        loop_recovery_enabled=True,
    )

    rows = adapter.predict_batch_rows([Image.new("RGB", (8, 8), "white")])

    assert rows[0]["prediction"] == "clean"
    assert rows[0]["loop_recovery"]["primary_detected"] is False
    assert rows[0]["loop_recovery"]["rerun_attempted"] is False
    assert len(captured_generate_kwargs) == 1


def test_our_adapter_keeps_primary_when_rerun_is_still_flagged(monkeypatch) -> None:
    from src.benchmark import adapters as adapters_module
    from src.grammar.constraint_factory import ConstraintBundle

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(maxlen=32, out_categories=7, bos_token_id=1, eos_token_id=2)

        def generate(self, **kwargs):
            return torch.tensor([[1, 3, 2, 0]], dtype=torch.long)

    fake_loaded = SimpleNamespace(
        model=FakeModel(),
        i2w={0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "4c"},
        pad_token_id=0,
        artifact=SimpleNamespace(decoding=DecodingSpec(strategy="greedy", num_beams=1, max_len=16)),
        image_width=1050,
        fixed_size=(1485, 1050),
        vocab_dir="./vocab/test",
    )

    monkeypatch.setattr("src.model.checkpoint_loader.load_model_from_checkpoint", lambda *_args: fake_loaded)
    monkeypatch.setattr("src.grammar.provider.GrammarProvider", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "src.grammar.constraint_factory.ConstrainedDecodingFactory",
        lambda **kwargs: SimpleNamespace(
            build=lambda settings: ConstraintBundle(
                logits_processors=None,
                stopping_criteria=None,
                generation_settings=settings,
                semantic_rule_factories=(),
            )
        ),
    )
    monkeypatch.setattr(
        "src.data.preprocessing.preprocess_pil_image",
        lambda **kwargs: (torch.zeros((3, 4, 4), dtype=torch.float32), (4, 4)),
    )
    monkeypatch.setattr(
        "src.grammar.semantic_sequence_finalizer.finalize_generated_kern_sequence",
        lambda **_kwargs: SimpleNamespace(text="looped"),
    )
    monkeypatch.setattr(
        "src.benchmark.adapters.analyze_catastrophic_repetition",
        lambda *_args, **_kwargs: SimpleNamespace(
            repeat_loop=True,
            repeat_loop_reason="identical_line_run",
            max_identical_line_run=20,
            max_repeated_ngram_occurrences=0,
            repeated_ngram_size=4,
            repeated_ngram_line_coverage=0.0,
        ),
    )

    adapter = adapters_module.OurCheckpointAdapter(
        "unused.ckpt",
        torch.device("cpu"),
        loop_recovery_enabled=True,
    )

    rows = adapter.predict_batch_rows([Image.new("RGB", (8, 8), "white")])

    assert rows[0]["prediction"] == "looped"
    assert rows[0]["loop_recovery"]["primary_detected"] is True
    assert rows[0]["loop_recovery"]["rerun_detected"] is True
    assert rows[0]["loop_recovery"]["replaced_prediction"] is False


def test_loop_recovery_counts_are_persisted_in_summary(tmp_path: Path) -> None:
    summary = DatasetModelSummary(
        "synth",
        "ours",
        "kern",
        2,
        0,
        0,
        100.0,
        10.0,
        20.0,
        3,
        3,
        2,
        1,
    )

    runner = BenchmarkRunner(BenchmarkRunnerConfig(output_root=str(tmp_path)))
    runner._write_summary_files([summary], summarize_overall([summary]))

    payload = json.loads((runner.run_dir / "summary.json").read_text())

    assert payload["rows"][0]["num_loop_recovery_primary_flagged"] == 3
    assert payload["rows"][0]["num_loop_recovery_recovered"] == 2


def test_runner_passes_smtpp_max_length_to_adapter(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class FakeAdapter:
        raw_format = "kern"

        def predict_batch(self, images):
            return []

    def fake_adapter(model_id, device, *, max_length=None):
        captured["model_id"] = model_id
        captured["device"] = device
        captured["max_length"] = max_length
        return FakeAdapter()

    monkeypatch.setattr("src.benchmark.runner.SMTPlusPlusAdapter", fake_adapter)

    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            output_root=str(tmp_path),
            models=("smtpp",),
            smtpp_max_length=256,
        )
    )

    adapter = runner._get_adapter("smtpp")

    assert isinstance(adapter, FakeAdapter)
    assert captured["model_id"] == "PRAIG/smt-fp-grandstaff"
    assert captured["max_length"] == 256


def test_smtpp_adapter_applies_decode_max_length(monkeypatch) -> None:
    from src.benchmark import adapters as adapters_module

    class FakePRAIGConfig:
        pass

    class FakePRAIGModel:
        config_class = FakePRAIGConfig

        def __init__(self) -> None:
            self.maxlen = 1024

        @classmethod
        def from_pretrained(cls, _model_id):
            assert cls.config_class.is_encoder_decoder is False
            assert cls.config_class.tie_word_embeddings is False
            assert cls.config_class._attn_implementation_internal is None
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

    fake_module = ModuleType("smt_model")
    fake_module.SMTModelForCausalLM = FakePRAIGModel
    monkeypatch.setitem(sys.modules, "smt_model", fake_module)

    adapter = adapters_module.SMTPlusPlusAdapter(
        "unused-model",
        torch.device("cpu"),
        max_length=256,
    )

    assert adapter.model.maxlen == 256


def test_smtpp_adapter_batches_images_for_single_predict_call(monkeypatch) -> None:
    from src.benchmark import adapters as adapters_module

    captured: dict[str, object] = {"predict_calls": 0}

    class FakePRAIGConfig:
        pass

    class FakePRAIGModel:
        config_class = FakePRAIGConfig

        @classmethod
        def from_pretrained(cls, _model_id):
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

        def predict(self, batch, image_sizes=None, convert_to_str=False):
            captured["predict_calls"] = int(captured["predict_calls"]) + 1
            captured["batch_shape"] = tuple(batch.shape)
            captured["image_sizes"] = image_sizes.detach().cpu()
            captured["convert_to_str"] = convert_to_str
            return [["A", "<b>", "B"], ["C", "<t>", "D"]], None

    fake_module = ModuleType("smt_model")
    fake_module.SMTModelForCausalLM = FakePRAIGModel
    monkeypatch.setitem(sys.modules, "smt_model", fake_module)

    adapter = adapters_module.SMTPlusPlusAdapter(
        "unused-model",
        torch.device("cpu"),
    )

    outputs = adapter.predict_batch(
        [
            Image.new("RGB", (8, 5), "white"),
            Image.new("RGB", (3, 7), "white"),
        ]
    )

    assert outputs == ["A\nB", "C\tD"]
    assert captured["predict_calls"] == 1
    assert captured["batch_shape"] == (2, 1, 7, 8)
    assert torch.equal(
        captured["image_sizes"],
        torch.tensor([[5, 8], [7, 3]], dtype=torch.long),
    )
    assert captured["convert_to_str"] is True


def test_legato_model_init_loads_nested_vision_model(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path("models/external/legato").resolve()))
    import legato.models.modeling_legato as modeling_legato

    fake_vision_model = SimpleNamespace(parameters=lambda: [])

    def fake_base_init(self, config):
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "model", SimpleNamespace(vision_model=None))

    monkeypatch.setattr(modeling_legato.MllamaForConditionalGeneration, "__init__", fake_base_init)
    monkeypatch.setattr(
        modeling_legato.MllamaVisionModel,
        "from_pretrained",
        staticmethod(lambda _ref: fake_vision_model),
    )

    config = SimpleNamespace(
        encoder_pretrained_model_name_or_path="encoder-ref",
        vision_config=SimpleNamespace(),
    )
    model = modeling_legato.LegatoModel(config, load_pretrained_encoder=True)

    assert model.model.vision_model is fake_vision_model


def test_legato_adapter_overrides_encoder_path(monkeypatch, tmp_path: Path) -> None:
    from src.benchmark import adapters as adapters_module

    captured: dict[str, object] = {}

    class FakeConfig:
        encoder_pretrained_model_name_or_path = "meta-llama/Llama-3.2-11B-Vision"

    class FakeLegatoModel:
        @classmethod
        def from_pretrained(cls, model_id, *args, **kwargs):
            captured["model_id"] = model_id
            captured["config"] = kwargs.get("config")
            return cls()

        def to(self, device=None):
            captured["device"] = device
            return self

        def eval(self):
            captured["eval_called"] = True
            return self

    fake_legato_package = ModuleType("legato")
    fake_legato_models = ModuleType("legato.models")
    fake_legato_models.LegatoModel = FakeLegatoModel
    fake_legato_package.models = fake_legato_models

    monkeypatch.setitem(sys.modules, "legato", fake_legato_package)
    monkeypatch.setitem(sys.modules, "legato.models", fake_legato_models)
    monkeypatch.setattr(adapters_module.AutoConfig, "from_pretrained", staticmethod(lambda _model_id: FakeConfig()))
    monkeypatch.setattr(
        adapters_module.AutoProcessor,
        "from_pretrained",
        staticmethod(lambda model_id: SimpleNamespace(model_id=model_id)),
    )

    encoder_dir = tmp_path / "llama32-11b-vision"
    encoder_dir.mkdir()

    adapter = adapters_module.LegatoAdapter(
        "legato-snapshot",
        torch.device("cpu"),
        encoder_path=str(encoder_dir),
    )

    assert adapter.processor.model_id == "legato-snapshot"
    assert captured["model_id"] == "legato-snapshot"
    assert captured["device"] == torch.device("cpu")
    assert captured["eval_called"] is True
    assert captured["config"].encoder_pretrained_model_name_or_path == str(encoder_dir)


def test_legato_adapter_rejects_missing_encoder_override(monkeypatch, tmp_path: Path) -> None:
    from src.benchmark import adapters as adapters_module

    class FakeLegatoModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise AssertionError("from_pretrained should not be called for a missing encoder path")

    fake_legato_package = ModuleType("legato")
    fake_legato_models = ModuleType("legato.models")
    fake_legato_models.LegatoModel = FakeLegatoModel
    fake_legato_package.models = fake_legato_models

    monkeypatch.setitem(sys.modules, "legato", fake_legato_package)
    monkeypatch.setitem(sys.modules, "legato.models", fake_legato_models)

    missing_path = tmp_path / "missing-encoder"
    with pytest.raises(FileNotFoundError, match="LEGATO encoder override path does not exist"):
        adapters_module.LegatoAdapter(
            "legato-snapshot",
            torch.device("cpu"),
            encoder_path=str(missing_path),
        )


def test_runner_rejects_unknown_metrics(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported benchmark metrics"):
        BenchmarkRunner(
            BenchmarkRunnerConfig(
                output_root=str(tmp_path),
                metrics=("omr_ned", "bogus"),
            )
        )


def test_runner_accepts_cer_metric(tmp_path: Path) -> None:
    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            output_root=str(tmp_path),
            metrics=("cer",),
        )
    )

    assert runner.selected_metrics == ("cer",)


def test_runner_passes_legato_generation_settings(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class FakeAdapter:
        raw_format = "abc"

        def predict_batch(self, images):
            return []

    def fake_adapter(model_id, device, *, encoder_path=None, max_length=2048, num_beams=10):
        captured["model_id"] = model_id
        captured["device"] = device
        captured["encoder_path"] = encoder_path
        captured["max_length"] = max_length
        captured["num_beams"] = num_beams
        return FakeAdapter()

    monkeypatch.setattr("src.benchmark.runner.LegatoAdapter", fake_adapter)

    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            output_root=str(tmp_path),
            models=("legato",),
            legato_max_length=256,
            legato_num_beams=1,
        )
    )

    adapter = runner._get_adapter("legato")

    assert isinstance(adapter, FakeAdapter)
    assert captured["model_id"] == "guangyangmusic/legato"
    assert captured["encoder_path"] is None
    assert captured["max_length"] == 256
    assert captured["num_beams"] == 1


def test_runner_passes_legato_encoder_path(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class FakeAdapter:
        raw_format = "abc"

        def predict_batch(self, images):
            return []

    def fake_adapter(model_id, device, *, encoder_path=None, max_length=2048, num_beams=10):
        captured["model_id"] = model_id
        captured["device"] = device
        captured["encoder_path"] = encoder_path
        captured["max_length"] = max_length
        captured["num_beams"] = num_beams
        return FakeAdapter()

    monkeypatch.setattr("src.benchmark.runner.LegatoAdapter", fake_adapter)

    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            output_root=str(tmp_path),
            models=("legato",),
            legato_model_id="/workspace/hf/legato",
            legato_encoder_path="/workspace/hf/llama32-11b-vision",
        )
    )

    adapter = runner._get_adapter("legato")

    assert isinstance(adapter, FakeAdapter)
    assert captured["model_id"] == "/workspace/hf/legato"
    assert captured["encoder_path"] == "/workspace/hf/llama32-11b-vision"


def test_build_profile_recorder_records_legato_encoder_path(tmp_path: Path) -> None:
    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            output_root=str(tmp_path),
            models=("legato",),
            legato_encoder_path="/workspace/hf/llama32-11b-vision",
            profile=True,
        )
    )

    recorder = runner._build_profile_recorder(dataset_name="synth", model_name="legato")

    assert recorder.metadata["checkpoint_path"] == "guangyangmusic/legato"
    assert recorder.metadata["legato_encoder_path"] == "/workspace/hf/llama32-11b-vision"


def test_compute_metric_job_maps_tedn_exception_to_100(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.benchmark.runner.convert_kern_to_musicxml",
        lambda *_args, **_kwargs: ConversionResult(musicxml="<score-partwise><part id='P1'/></score-partwise>"),
    )
    monkeypatch.setattr("src.benchmark.runner.compute_tedn_from_musicxml", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("tedn broke")))
    monkeypatch.setattr(
        "src.benchmark.runner.compute_omr_ned",
        lambda *_args, **_kwargs: OMRNEDResult(omr_ned=12.5, edit_distance=1, pred_notation_size=1, gt_notation_size=1, parse_error=None, syntax_errors_fixed=0),
    )

    result = compute_metric_job(
        MetricJob(
            sample_index=0,
            sample_key="000000",
            source="",
            prediction="4c",
            ground_truth="4c",
            raw_format="kern",
            gold_xml_text="<score-partwise><part id='P1'/></score-partwise>",
            hum2xml_path="hum2xml",
            abc2xml_command=None,
            metrics=("omr_ned", "tedn"),
        )
    )

    assert result.sample_metric.tedn == 100.0
    assert result.sample_metric.tedn_error == "tedn broke"
    assert result.sample_metric.omr_ned == 12.5


def test_compute_metric_job_computes_cer_for_kern_when_selected() -> None:
    result = compute_metric_job(
        MetricJob(
            sample_index=0,
            sample_key="000000",
            source="",
            prediction="**kern\n4c\n*-",
            ground_truth="**kern\n4d\n*-",
            raw_format="kern",
            gold_xml_text="",
            hum2xml_path="hum2xml",
            abc2xml_command=None,
            metrics=("cer",),
            compute_cer=True,
        )
    )

    assert result.sample_metric.cer == pytest.approx(100.0 / len("**kern\n4d\n*-"))


def test_compute_metric_job_leaves_cer_empty_for_abc_when_selected() -> None:
    result = compute_metric_job(
        MetricJob(
            sample_index=0,
            sample_key="000000",
            source="",
            prediction="X:1\nC",
            ground_truth="**kern\n4d\n*-",
            raw_format="abc",
            gold_xml_text="",
            hum2xml_path="hum2xml",
            abc2xml_command=None,
            metrics=("cer",),
            compute_cer=True,
        )
    )

    assert result.sample_metric.cer is None
    assert result.sample_metric.conversion_failed is False


def test_compute_metric_job_maps_omr_exception_to_100(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.benchmark.runner.convert_kern_to_musicxml",
        lambda *_args, **_kwargs: ConversionResult(musicxml="<score-partwise><part id='P1'/></score-partwise>"),
    )
    monkeypatch.setattr("src.benchmark.runner.compute_tedn_from_musicxml", lambda *_args, **_kwargs: 7.0)
    monkeypatch.setattr("src.benchmark.runner.compute_omr_ned", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("omr broke")))

    result = compute_metric_job(
        MetricJob(
            sample_index=0,
            sample_key="000000",
            source="",
            prediction="4c",
            ground_truth="4c",
            raw_format="kern",
            gold_xml_text="<score-partwise><part id='P1'/></score-partwise>",
            hum2xml_path="hum2xml",
            abc2xml_command=None,
            metrics=("omr_ned", "tedn"),
        )
    )

    assert result.sample_metric.tedn == 7.0
    assert result.sample_metric.omr_ned == 100.0
    assert result.sample_metric.omr_ned_error == "omr broke"


def test_compute_metric_job_maps_omr_parse_failure_to_100(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.benchmark.runner.convert_kern_to_musicxml",
        lambda *_args, **_kwargs: ConversionResult(musicxml="<score-partwise><part id='P1'/></score-partwise>"),
    )
    monkeypatch.setattr("src.benchmark.runner.compute_tedn_from_musicxml", lambda *_args, **_kwargs: 7.0)
    monkeypatch.setattr(
        "src.benchmark.runner.compute_omr_ned",
        lambda *_args, **_kwargs: OMRNEDResult(
            omr_ned=None,
            edit_distance=None,
            pred_notation_size=None,
            gt_notation_size=None,
            parse_error="parse failed",
            syntax_errors_fixed=0,
        ),
    )

    result = compute_metric_job(
        MetricJob(
            sample_index=0,
            sample_key="000000",
            source="",
            prediction="4c",
            ground_truth="4c",
            raw_format="kern",
            gold_xml_text="<score-partwise><part id='P1'/></score-partwise>",
            hum2xml_path="hum2xml",
            abc2xml_command=None,
            metrics=("omr_ned", "tedn"),
        )
    )

    assert result.sample_metric.tedn == 7.0
    assert result.sample_metric.omr_ned == 100.0
    assert result.sample_metric.omr_ned_error == "parse failed"


def test_compute_metric_job_maps_worker_exception_to_failed_sample(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.benchmark.runner.convert_kern_to_musicxml",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = compute_metric_job(
        MetricJob(
            sample_index=0,
            sample_key="000000",
            source="",
            prediction="4c",
            ground_truth="4c",
            raw_format="kern",
            gold_xml_text="<score-partwise><part id='P1'/></score-partwise>",
            hum2xml_path="hum2xml",
            abc2xml_command=None,
            metrics=("omr_ned", "tedn"),
        )
    )

    assert result.sample_metric.conversion_failed is True
    assert result.sample_metric.metric_worker_failed is True
    assert result.sample_metric.tedn == 100.0
    assert "Metric worker crashed: boom" in (result.sample_metric.conversion_error or "")


def test_compute_metric_job_omr_only_worker_exception_is_not_conversion_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.benchmark.runner.compute_omr_ned",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = compute_metric_job(
        MetricJob(
            sample_index=0,
            sample_key="000000",
            source="",
            prediction="4c",
            ground_truth="4c",
            raw_format="kern",
            gold_xml_text="<score-partwise><part id='P1'/></score-partwise>",
            hum2xml_path="hum2xml",
            abc2xml_command=None,
            metrics=("omr_ned",),
        )
    )

    assert result.sample_metric.metric_worker_failed is False
    assert result.sample_metric.conversion_failed is False
    assert result.sample_metric.conversion_error is None
    assert result.sample_metric.omr_ned == 100.0
    assert result.sample_metric.omr_ned_error == "boom"


def test_compute_metric_job_outer_worker_crash_does_not_fake_conversion_failure_for_kern_omr_only(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.benchmark.runner.compute_omr_ned",
        lambda *_args, **_kwargs: OMRNEDResult(
            omr_ned=12.5,
            edit_distance=1,
            pred_notation_size=2,
            gt_notation_size=2,
            parse_error=None,
            syntax_errors_fixed=0,
        ),
    )
    monkeypatch.setattr(
        "src.benchmark.runner.resolve_omr_ned_score",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = compute_metric_job(
        MetricJob(
            sample_index=0,
            sample_key="000000",
            source="",
            prediction="4c",
            ground_truth="4c",
            raw_format="kern",
            gold_xml_text="<score-partwise><part id='P1'/></score-partwise>",
            hum2xml_path="hum2xml",
            abc2xml_command=None,
            metrics=("omr_ned",),
        )
    )

    assert result.sample_metric.metric_worker_failed is True
    assert result.sample_metric.conversion_failed is False
    assert result.sample_metric.conversion_error is None
    assert result.sample_metric.omr_ned == 100.0
    assert "Metric worker crashed: boom" == result.sample_metric.omr_ned_error


def test_compute_metric_job_keeps_kern_omr_when_xml_conversion_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.benchmark.runner.convert_kern_to_musicxml",
        lambda *_args, **_kwargs: ConversionResult(musicxml=None, error="hum2xml failed"),
    )
    monkeypatch.setattr(
        "src.benchmark.runner.compute_omr_ned",
        lambda *_args, **_kwargs: OMRNEDResult(
            omr_ned=17.5,
            edit_distance=2,
            pred_notation_size=4,
            gt_notation_size=4,
            parse_error=None,
            syntax_errors_fixed=0,
        ),
    )

    result = compute_metric_job(
        MetricJob(
            sample_index=0,
            sample_key="000000",
            source="",
            prediction="4c",
            ground_truth="4d",
            raw_format="kern",
            gold_xml_text="<score-partwise><part id='P1'/></score-partwise>",
            hum2xml_path="hum2xml",
            abc2xml_command=None,
            metrics=("cer", "omr_ned", "tedn"),
            compute_cer=True,
        )
    )

    assert result.sample_metric.conversion_failed is True
    assert result.sample_metric.conversion_error == "hum2xml failed"
    assert result.sample_metric.tedn == 100.0
    assert result.sample_metric.tedn_error == "hum2xml failed"
    assert result.sample_metric.omr_ned == 17.5
    assert result.sample_metric.omr_ned_error is None
    assert result.sample_metric.cer == pytest.approx(50.0)


def test_compute_metrics_enables_cer_for_smtpp_kern_predictions(monkeypatch, tmp_path: Path) -> None:
    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            output_root=str(tmp_path),
            models=("smtpp",),
            metrics=("cer",),
            metric_workers=1,
        )
    )
    captured_jobs: list[MetricJob] = []

    def fake_compute_metric_job(job: MetricJob):
        captured_jobs.append(job)
        return SimpleNamespace(
            sample_metric=SampleMetric(
                sample_index=job.sample_index,
                sample_key=job.sample_key,
                source=job.source,
                cer=12.5,
            ),
            pred_xml_text=None,
        )

    monkeypatch.setattr("src.benchmark.runner.compute_metric_job", fake_compute_metric_job)

    rows = runner._compute_metrics(
        dataset_name="synth",
        model_name="smtpp",
        samples=[BenchmarkSample(0, "000000", "", "**kern\n4d\n*-")],
        gold_xml={},
        prediction_by_key={"000000": {"prediction": "**kern\n4c\n*-", "raw_format": "kern"}},
        raw_format="kern",
        pred_xml_dir=tmp_path / "pred_xml",
        metrics_start=0.0,
    )

    assert len(captured_jobs) == 1
    assert captured_jobs[0].compute_cer is True
    assert rows[0].cer == 12.5


def test_compute_metric_job_omr_only_skips_kern_xml_conversion(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.benchmark.runner.convert_kern_to_musicxml",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("kern XML conversion should be skipped")),
    )
    monkeypatch.setattr(
        "src.benchmark.runner.compute_omr_ned",
        lambda *_args, **_kwargs: OMRNEDResult(
            omr_ned=22.5,
            edit_distance=1,
            pred_notation_size=2,
            gt_notation_size=2,
            parse_error=None,
            syntax_errors_fixed=0,
        ),
    )

    result = compute_metric_job(
        MetricJob(
            sample_index=0,
            sample_key="000000",
            source="",
            prediction="4c",
            ground_truth="4d",
            raw_format="kern",
            gold_xml_text="<score-partwise><part id='P1'/></score-partwise>",
            hum2xml_path="hum2xml",
            abc2xml_command=None,
            metrics=("omr_ned",),
        )
    )

    assert result.sample_metric.conversion_failed is False
    assert result.sample_metric.tedn is None
    assert result.sample_metric.omr_ned == 22.5
    assert result.pred_xml_text is None


def test_compute_metric_job_tedn_only_skips_kern_omr(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.benchmark.runner.compute_omr_ned",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("OMR-NED should be skipped")),
    )
    monkeypatch.setattr(
        "src.benchmark.runner.convert_kern_to_musicxml",
        lambda *_args, **_kwargs: ConversionResult(musicxml="<score-partwise><part id='P1'/></score-partwise>"),
    )
    monkeypatch.setattr("src.benchmark.runner.compute_tedn_from_musicxml", lambda *_args, **_kwargs: 7.0)

    result = compute_metric_job(
        MetricJob(
            sample_index=0,
            sample_key="000000",
            source="",
            prediction="4c",
            ground_truth="4d",
            raw_format="kern",
            gold_xml_text="<score-partwise><part id='P1'/></score-partwise>",
            hum2xml_path="hum2xml",
            abc2xml_command=None,
            metrics=("tedn",),
        )
    )

    assert result.sample_metric.tedn == 7.0
    assert result.sample_metric.omr_ned is None


def test_compute_metric_job_abc_omr_only_uses_musicxml(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.benchmark.runner.convert_abc_to_musicxml",
        lambda *_args, **_kwargs: ConversionResult(musicxml="<score-partwise><part id='P1'/></score-partwise>"),
    )
    monkeypatch.setattr(
        "src.benchmark.runner.compute_omr_ned_xml",
        lambda *_args, **_kwargs: OMRNEDResult(
            omr_ned=9.5,
            edit_distance=1,
            pred_notation_size=4,
            gt_notation_size=4,
            parse_error=None,
            syntax_errors_fixed=0,
        ),
    )
    monkeypatch.setattr(
        "src.benchmark.runner.compute_tedn_from_musicxml",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("TEDn should be skipped")),
    )

    result = compute_metric_job(
        MetricJob(
            sample_index=0,
            sample_key="000000",
            source="",
            prediction="X:1\nC",
            ground_truth="4d",
            raw_format="abc",
            gold_xml_text="<score-partwise><part id='P1'/></score-partwise>",
            hum2xml_path="hum2xml",
            abc2xml_command=("abc2xml",),
            metrics=("omr_ned",),
        )
    )

    assert result.sample_metric.tedn is None
    assert result.sample_metric.omr_ned == 9.5
    assert result.pred_xml_text == "<score-partwise><part id='P1'/></score-partwise>"


def test_parallel_metrics_preserve_stable_order_and_use_spawn(monkeypatch, tmp_path: Path) -> None:
    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            output_root=str(tmp_path),
            models=("ours",),
            metric_workers=2,
        )
    )
    samples = [
        BenchmarkSample(0, "000000", "", "4c"),
        BenchmarkSample(1, "000001", "", "4d"),
    ]
    prediction_by_key = {
        "000000": {"prediction": "pred-0", "raw_format": "kern"},
        "000001": {"prediction": "pred-1", "raw_format": "kern"},
    }
    gold_xml = {
        "000000": "<score-partwise><part id='P1'/></score-partwise>",
        "000001": "<score-partwise><part id='P1'/></score-partwise>",
    }

    class FakeFuture:
        def __init__(self, value) -> None:
            self._value = value

        def result(self):
            return self._value

    class FakeExecutor:
        def __init__(self, *, max_workers, mp_context) -> None:
            self.max_workers = max_workers
            self.mp_context = mp_context
            captured["max_workers"] = max_workers
            captured["mp_context"] = mp_context

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, job):
            return FakeFuture(fn(job))

    captured: dict[str, object] = {}

    monkeypatch.setattr("src.benchmark.runner.multiprocessing.get_context", lambda mode: f"{mode}-context")
    monkeypatch.setattr("src.benchmark.runner.ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr("src.benchmark.runner.as_completed", lambda futures: list(reversed(list(futures))))
    monkeypatch.setattr(
        "src.benchmark.runner.compute_metric_job",
        lambda job: SimpleNamespace(
            sample_metric=SimpleNamespace(
                sample_index=job.sample_index,
                sample_key=job.sample_key,
                source=job.source,
                conversion_failed=False,
                conversion_error=None,
                tedn=float(job.sample_index),
                tedn_error=None,
                omr_ned=float(job.sample_index),
                omr_ned_error=None,
                cer=1.5 + job.sample_index,
            ),
            pred_xml_text=f"<xml>{job.sample_key}</xml>",
        ),
    )

    rows = runner._compute_metrics(
        dataset_name="synth",
        model_name="ours",
        samples=samples,
        gold_xml=gold_xml,
        prediction_by_key=prediction_by_key,
        raw_format="kern",
        pred_xml_dir=tmp_path / "pred_xml",
        metrics_start=0.0,
    )

    assert captured["max_workers"] == 2
    assert captured["mp_context"] == "spawn-context"
    assert [row.sample_key for row in rows] == ["000000", "000001"]
    assert (tmp_path / "pred_xml" / "000000.xml").read_text() == "<xml>000000</xml>"
    assert rows[0].cer == 1.5


def test_metric_pool_start_failure_raises_clear_message(monkeypatch, tmp_path: Path) -> None:
    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            output_root=str(tmp_path),
            models=("ours",),
            metric_workers=2,
        )
    )

    monkeypatch.setattr("src.benchmark.runner.multiprocessing.get_context", lambda mode: f"{mode}-context")

    def fail_executor(*_args, **_kwargs):
        raise RuntimeError("pool broke")

    monkeypatch.setattr("src.benchmark.runner.ProcessPoolExecutor", fail_executor)

    with pytest.raises(RuntimeError, match="--metric-workers 1"):
        runner._compute_metrics(
            dataset_name="synth",
            model_name="ours",
            samples=[BenchmarkSample(0, "000000", "", "4c")],
            gold_xml={"000000": "<score-partwise><part id='P1'/></score-partwise>"},
            prediction_by_key={"000000": {"prediction": "pred", "raw_format": "kern"}},
            raw_format="kern",
            pred_xml_dir=tmp_path / "pred_xml",
            metrics_start=0.0,
        )


def test_summary_table_shows_cer_for_ours_and_na_for_other_models(monkeypatch, tmp_path: Path) -> None:
    runner = BenchmarkRunner(BenchmarkRunnerConfig(output_root=str(tmp_path)))
    console = Console(force_terminal=False, record=True, width=160)
    monkeypatch.setattr("src.benchmark.runner.console", console)

    runner._print_summary_table(
        "mixed",
        [
            DatasetModelSummary(
                dataset_name="synth",
                model_name="ours",
                raw_format="kern",
                num_samples=1,
                conversion_success_rate=100.0,
                cer=12.5,
                tedn=5.0,
                omr_ned=7.0,
            ),
            DatasetModelSummary(
                dataset_name="synth",
                model_name="smtpp",
                raw_format="kern",
                num_samples=1,
                conversion_success_rate=100.0,
                tedn=8.0,
                omr_ned=9.0,
            ),
        ],
    )

    rendered = console.export_text()
    assert "CER" in rendered
    assert "12.50%" in rendered
    assert "N/A" in rendered


def test_summary_and_per_sample_csv_include_cer_column(tmp_path: Path) -> None:
    runner = BenchmarkRunner(BenchmarkRunnerConfig(output_root=str(tmp_path)))
    sample_rows = [
        SampleMetric(
            sample_index=0,
            sample_key="000000",
            source="",
            tedn=1.0,
            omr_ned=2.0,
            cer=3.0,
        )
    ]
    runner._write_sample_metrics(tmp_path / "per_sample_metrics.csv", sample_rows)

    summary = DatasetModelSummary(
        dataset_name="synth",
        model_name="ours",
        raw_format="kern",
        num_samples=1,
        conversion_success_rate=100.0,
        cer=3.0,
        tedn=1.0,
        omr_ned=2.0,
    )
    runner._write_summary_files([summary], summarize_overall([summary]))

    per_sample_header = (tmp_path / "per_sample_metrics.csv").read_text().splitlines()[0]
    summary_header = (runner.run_dir / "summary.csv").read_text().splitlines()[0]

    assert "cer" in per_sample_header.split(",")
    assert "cer" in summary_header.split(",")


def test_resume_reads_legacy_summary_without_cer(tmp_path: Path) -> None:
    runner = BenchmarkRunner(BenchmarkRunnerConfig(output_root=str(tmp_path), resume=True))
    summary_path = runner.run_dir / "synth" / "ours" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "dataset_name": "synth",
                "model_name": "ours",
                "raw_format": "kern",
                "num_samples": 1,
                "num_metric_worker_failures": 0,
                "num_conversion_failures": 0,
                "conversion_success_rate": 100.0,
                "tedn": 1.0,
                "omr_ned": 2.0,
                "num_loop_recovery_primary_flagged": 0,
                "num_loop_recovery_rerun_attempted": 0,
                "num_loop_recovery_recovered": 0,
                "num_loop_recovery_unrecovered": 0,
            }
        )
    )

    summary = runner._evaluate_model(dataset_name="synth", dataset=[], samples=[], model_name="ours")

    assert summary.cer is None
    assert summary.omr_ned == 2.0


def test_run_manifest_records_requested_and_effective_our_decode_policy(
    monkeypatch, tmp_path: Path
) -> None:
    dataset_root = tmp_path / "benchmark"
    (dataset_root / "synth").mkdir(parents=True)

    expected_policy = {
        "requested": {
            "constraints_enabled": False,
            "checkpoint": {"strategy": "greedy", "num_beams": 1},
            "overrides": {"strategy": "beam", "num_beams": 10},
            "resolved": {"strategy": "beam", "num_beams": 10},
        },
        "effective": {
            "constraints_enabled": False,
            "settings": {"strategy": "beam", "num_beams": 10},
            "downgraded_to_greedy": False,
        },
    }

    class FakeAdapter:
        raw_format = "kern"
        decode_policy = expected_policy

        def predict_batch(self, images):
            return []

    monkeypatch.setattr("src.benchmark.runner.load_from_disk", lambda _path: [])
    monkeypatch.setattr(
        "src.benchmark.runner.OurCheckpointAdapter",
        lambda *_args, **_kwargs: FakeAdapter(),
    )
    monkeypatch.setattr("src.benchmark.runner.BenchmarkRunner._print_summary_table", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "src.benchmark.runner.BenchmarkRunner._evaluate_model",
        lambda self, **_kwargs: DatasetModelSummary(
            dataset_name="synth",
            model_name="ours",
            raw_format="kern",
            num_samples=1,
            conversion_success_rate=100.0,
            omr_ned=0.0,
        ),
    )

    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            dataset_root=str(dataset_root),
            output_root=str(tmp_path / "outputs"),
            datasets=("synth",),
            models=("ours",),
            metrics=("omr_ned",),
            ours_strategy="beam",
            ours_num_beams=10,
            disable_constraints=True,
        )
    )

    runner.run()

    manifest = json.loads((runner.run_dir / "run_manifest.json").read_text())
    assert manifest["our_decode_policy"] == expected_policy


def test_run_manifest_records_legato_encoder_path(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path / "benchmark"
    (dataset_root / "synth").mkdir(parents=True)

    class FakeAdapter:
        raw_format = "abc"

        def predict_batch(self, images):
            return []

    monkeypatch.setattr("src.benchmark.runner.load_from_disk", lambda _path: [])
    monkeypatch.setattr(
        "src.benchmark.runner.LegatoAdapter",
        lambda *_args, **_kwargs: FakeAdapter(),
    )
    monkeypatch.setattr("src.benchmark.runner.BenchmarkRunner._print_summary_table", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "src.benchmark.runner.BenchmarkRunner._evaluate_model",
        lambda self, **_kwargs: DatasetModelSummary(
            dataset_name="synth",
            model_name="legato",
            raw_format="abc",
            num_samples=1,
            conversion_success_rate=100.0,
            omr_ned=0.0,
        ),
    )

    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            dataset_root=str(dataset_root),
            output_root=str(tmp_path / "outputs"),
            datasets=("synth",),
            models=("legato",),
            metrics=("omr_ned",),
            legato_model_id="/workspace/hf/legato",
            legato_encoder_path="/workspace/hf/llama32-11b-vision",
        )
    )

    runner.run()

    manifest = json.loads((runner.run_dir / "run_manifest.json").read_text())
    assert manifest["model_ids"]["legato"] == "/workspace/hf/legato"
    assert manifest["model_ids"]["legato_encoder"] == "/workspace/hf/llama32-11b-vision"


@pytest.mark.skipif(
    os.getenv("RUN_BENCHMARK_SMOKE") != "1"
    or shutil.which("hum2xml") is None
    or not is_musicdiff_available(),
    reason="Benchmark smoke test requires RUN_BENCHMARK_SMOKE=1, hum2xml, and musicdiff",
)
def test_benchmark_smoke_single_sample(tmp_path: Path) -> None:
    config = BenchmarkRunnerConfig(
        output_root=str(tmp_path / "outputs"),
        datasets=("synth",),
        models=("ours",),
        limit=1,
        metrics=("omr_ned",),
        ours_checkpoint="weights/long-data-non-augment.ckpt",
        metric_workers=1,
    )
    runner = BenchmarkRunner(config)
    payload = runner.run()
    assert payload["rows"]
    summary_path = Path(payload["run_dir"]) / "summary.json"
    assert summary_path.exists()
