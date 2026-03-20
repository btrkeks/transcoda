"""Unit tests for WandbVisualizerCallback."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from lightning.pytorch.loggers import WandbLogger

from src.callbacks.wandb_visualizer import WandbVisualizerCallback, _HeapEntry


@pytest.fixture
def mock_i2w():
    """Simple index-to-word mapping for testing."""
    return {
        0: "<pad>",
        1: "<bos>",
        2: "<eos>",
        3: "4",
        4: "c",
        5: "d",
        6: "e",
        7: "f",
    }


@pytest.fixture
def callback(mock_i2w):
    """Create a WandbVisualizerCallback instance for testing."""
    return WandbVisualizerCallback(
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        i2w=mock_i2w,
        n_best=2,
        n_worst=3,
    )


@pytest.fixture
def mock_trainer():
    """Create a mock Lightning trainer."""
    trainer = Mock()
    logger = Mock(spec=WandbLogger)
    logger.experiment = Mock()
    trainer.logger = logger
    return trainer


@pytest.fixture
def mock_pl_module():
    """Create a mock LightningModule with training config."""
    module = Mock()
    module.hparams.training.log_example_images = True
    module.current_epoch = 5
    module.should_log_validation_examples.return_value = True
    return module


def _make_outputs(val_set_name, sample_ids, cers, cers_no_ties_beams=None, sources=None):
    """Helper to build a full outputs dict for on_validation_batch_end."""
    n = len(sample_ids)
    outputs = {
        "val_set_name": val_set_name,
        "sample_ids": torch.tensor(sample_ids),
        "cers": cers,
        "pred_ids": [[1, 3, 4, 2] for _ in range(n)],
        "gt_ids": [[1, 3, 5, 2] for _ in range(n)],
    }
    if cers_no_ties_beams is not None:
        outputs["cers_no_ties_beams"] = cers_no_ties_beams
    if sources is not None:
        outputs["sources"] = sources
    return outputs


def _make_batch(n: int, *, n_channels=3, h=64, w=64):
    return {"pixel_values": torch.randn(n, n_channels, h, w, dtype=torch.float32)}


def test_initialization(callback, mock_i2w):
    """Test callback initialization with correct parameters."""
    assert callback.pad_token_id == 0
    assert callback.bos_token_id == 1
    assert callback.eos_token_id == 2
    assert callback.i2w == mock_i2w
    assert callback.n_best == 2
    assert callback.n_worst == 3
    assert callback._best_heaps == {}
    assert callback._worst_heaps == {}


def test_should_log_examples_all_conditions_met(callback, mock_trainer, mock_pl_module):
    """Test _should_log_examples returns True when all conditions are met."""
    # Put something in a heap
    callback._best_heaps.setdefault("synth", []).append(
        _HeapEntry(
            sort_key=(-0.1, "synth", 1),
            image=torch.zeros(3, 8, 8),
            pred_ids=[1],
            gt_ids=[1],
            cer=0.1,
            cer_no_ties_beams=None,
            val_set_name="synth",
            sample_id=1,
        )
    )
    assert callback._should_log_examples(mock_trainer, mock_pl_module) is True


def test_should_log_examples_logging_disabled(callback, mock_trainer, mock_pl_module):
    """Test _should_log_examples returns False when logging is disabled in config."""
    mock_pl_module.should_log_validation_examples.return_value = False
    callback._best_heaps.setdefault("synth", []).append(
        _HeapEntry(
            sort_key=(-0.1, "synth", 1),
            image=torch.zeros(3, 8, 8),
            pred_ids=[1],
            gt_ids=[1],
            cer=0.1,
            cer_no_ties_beams=None,
            val_set_name="synth",
            sample_id=1,
        )
    )
    assert callback._should_log_examples(mock_trainer, mock_pl_module) is False


def test_should_log_examples_no_logger(callback, mock_pl_module):
    """Test _should_log_examples returns False when no logger is attached."""
    mock_trainer = Mock()
    mock_trainer.logger = None
    callback._best_heaps.setdefault("synth", []).append(
        _HeapEntry(
            sort_key=(-0.1, "synth", 1),
            image=torch.zeros(3, 8, 8),
            pred_ids=[1],
            gt_ids=[1],
            cer=0.1,
            cer_no_ties_beams=None,
            val_set_name="synth",
            sample_id=1,
        )
    )
    assert callback._should_log_examples(mock_trainer, mock_pl_module) is False


def test_should_log_examples_wrong_logger_type(callback, mock_pl_module):
    """Test _should_log_examples returns False for non-WandB loggers."""
    mock_trainer = Mock()
    mock_trainer.logger = Mock()  # Not a WandbLogger
    callback._best_heaps.setdefault("synth", []).append(
        _HeapEntry(
            sort_key=(-0.1, "synth", 1),
            image=torch.zeros(3, 8, 8),
            pred_ids=[1],
            gt_ids=[1],
            cer=0.1,
            cer_no_ties_beams=None,
            val_set_name="synth",
            sample_id=1,
        )
    )
    assert callback._should_log_examples(mock_trainer, mock_pl_module) is False


def test_should_log_examples_empty_heaps(callback, mock_trainer, mock_pl_module):
    """Test _should_log_examples returns False when heaps are empty."""
    assert callback._best_heaps == {}
    assert callback._worst_heaps == {}
    assert callback._should_log_examples(mock_trainer, mock_pl_module) is False


def test_on_validation_batch_end_stores_records(callback, mock_trainer, mock_pl_module):
    """Test on_validation_batch_end populates heaps."""
    outputs = _make_outputs("synth", [11, 12], [0.3, 0.7])
    callback.on_validation_batch_end(mock_trainer, mock_pl_module, outputs, _make_batch(2), 0)

    assert len(callback._best_heaps["synth"]) == 2
    assert len(callback._worst_heaps["synth"]) == 2


def test_on_validation_batch_end_stores_optional_stripped_cer(
    callback, mock_trainer, mock_pl_module
):
    outputs = _make_outputs("polish", [11], [0.7], cers_no_ties_beams=[0.4])
    callback.on_validation_batch_end(mock_trainer, mock_pl_module, outputs, _make_batch(1), 0)

    best_entry = callback._best_heaps["polish"][0]
    worst_entry = callback._worst_heaps["polish"][0]
    assert best_entry.cer_no_ties_beams == pytest.approx(0.4)
    assert worst_entry.cer_no_ties_beams == pytest.approx(0.4)


def test_on_validation_batch_end_raises_on_mismatched_field_lengths(
    callback, mock_trainer, mock_pl_module
):
    """Test on_validation_batch_end rejects mismatched per-sample field lengths."""
    outputs = {
        "val_set_name": "synth",
        "sample_ids": torch.tensor([1, 2, 3]),
        "cers": [0.1, 0.2, 0.3],
        "pred_ids": [[1, 2], [1, 2]],  # only 2 instead of 3
        "gt_ids": [[1, 2], [1, 2], [1, 2]],
    }
    with pytest.raises(ValueError, match="matching lengths"):
        callback.on_validation_batch_end(mock_trainer, mock_pl_module, outputs, _make_batch(3), 0)


def test_on_validation_batch_end_raises_on_mismatched_optional_field_lengths(
    callback, mock_trainer, mock_pl_module
):
    outputs = _make_outputs("polish", [1, 2], [0.1, 0.2], cers_no_ties_beams=[0.1])
    with pytest.raises(ValueError, match="matching lengths"):
        callback.on_validation_batch_end(mock_trainer, mock_pl_module, outputs, _make_batch(2), 0)


def test_on_validation_batch_end_raises_on_mismatched_sources_lengths(
    callback, mock_trainer, mock_pl_module
):
    outputs = _make_outputs("synth", [1, 2], [0.1, 0.2], sources=["train_000001.krn"])
    with pytest.raises(ValueError, match="matching lengths"):
        callback.on_validation_batch_end(mock_trainer, mock_pl_module, outputs, _make_batch(2), 0)


def test_on_validation_batch_end_ignores_none_outputs(callback, mock_trainer, mock_pl_module):
    """Test on_validation_batch_end ignores None outputs."""
    callback.on_validation_batch_end(mock_trainer, mock_pl_module, None, _make_batch(1), 0)
    assert callback._best_heaps == {}
    assert callback._worst_heaps == {}


def test_on_validation_batch_end_respects_logging_flag(callback, mock_trainer, mock_pl_module):
    """Test on_validation_batch_end doesn't store when logging is disabled."""
    mock_pl_module.should_log_validation_examples.return_value = False
    outputs = _make_outputs("synth", [1], [0.1])
    callback.on_validation_batch_end(mock_trainer, mock_pl_module, outputs, _make_batch(1), 0)
    assert callback._best_heaps == {}
    assert callback._worst_heaps == {}


def test_best_heap_keeps_lowest_cer(mock_i2w, mock_trainer, mock_pl_module):
    """Best heap with capacity 2 should retain only the 2 lowest-CER samples."""
    cb = WandbVisualizerCallback(
        pad_token_id=0, bos_token_id=1, eos_token_id=2, i2w=mock_i2w, n_best=2, n_worst=2
    )
    # Push 5 samples with CER 0.1..0.5
    for sid, cer in [(1, 0.5), (2, 0.3), (3, 0.1), (4, 0.4), (5, 0.2)]:
        outputs = _make_outputs("synth", [sid], [cer])
        cb.on_validation_batch_end(mock_trainer, mock_pl_module, outputs, _make_batch(1), 0)

    assert len(cb._best_heaps["synth"]) == 2
    retained_cers = sorted(e.cer for e in cb._best_heaps["synth"])
    assert retained_cers == [0.1, 0.2]


def test_worst_heap_keeps_highest_cer(mock_i2w, mock_trainer, mock_pl_module):
    """Worst heap with capacity 3 should retain only the 3 highest-CER samples."""
    cb = WandbVisualizerCallback(
        pad_token_id=0, bos_token_id=1, eos_token_id=2, i2w=mock_i2w, n_best=2, n_worst=3
    )
    # Push 5 samples with CER 0.1..0.5
    for sid, cer in [(1, 0.5), (2, 0.3), (3, 0.1), (4, 0.4), (5, 0.2)]:
        outputs = _make_outputs("synth", [sid], [cer])
        cb.on_validation_batch_end(mock_trainer, mock_pl_module, outputs, _make_batch(1), 0)

    assert len(cb._worst_heaps["synth"]) == 3
    retained_cers = sorted(e.cer for e in cb._worst_heaps["synth"])
    assert retained_cers == [0.3, 0.4, 0.5]


def test_heap_deterministic_tie_break(mock_i2w, mock_trainer, mock_pl_module):
    """Samples with equal CER should be tie-broken deterministically by id within a set."""
    cb = WandbVisualizerCallback(
        pad_token_id=0, bos_token_id=1, eos_token_id=2, i2w=mock_i2w, n_best=2, n_worst=2
    )
    # Push 4 samples all with CER=0.5 in a single set, different ids.
    for sid in [10, 3, 1, 7]:
        outputs = _make_outputs("synth", [sid], [0.5])
        cb.on_validation_batch_end(mock_trainer, mock_pl_module, outputs, _make_batch(1), 0)

    # Best heap: sort_key=(-0.5, "synth", id), so largest ids win.
    best_ids = sorted(e.sample_id for e in cb._best_heaps["synth"])
    assert best_ids == [7, 10]

    # Worst heap: sort_key=(0.5, "synth", id), so largest ids win.
    worst_ids = sorted(e.sample_id for e in cb._worst_heaps["synth"])
    assert worst_ids == [7, 10]


def test_heaps_are_independent_per_dataset(mock_i2w, mock_trainer, mock_pl_module):
    cb = WandbVisualizerCallback(
        pad_token_id=0, bos_token_id=1, eos_token_id=2, i2w=mock_i2w, n_best=1, n_worst=1
    )
    for val_set_name, sid, cer in [
        ("synth", 1, 0.1),
        ("synth", 2, 0.2),
        ("polish", 3, 0.7),
        ("polish", 4, 0.8),
    ]:
        outputs = _make_outputs(val_set_name, [sid], [cer])
        cb.on_validation_batch_end(mock_trainer, mock_pl_module, outputs, _make_batch(1), 0)

    assert len(cb._best_heaps["synth"]) == 1
    assert len(cb._worst_heaps["synth"]) == 1
    assert len(cb._best_heaps["polish"]) == 1
    assert len(cb._worst_heaps["polish"]) == 1
    assert cb._best_heaps["synth"][0].cer == 0.1
    assert cb._worst_heaps["synth"][0].cer == 0.2
    assert cb._best_heaps["polish"][0].cer == 0.7
    assert cb._worst_heaps["polish"][0].cer == 0.8


def test_token_ids_to_string(callback):
    """Test token ID to string conversion."""
    token_ids = [1, 3, 4, 5, 2, 0, 0]  # bos, 4, c, d, eos, pad, pad
    result = callback._token_ids_to_string(token_ids)
    assert isinstance(result, str)
    assert "<pad>" not in result
    assert "<bos>" not in result
    assert "<eos>" not in result


@patch("src.callbacks.wandb_visualizer.wandb.Table")
@patch("src.callbacks.wandb_visualizer.wandb.Image")
@patch("src.callbacks.wandb_visualizer.wandb.Html")
@patch("src.callbacks.wandb_visualizer.generate_html_diff")
def test_build_table_from_heaps(mock_diff, mock_html, mock_image, mock_table, callback):
    """Test _build_table_from_heaps creates correct table from heap contents."""
    table_instance = Mock()
    mock_table.return_value = table_instance
    mock_diff.return_value = "<html/>"
    mock_html.side_effect = lambda x: x
    mock_image.side_effect = lambda x: x

    # Manually populate heaps: 2 best + 3 worst
    for sid, cer in [(1, 0.05), (2, 0.10)]:
        callback._best_heaps.setdefault("synth", []).append(
            _HeapEntry(
                sort_key=(-cer, "synth", sid),
                image=torch.randn(3, 8, 8),
                pred_ids=[1, 3, 4, 2],
                gt_ids=[1, 3, 5, 2],
                cer=cer,
            cer_no_ties_beams=None,
                val_set_name="synth",
                sample_id=sid,
            )
        )
    for sid, cer in [(10, 0.9), (11, 0.8), (12, 0.7)]:
        callback._worst_heaps.setdefault("synth", []).append(
            _HeapEntry(
                sort_key=(cer, "synth", sid),
                image=torch.randn(3, 8, 8),
                pred_ids=[1, 3, 4, 2],
                gt_ids=[1, 3, 5, 2],
                cer=cer,
            cer_no_ties_beams=None,
                val_set_name="synth",
                sample_id=sid,
            )
        )

    table = callback._build_table_from_heaps(current_epoch=5)

    mock_table.assert_called_once()
    call_args = mock_table.call_args
    assert "ID" in call_args.kwargs["columns"]
    assert "Category" in call_args.kwargs["columns"]
    assert "CER" in call_args.kwargs["columns"]
    assert "CER_no_ties_beams" in call_args.kwargs["columns"]
    assert "SER" in call_args.kwargs["columns"]
    assert "Diff" in call_args.kwargs["columns"]
    assert "Prediction Rendered Image" in call_args.kwargs["columns"]
    assert len(call_args.kwargs["data"]) == 5  # 2 best + 3 worst
    assert mock_image.call_count == 5
    assert table is table_instance


def test_on_validation_epoch_end_full_workflow(callback, mock_trainer, mock_pl_module):
    """Test full validation epoch end workflow: build table, log, clear heaps."""
    # Populate heaps
    for sid, cer in [(1, 0.1), (2, 0.2)]:
        callback._best_heaps.setdefault("synth", []).append(
            _HeapEntry(
                sort_key=(-cer, "synth", sid),
                image=torch.randn(3, 8, 8),
                pred_ids=[1, 3, 4, 2],
                gt_ids=[1, 3, 5, 2],
                cer=cer,
            cer_no_ties_beams=None,
                val_set_name="synth",
                sample_id=sid,
            )
        )
    callback._worst_heaps.setdefault("synth", []).append(
        _HeapEntry(
            sort_key=(0.9, "synth", 3),
            image=torch.randn(3, 8, 8),
            pred_ids=[1, 3, 4, 2],
            gt_ids=[1, 3, 5, 2],
            cer=0.9,
            cer_no_ties_beams=None,
            val_set_name="synth",
            sample_id=3,
        )
    )

    with patch.object(callback, "_build_table_from_heaps", return_value=Mock()) as mock_build:
        callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

    mock_build.assert_called_once_with(current_epoch=5)
    mock_trainer.logger.experiment.log.assert_called_once()
    assert callback._best_heaps == {}
    assert callback._worst_heaps == {}


def test_on_validation_epoch_end_skips_when_should_not_log(callback, mock_trainer, mock_pl_module):
    """Test on_validation_epoch_end skips processing when logging disabled."""
    mock_pl_module.should_log_validation_examples.return_value = False
    # Put something in a heap to verify it gets cleared
    callback._best_heaps.setdefault("synth", []).append(
        _HeapEntry(
            sort_key=(-0.1, "synth", 1),
            image=torch.zeros(3, 8, 8),
            pred_ids=[1],
            gt_ids=[1],
            cer=0.1,
            cer_no_ties_beams=None,
            val_set_name="synth",
            sample_id=1,
        )
    )

    with patch.object(callback, "_build_table_from_heaps") as mock_build:
        callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

    mock_build.assert_not_called()
    mock_trainer.logger.experiment.log.assert_not_called()
    assert callback._best_heaps == {}
    assert callback._worst_heaps == {}


def test_module_should_log_examples_falls_back_to_hparams(callback):
    module = SimpleNamespace(hparams=SimpleNamespace(training=SimpleNamespace(log_example_images=True)))

    assert callback._module_should_log_examples(module) is True


def test_on_validation_batch_end_uses_batch_images(callback, mock_trainer, mock_pl_module):
    outputs = _make_outputs("synth", [1], [0.1])
    batch = {"pixel_values": torch.ones(1, 3, 8, 8, dtype=torch.float32)}

    callback.on_validation_batch_end(mock_trainer, mock_pl_module, outputs, batch, 0)

    assert callback._best_heaps["synth"]
    best_image = callback._best_heaps["synth"][0].image
    assert best_image.dtype == torch.float16
    assert torch.all(best_image == 1.0)


@patch("src.callbacks.wandb_visualizer.wandb.Image")
@patch("src.callbacks.wandb_visualizer.wandb.Html")
@patch("src.callbacks.wandb_visualizer.generate_html_diff")
def test_entry_to_row_uses_source_name_for_id_with_basename(
    mock_diff, mock_html, mock_image, callback
):
    mock_diff.return_value = "<html/>"
    mock_html.side_effect = lambda x: x
    mock_image.side_effect = lambda x: x
    entry = _HeapEntry(
        sort_key=(-0.1, "polish", 86),
        image=torch.randn(3, 8, 8),
        pred_ids=[1, 3, 4, 2],
        gt_ids=[1, 3, 5, 2],
        cer=0.1,
        cer_no_ties_beams=None,
        val_set_name="polish",
        sample_id=86,
        source_name="/tmp/train_000081.krn",
    )

    row = callback._entry_to_row(entry, category="best", current_epoch=0)
    assert row[0] == "train_000081.krn"


@patch("src.callbacks.wandb_visualizer.wandb.Image")
@patch("src.callbacks.wandb_visualizer.wandb.Html")
@patch("src.callbacks.wandb_visualizer.generate_html_diff")
def test_entry_to_row_falls_back_to_legacy_id_when_source_missing(
    mock_diff, mock_html, mock_image, callback
):
    mock_diff.return_value = "<html/>"
    mock_html.side_effect = lambda x: x
    mock_image.side_effect = lambda x: x
    entry = _HeapEntry(
        sort_key=(-0.1, "polish", 86),
        image=torch.randn(3, 8, 8),
        pred_ids=[1, 3, 4, 2],
        gt_ids=[1, 3, 5, 2],
        cer=0.1,
        cer_no_ties_beams=None,
        val_set_name="polish",
        sample_id=86,
        source_name=None,
    )

    row = callback._entry_to_row(entry, category="best", current_epoch=0)
    assert row[0] == "epoch_0_polish_ex_86"


def test_on_validation_batch_end_skips_image_copy_when_not_retained(
    mock_i2w, mock_trainer, mock_pl_module
):
    cb = WandbVisualizerCallback(
        pad_token_id=0, bos_token_id=1, eos_token_id=2, i2w=mock_i2w, n_best=1, n_worst=1
    )
    cb._best_heaps.setdefault("synth", []).append(
        _HeapEntry(
            sort_key=(-0.1, "synth", 1),
            image=torch.zeros(3, 8, 8),
            pred_ids=[1],
            gt_ids=[1],
            cer=0.1,
            cer_no_ties_beams=None,
            val_set_name="synth",
            sample_id=1,
        )
    )
    cb._worst_heaps.setdefault("synth", []).append(
        _HeapEntry(
            sort_key=(0.95, "synth", 2),
            image=torch.zeros(3, 8, 8),
            pred_ids=[1],
            gt_ids=[1],
            cer=0.95,
            cer_no_ties_beams=None,
            val_set_name="synth",
            sample_id=2,
        )
    )

    outputs = _make_outputs("synth", [3], [0.9])
    # With CER=0.9, sample is neither better than best(0.1) nor worse than worst(0.95).
    cb.on_validation_batch_end(mock_trainer, mock_pl_module, outputs, {}, 0)

    assert len(cb._best_heaps["synth"]) == 1
    assert len(cb._worst_heaps["synth"]) == 1


def test_on_validation_batch_end_allows_zero_heap_capacities(mock_i2w, mock_trainer, mock_pl_module):
    cb = WandbVisualizerCallback(
        pad_token_id=0, bos_token_id=1, eos_token_id=2, i2w=mock_i2w, n_best=0, n_worst=0
    )
    outputs = _make_outputs("synth", [1], [0.4])

    cb.on_validation_batch_end(mock_trainer, mock_pl_module, outputs, {}, 0)

    assert cb._best_heaps == {}
    assert cb._worst_heaps == {}
