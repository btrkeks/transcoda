import json
import os
import subprocess
from pathlib import Path

import pytest

from pipelines.datasets.generation.continuations import prepare, selected


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan"), float("inf")])
def test_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        selected("sample", fraction)


def test_hash_selection_is_stable_and_independent_of_order():
    samples = [f"score-{i}" for i in range(1000)]
    forward = {sample for sample in samples if selected(sample)}
    reverse = {sample for sample in reversed(samples) if selected(sample)}
    assert forward == reverse
    assert 200 < len(forward) < 300
    assert not any(selected(sample, 0) for sample in samples)
    assert all(selected(sample, 1) for sample in samples)


@pytest.fixture
def binary():
    path = Path(os.environ.get("CONTINUATION_PREPARE", "target/release/continuation_prepare"))
    if not path.is_file():
        pytest.fail(f"build continuation_prepare before running integration tests: {path}")
    return path.resolve()


def test_production_stage_keeps_openings_and_preserves_later_meter_change(tmp_path, binary):
    inputs, sources, out = (tmp_path / name for name in ("input", "sources", "out"))
    inputs.mkdir()
    sources.mkdir()
    text = "**kern\n*clefG2\n*k[f#]\n*M3/4\n=1\n2.c\n=2\n*M2/4\n2d\n*-\n"
    (inputs / "opening.krn").write_text(text)
    row = {
        "output_path": "opening.krn",
        "output_id": "opening",
        "sources": [{"source_path": "score.krn", "unit_kind": "original"}],
    }
    (inputs / "concatenation_manifest.jsonl").write_text(json.dumps(row) + "\n")
    summary = prepare(
        input_dir=inputs, source_dir=sources, output_dir=out, binary=binary, fraction=1
    )
    assert summary["output_rows"] == 2
    assert (out / "opening.krn").read_text() == text
    hidden = (out / "opening_continuation.krn").read_text()
    assert "*M3/4\n*met()" in hidden

    def target(path):
        result = subprocess.run(
            [
                str(binary.with_name("kern_parser")),
                "serialize-jsonl",
                "--preset",
                "transcoda-v2-training",
                "--input-dialect",
                "humdrum-kern",
            ],
            input=json.dumps({"row_index": 0, "transcription": path.read_text()}) + "\n",
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        return json.loads(result)["transcription"]

    normal = target(out / "opening.krn")
    continuation = target(out / "opening_continuation.krn")
    assert "*M3/4" not in continuation
    assert "*M2/4" in continuation
    assert normal.replace("*M3/4\n", "") == continuation
    assert "*M2/4\n" in hidden
    assert "*clefG2" in hidden and "*k[f#]" in hidden
    before = (out / "concatenation_manifest.jsonl").read_bytes()
    prepare(input_dir=inputs, source_dir=sources, output_dir=out, binary=binary, fraction=1)
    assert before == (out / "concatenation_manifest.jsonl").read_bytes()
    prepare(input_dir=inputs, source_dir=sources, output_dir=out, binary=binary, fraction=0)
    assert not (out / "opening_continuation.krn").exists()
    assert (out / "opening.krn").read_text() == text


def test_natural_continuation_with_authored_boundary_change_keeps_it(tmp_path, binary):
    inputs, sources, out = (tmp_path / name for name in ("input", "sources", "out"))
    inputs.mkdir()
    sources.mkdir()
    (sources / "score.krn").write_text("**kern\n*clefG2\n*k[]\n*M3/4\n=1\n2.c\n=2\n*M2/4\n2d\n*-\n")
    text = "**kern\n*clefG2\n*k[]\n*M2/4\n=2\n2d\n*-\n"
    (inputs / "page.krn").write_text(text)
    row = {
        "output_path": "page.krn",
        "output_id": "page",
        "sources": [
            {
                "source_path": "score.krn",
                "unit_kind": "split_chunk",
                "start_bar_index": 1,
                "end_bar_index": 2,
            }
        ],
    }
    (inputs / "concatenation_manifest.jsonl").write_text(json.dumps(row) + "\n")
    result = prepare(
        input_dir=inputs, source_dir=sources, output_dir=out, binary=binary, fraction=1
    )
    assert result["output_rows"] == 1
    assert "*M2/4\n" in (out / "page.krn").read_text()
    assert "*M2/4yy" not in (out / "page.krn").read_text()






def test_prepared_provenance_registers_the_actual_manifest(tmp_path, binary):
    inputs = tmp_path / "6_standalone"
    inputs.mkdir()
    (inputs / "page.krn").write_text("**kern\n*clefG2\n*M4/4\n1c\n*-\n")
    row = {
        "output_path": "page.krn",
        "output_id": "page",
        "sources": [{"source_path": "page.krn", "unit_kind": "original"}],
    }
    (inputs / "concatenation_manifest.jsonl").write_text(json.dumps(row) + "\n")
    out = tmp_path / "6_concatenated"
    prepare(input_dir=inputs, source_dir=tmp_path, output_dir=out, binary=binary, fraction=1)
    component = json.loads((tmp_path / "dataset_manifest.json").read_text())["components"][
        "concatenation_provenance"
    ]
    assert component["summary"] == {"input_count": 1, "output_count": 2, "failure_count": 0}
    assert component["paths"] == [
        "6_concatenated/concatenation_manifest.jsonl",
        "6_concatenated/continuations.json",
    ]


def test_natural_continuation_hides_its_carried_meter(tmp_path, binary):
    inputs, sources, out = (tmp_path / name for name in ("input", "sources", "out"))
    inputs.mkdir()
    sources.mkdir()
    (sources / "score.krn").write_text("**kern\n*clefG2\n*k[]\n*M3/4\n=1\n2.c\n=2\n2.d\n*-\n")
    (inputs / "page.krn").write_text("**kern\n*clefG2\n*k[]\n*M3/4\n=2\n2.d\n*-\n")
    row = {
        "output_path": "page.krn",
        "output_id": "page",
        "sources": [
            {
                "source_path": "score.krn",
                "unit_kind": "split_chunk",
                "start_bar_index": 1,
                "end_bar_index": 2,
            }
        ],
    }
    (inputs / "concatenation_manifest.jsonl").write_text(json.dumps(row) + "\n")
    result = prepare(
        input_dir=inputs, source_dir=sources, output_dir=out, binary=binary, fraction=1
    )
    assert result["natural_hidden"] == 1
    assert result["output_rows"] == 1
    hidden = (out / "page.krn").read_text()
    assert "*met()" in hidden
    label = subprocess.run(
        [
            str(binary.with_name("kern_parser")),
            "serialize-jsonl",
            "--preset",
            "transcoda-v2-training",
            "--input-dialect",
            "humdrum-kern",
        ],
        input=json.dumps({"row_index": 0, "transcription": (out / "page.krn").read_text()}) + "\n",
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    label = json.loads(label)["transcription"]
    assert "*M" not in label
    assert "*clefG2" in label and "*k[]" in label




def continuation_cohort(tmp_path, bad=1, good=0, natural=False):
    inputs, sources = tmp_path / "inputs", tmp_path / "sources"
    inputs.mkdir()
    sources.mkdir()
    rows = []
    for index in range(bad + good):
        name = f"page{index}.krn"
        text = "**kern\n*clefG2\n*M4/4\n1c\n*-\n"
        (inputs / name).write_text(text)
        (sources / name).write_text("**kern\n4.invalid\n*-\n" if index < bad else text)
        rows.append(
            {
                "output_path": name,
                "output_id": name,
                "sources": [
                    {"source_path": name, "start_bar_index": 1 if natural or index < bad else 0}
                ],
            }
        )
    (inputs / "concatenation_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    return inputs, sources


def test_failed_row_keeps_standalone_and_other_rows_continue(tmp_path, binary):
    inputs, sources = continuation_cohort(tmp_path, good=1)
    out = tmp_path / "out"
    summary = prepare(
        input_dir=inputs,
        source_dir=sources,
        output_dir=out,
        binary=binary,
        fraction=1,
        max_failure_fraction=0.5,
    )
    assert summary["failure_count"] == 1
    assert not summary["threshold_exceeded"]
    assert summary["failure_classes"] == {"parse": 1}
    assert (out / "page0.krn").read_bytes() == (inputs / "page0.krn").read_bytes()
    assert not (out / "page0_continuation.krn").exists()
    assert (out / "page1_continuation.krn").exists()
    rows = [
        json.loads(line) for line in (out / "concatenation_manifest.jsonl").read_text().splitlines()
    ]
    assert rows[0]["continuation"]["failure"]["class"] == "parse"
    assert json.loads((out / "continuations.json").read_text())["failure_count"] == 1


def test_stage_fails_only_above_threshold_after_recording_failures(tmp_path, binary):
    from pipelines.datasets.generation.continuations import ContinuationFailures

    inputs, sources = continuation_cohort(tmp_path, good=1)
    out = tmp_path / "out"
    with pytest.raises(ContinuationFailures) as caught:
        prepare(input_dir=inputs, source_dir=sources, output_dir=out, binary=binary, fraction=1)
    assert caught.value.summary["failure_count"] == 1
    assert caught.value.summary["max_failure_fraction"] == 0.005
    assert (out / "continuations.json").exists()
    assert (out / "page1_continuation.krn").exists()


def test_preflight_checks_every_row_without_writes_even_with_zero_fraction(tmp_path, binary):
    from pipelines.datasets.generation.continuations import ContinuationFailures

    inputs, sources = continuation_cohort(tmp_path, good=1)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    out = tmp_path / "does-not-exist"
    with pytest.raises(ContinuationFailures) as caught:
        prepare(
            input_dir=inputs,
            source_dir=sources,
            output_dir=out,
            binary=binary,
            fraction=0,
            preflight=True,
        )
    assert caught.value.summary["fraction"] == 1.0
    assert caught.value.summary["failure_count"] == 1
    assert not out.exists()
    after = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before


def test_preflight_can_run_on_input_without_any_output_directory(tmp_path, binary):
    inputs, sources = continuation_cohort(tmp_path, bad=0, good=1)
    result = prepare(input_dir=inputs, source_dir=sources, binary=binary, preflight=True)
    assert result["failure_count"] == 0
    assert result["preflight"]


def test_panics_are_classified():
    from pipelines.datasets.generation.continuations import failure_class

    assert failure_class("thread main panicked at split.rs", 101) == "panic"
