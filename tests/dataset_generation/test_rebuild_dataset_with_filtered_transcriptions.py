import json
from pathlib import Path

from datasets import Dataset, load_from_disk
from PIL import Image

from scripts.dataset_generation.rebuild_dataset_with_filtered_transcriptions import (
    rebuild_dataset_with_filtered_transcriptions,
)


def _build_dataset(rows):
    return Dataset.from_list(rows)


def _write_krn(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_rebuild_dataset_with_filtered_transcriptions_recovers_rows_and_preserves_schema(
    tmp_path,
):
    dataset_dir = tmp_path / "datasets" / "train_full"
    output_dir = tmp_path / "datasets" / "train_full_filtered"
    manifest_path = tmp_path / "reports" / "latest.json"
    interim_root = tmp_path / "interim" / "train"

    norm_unique = "*clefG2\n4c\n"
    norm_collapsed = "*clefG2\n4d\n"
    norm_ambiguous = "*clefG2\n4e\n"
    missing = "*clefG2\n4f\n"

    filtered_unique = "**kern\n=1\n4c\n"
    filtered_collapsed = "**kern\n=1\n4d\n"
    filtered_ambig_a = "**kern\n=1\n4eA\n"
    filtered_ambig_b = "**kern\n=1\n4eB\n"

    _write_krn(interim_root / "grandstaff" / "3_normalized" / "unique.krn", norm_unique)
    _write_krn(interim_root / "grandstaff" / "2_filtered" / "unique.krn", filtered_unique)

    _write_krn(interim_root / "grandstaff" / "3_normalized" / "collapsed_a.krn", norm_collapsed)
    _write_krn(interim_root / "grandstaff" / "2_filtered" / "collapsed_a.krn", filtered_collapsed)
    _write_krn(interim_root / "pdmx" / "3_normalized" / "collapsed_b.krn", norm_collapsed)
    _write_krn(interim_root / "pdmx" / "2_filtered" / "collapsed_b.krn", filtered_collapsed)

    _write_krn(interim_root / "grandstaff" / "3_normalized" / "ambig_a.krn", norm_ambiguous)
    _write_krn(interim_root / "grandstaff" / "2_filtered" / "ambig_a.krn", filtered_ambig_a)
    _write_krn(interim_root / "pdmx" / "3_normalized" / "ambig_b.krn", norm_ambiguous)
    _write_krn(interim_root / "pdmx" / "2_filtered" / "ambig_b.krn", filtered_ambig_b)

    dataset = _build_dataset(
        [
            {
                "image": Image.new("RGB", (10, 10), color=(255, 255, 255)),
                "transcription": norm_unique,
                "extra": "u",
            },
            {
                "image": Image.new("RGB", (10, 10), color=(254, 254, 254)),
                "transcription": norm_collapsed,
                "extra": "c",
            },
            {
                "image": Image.new("RGB", (10, 10), color=(253, 253, 253)),
                "transcription": norm_ambiguous,
                "extra": "a",
            },
            {
                "image": Image.new("RGB", (10, 10), color=(252, 252, 252)),
                "transcription": missing,
                "extra": "m",
            },
        ]
    )
    dataset.save_to_disk(str(dataset_dir))

    summary = rebuild_dataset_with_filtered_transcriptions(
        dataset_path=str(dataset_dir),
        interim_root=str(interim_root),
        source_datasets="grandstaff,pdmx",
        output_dir=str(output_dir),
        manifest_out=str(manifest_path),
        seed=11,
        quiet=True,
    )

    rebuilt = load_from_disk(str(output_dir))
    assert rebuilt.column_names == ["image", "transcription", "extra"]
    assert len(rebuilt) == 3
    assert rebuilt["extra"] == ["u", "c", "a"]
    assert rebuilt["transcription"][0] == filtered_unique
    assert rebuilt["transcription"][1] == filtered_collapsed
    assert rebuilt["transcription"][2] in {filtered_ambig_a, filtered_ambig_b}

    assert summary["input_row_count"] == 4
    assert summary["output_row_count"] == 3
    assert summary["skipped_row_count"] == 1
    assert summary["ambiguous_row_count"] == 1
    assert summary["unique_match_row_count"] == 2
    assert summary["scanned_source_files"] == 5
    assert summary["paired_source_files"] == 5
    assert summary["ignored_source_files"] == 0

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload == summary

    audit_path = manifest_path.with_name("latest.rows.jsonl")
    audit_rows = [
        json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["status"] for row in audit_rows] == [
        "unique_match",
        "unique_match",
        "ambiguous_match",
        "missing_match",
    ]
    assert audit_rows[0]["output_row_index"] == 0
    assert audit_rows[1]["output_row_index"] == 1
    assert audit_rows[2]["output_row_index"] == 2
    assert audit_rows[3]["output_row_index"] is None
    assert audit_rows[1]["distinct_filtered_candidate_count"] == 1
    assert len(audit_rows[1]["candidate_normalized_sources"]) == 2
    assert audit_rows[2]["distinct_filtered_candidate_count"] == 2
    assert len(audit_rows[2]["candidate_filtered_representatives"]) == 2


def test_rebuild_dataset_with_filtered_transcriptions_is_deterministic_for_ambiguous_rows(
    tmp_path,
):
    dataset_dir = tmp_path / "datasets" / "train_full"
    output_a = tmp_path / "datasets" / "out_a"
    output_b = tmp_path / "datasets" / "out_b"
    manifest_a = tmp_path / "reports" / "a.json"
    manifest_b = tmp_path / "reports" / "b.json"
    interim_root = tmp_path / "interim" / "train"

    norm_ambiguous = "*clefG2\n4e\n"
    filtered_ambig_a = "**kern\n=1\n4eA\n"
    filtered_ambig_b = "**kern\n=1\n4eB\n"

    _write_krn(interim_root / "grandstaff" / "3_normalized" / "ambig_a.krn", norm_ambiguous)
    _write_krn(interim_root / "grandstaff" / "2_filtered" / "ambig_a.krn", filtered_ambig_a)
    _write_krn(interim_root / "pdmx" / "3_normalized" / "ambig_b.krn", norm_ambiguous)
    _write_krn(interim_root / "pdmx" / "2_filtered" / "ambig_b.krn", filtered_ambig_b)

    dataset = _build_dataset(
        [
            {
                "image": Image.new("RGB", (10, 10), color=(255, 255, 255)),
                "transcription": norm_ambiguous,
                "extra": "a",
            }
        ]
    )
    dataset.save_to_disk(str(dataset_dir))

    rebuild_dataset_with_filtered_transcriptions(
        dataset_path=str(dataset_dir),
        interim_root=str(interim_root),
        source_datasets="grandstaff,pdmx",
        output_dir=str(output_a),
        manifest_out=str(manifest_a),
        seed=17,
        quiet=True,
    )
    rebuild_dataset_with_filtered_transcriptions(
        dataset_path=str(dataset_dir),
        interim_root=str(interim_root),
        source_datasets="grandstaff,pdmx",
        output_dir=str(output_b),
        manifest_out=str(manifest_b),
        seed=17,
        quiet=True,
    )

    ds_a = load_from_disk(str(output_a))
    ds_b = load_from_disk(str(output_b))
    assert ds_a["transcription"] == ds_b["transcription"]

    audit_a = manifest_a.with_name("a.rows.jsonl").read_text(encoding="utf-8")
    audit_b = manifest_b.with_name("b.rows.jsonl").read_text(encoding="utf-8")
    assert audit_a == audit_b


def test_rebuild_dataset_with_filtered_transcriptions_prefers_direct_provenance(tmp_path):
    dataset_dir = tmp_path / "datasets" / "train_full"
    output_dir = tmp_path / "datasets" / "train_full_filtered"
    manifest_path = tmp_path / "reports" / "latest.json"
    interim_root = tmp_path / "interim" / "train"

    direct_filtered = "**kern\n=1\n4g\n"
    _write_krn(interim_root / "grandstaff" / "2_filtered" / "known.krn", direct_filtered)

    dataset = _build_dataset(
        [
            {
                "image": Image.new("RGB", (10, 10), color=(255, 255, 255)),
                "transcription": "this should be ignored",
                "source_dataset": "grandstaff",
                "source": "known.krn",
            }
        ]
    )
    dataset.save_to_disk(str(dataset_dir))

    rebuild_dataset_with_filtered_transcriptions(
        dataset_path=str(dataset_dir),
        interim_root=str(interim_root),
        source_datasets="grandstaff",
        output_dir=str(output_dir),
        manifest_out=str(manifest_path),
        quiet=True,
    )

    rebuilt = load_from_disk(str(output_dir))
    assert rebuilt["transcription"] == [direct_filtered]

    audit_row = json.loads(
        manifest_path.with_name("latest.rows.jsonl").read_text(encoding="utf-8").strip()
    )
    assert audit_row["status"] == "unique_match"
    assert audit_row["chosen_filtered_source"].endswith("grandstaff/2_filtered/known.krn")


def test_rebuild_dataset_with_filtered_transcriptions_ignores_missing_filtered_sources(tmp_path):
    dataset_dir = tmp_path / "datasets" / "train_full"
    output_dir = tmp_path / "datasets" / "train_full_filtered"
    manifest_path = tmp_path / "reports" / "latest.json"
    interim_root = tmp_path / "interim" / "train"

    _write_krn(interim_root / "grandstaff" / "3_normalized" / "dangling.krn", "*clefG2\n4a\n")
    _write_krn(interim_root / "grandstaff" / "3_normalized" / "paired.krn", "*clefG2\n4b\n")
    _write_krn(interim_root / "grandstaff" / "2_filtered" / "paired.krn", "**kern\n=1\n4b\n")

    dataset = _build_dataset(
        [
            {
                "image": Image.new("RGB", (10, 10), color=(255, 255, 255)),
                "transcription": "*clefG2\n4b\n",
            }
        ]
    )
    dataset.save_to_disk(str(dataset_dir))

    summary = rebuild_dataset_with_filtered_transcriptions(
        dataset_path=str(dataset_dir),
        interim_root=str(interim_root),
        source_datasets="grandstaff",
        output_dir=str(output_dir),
        manifest_out=str(manifest_path),
        quiet=True,
    )

    rebuilt = load_from_disk(str(output_dir))
    assert len(rebuilt) == 1
    assert rebuilt["transcription"] == ["**kern\n=1\n4b\n"]
    assert summary["scanned_source_files"] == 2
    assert summary["paired_source_files"] == 1
    assert summary["ignored_source_files"] == 1
