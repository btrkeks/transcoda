import json

from scripts.dataset_generation.normalization import main as normalization_main


def test_normalize_kern_files_writes_stats_json(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    stats_path = tmp_path / "stats" / "normalize.json"

    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "good.krn").write_text("good", encoding="utf-8")
    (input_dir / "bad.krn").write_text("bad", encoding="utf-8")

    def fake_normalize_file(input_path, output_path):
        if input_path.name == "bad.krn":
            return input_path, False, "ValueError", "boom", None
        output_path.write_text("normalized", encoding="utf-8")
        return input_path, True, None, None, None

    monkeypatch.setattr(normalization_main, "normalize_file", fake_normalize_file)

    normalization_main.normalize_kern_files(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        workers=1,
        quiet=True,
        stats_json=str(stats_path),
    )

    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    assert payload["input_count"] == 2
    assert payload["success_count"] == 1
    assert payload["error_count"] == 1
    assert payload["error_types"]["ValueError"] == 1
    assert payload["failed_examples"][0]["file"].endswith("bad.krn")
