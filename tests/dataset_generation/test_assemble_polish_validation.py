from scripts.dataset_generation import assemble_polish_validation as apv


def test_assemble_polish_validation_wraps_split_aware_builder(monkeypatch, tmp_path):
    calls = []

    def fake_builder(**kwargs):
        calls.append(kwargs)
        return {"num_rows": 1}

    monkeypatch.setattr(apv, "assemble_polish_scores_dataset", fake_builder)

    apv.main(
        normalized_dir=str(tmp_path / "normalized"),
        output_dir=str(tmp_path / "datasets" / "validation" / "polish"),
        image_width=1050,
        image_height=1485,
        strict_data_spec=False,
        hf_cache_dir=str(tmp_path / "hf-cache"),
        quiet=True,
    )

    assert len(calls) == 1
    assert calls[0]["include_splits"] == ["train", "val", "test"]
    assert calls[0]["hf_cache_dir"] == str(tmp_path / "hf-cache")
