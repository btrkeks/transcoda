import json

from scripts.dataset_generation import filter_by_seq_len as filter_module


class _FakeEncoding:
    def __init__(self, length: int):
        self.ids = [0] * length


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True):
        return _FakeEncoding(len(text))


class _FakeDataset:
    def __init__(self, transcriptions: list[str]):
        self._transcriptions = transcriptions
        self.saved_path = None

    def __len__(self):
        return len(self._transcriptions)

    def __getitem__(self, key):
        if key != "transcription":
            raise KeyError(key)
        return self._transcriptions

    def select(self, indices: list[int]):
        return _FakeDataset([self._transcriptions[i] for i in indices])

    def save_to_disk(self, path: str):
        self.saved_path = path


def test_filter_by_seq_len_writes_stats_json(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dataset"
    vocab_path = tmp_path / "vocab"
    output_path = tmp_path / "filtered"
    stats_path = tmp_path / "stats" / "seq_len.json"

    dataset_path.mkdir(parents=True, exist_ok=True)
    vocab_path.mkdir(parents=True, exist_ok=True)
    (vocab_path / "tokenizer.json").write_text("{}", encoding="utf-8")

    fake_dataset = _FakeDataset(["aa", "bbbb", "ccc"])

    monkeypatch.setattr(filter_module, "load_from_disk", lambda _path: fake_dataset)
    monkeypatch.setattr(filter_module.Tokenizer, "from_file", lambda _path: _FakeTokenizer())

    filter_module.filter_by_seq_len(
        dataset_path=str(dataset_path),
        vocab_path=str(vocab_path),
        max_seq_len=3,
        output_path=str(output_path),
        stats_json=str(stats_path),
    )

    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    assert payload["total"] == 3
    assert payload["kept"] == 2
    assert payload["removed"] == 1
    assert payload["max_seq_len"] == 3
    assert payload["before"]["max_length"] == 4
    assert payload["after"]["max_length"] == 3
