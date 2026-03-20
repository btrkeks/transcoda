import json

from src.artifacts import (
    DecodingSpec,
    EnvSpec,
    PreprocessingSpec,
    RunArtifact,
    SeedSpec,
    SlurmSpec,
    TokenizerSpec,
    VocabSpec,
    collect_slurm,
)


def test_collect_slurm_returns_none_when_env_absent(monkeypatch):
    keys = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_PARTITION",
        "SLURM_NODELIST",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS_ON_NODE",
        "SLURM_GPU_BIND",
        "SLURM_SUBMIT_HOST",
        "SLURM_CLUSTER_NAME",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)

    assert collect_slurm() is None


def test_collect_slurm_parses_known_env(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_JOB_NAME", "smt-validate")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "1")

    slurm = collect_slurm()
    assert slurm is not None
    assert slurm.job_id == "123"
    assert slurm.job_name == "smt-validate"
    assert slurm.cpus_per_task == 8
    assert slurm.gpus_on_node == "1"


def test_collect_slurm_handles_malformed_int(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-an-int")
    slurm = collect_slurm()

    assert slurm is not None
    assert slurm.job_id == "123"
    assert slurm.cpus_per_task is None


def test_run_artifact_round_trip_with_slurm():
    artifact = RunArtifact(
        experiment_config={"a": 1},
        preprocessing=PreprocessingSpec(image_width=1050, fixed_size=(1485, 1050)),
        decoding=DecodingSpec(strategy="beam", max_len=100, eos_token="</eos>"),
        vocab=VocabSpec(
            w2i_hash="w2i",
            i2w_hash="i2w",
            pad_token=0,
            bos_token="<bos>",
            eos_token="</eos>",
        ),
        tokenizer=TokenizerSpec(vocab_size=3000),
        env=EnvSpec(
            torch="1.0",
            lightning="2.0",
            transformers="4.0",
            cuda=None,
            cudnn=None,
            python="3.12",
            platform="Linux-x86_64",
        ),
        seed=SeedSpec(global_seed=42, deterministic=False),
        slurm=SlurmSpec(job_id="123", cpus_per_task=8),
    )

    loaded = RunArtifact.from_json(artifact.to_json())
    assert loaded.slurm is not None
    assert loaded.slurm.job_id == "123"
    assert loaded.slurm.cpus_per_task == 8
    assert loaded.decoding.repetition_penalty == 1.1


def test_run_artifact_from_legacy_json_without_slurm():
    artifact = RunArtifact(
        experiment_config={"a": 1},
        preprocessing=PreprocessingSpec(image_width=1050, fixed_size=(1485, 1050)),
        decoding=DecodingSpec(strategy="beam", max_len=100, eos_token="</eos>"),
        vocab=VocabSpec(
            w2i_hash="w2i",
            i2w_hash="i2w",
            pad_token=0,
            bos_token="<bos>",
            eos_token="</eos>",
        ),
        tokenizer=TokenizerSpec(vocab_size=3000),
        env=EnvSpec(
            torch="1.0",
            lightning="2.0",
            transformers="4.0",
            cuda=None,
            cudnn=None,
            python="3.12",
            platform="Linux-x86_64",
        ),
        seed=SeedSpec(global_seed=42, deterministic=False),
    )

    payload = json.loads(artifact.to_json())
    payload.pop("slurm", None)
    loaded = RunArtifact.from_json(json.dumps(payload))
    assert loaded.slurm is None


def test_run_artifact_from_legacy_json_without_repetition_penalty():
    artifact = RunArtifact(
        experiment_config={"a": 1},
        preprocessing=PreprocessingSpec(image_width=1050, fixed_size=(1485, 1050)),
        decoding=DecodingSpec(strategy="beam", max_len=100, eos_token="</eos>", repetition_penalty=1.3),
        vocab=VocabSpec(
            w2i_hash="w2i",
            i2w_hash="i2w",
            pad_token=0,
            bos_token="<bos>",
            eos_token="</eos>",
        ),
        tokenizer=TokenizerSpec(vocab_size=3000),
        env=EnvSpec(
            torch="1.0",
            lightning="2.0",
            transformers="4.0",
            cuda=None,
            cudnn=None,
            python="3.12",
            platform="Linux-x86_64",
        ),
        seed=SeedSpec(global_seed=42, deterministic=False),
    )

    payload = json.loads(artifact.to_json())
    payload["decoding"].pop("repetition_penalty", None)
    loaded = RunArtifact.from_json(json.dumps(payload))

    assert loaded.decoding.repetition_penalty == 1.1
