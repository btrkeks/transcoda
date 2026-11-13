"""Committed checkpoint records: a checkpoint without a record does not exist."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

COMMITTED_DIRNAME = "committed"


class CommittedCheckpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(ge=0)
    filename: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)
    target_steps: int | None = None

    @field_validator("filename")
    @classmethod
    def _plain_filename(cls, value: str) -> str:
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("filename must be a plain filename")
        return value

    @property
    def partial(self) -> bool:
        return self.target_steps is not None and self.step < self.target_steps


def committed_dir(dirpath: Path) -> Path:
    return Path(dirpath) / COMMITTED_DIRNAME


def record_path(dirpath: Path, step: int) -> Path:
    return committed_dir(dirpath) / f"{step:08d}.json"


def checkpoint_path(dirpath: Path, record: CommittedCheckpoint) -> Path:
    return Path(dirpath) / record.filename


def read_committed_records(dirpath: Path) -> list[CommittedCheckpoint]:
    """Parseable records sorted by step, oldest first."""
    root = committed_dir(dirpath)
    if not root.is_dir():
        return []
    records: list[CommittedCheckpoint] = []
    for path in sorted(root.glob("*.json")):
        try:
            records.append(CommittedCheckpoint.model_validate_json(path.read_bytes()))
        except (OSError, ValidationError):
            continue
    return sorted(records, key=lambda record: record.step)


def newest_committed_checkpoint(
    dirpath: Path, *, verify_sha: bool = False
) -> CommittedCheckpoint | None:
    for record in reversed(read_committed_records(dirpath)):
        if verify_committed_checkpoint(dirpath, record, verify_sha=verify_sha):
            return record
    return None


def verify_committed_checkpoint(
    dirpath: Path, record: CommittedCheckpoint, *, verify_sha: bool = False
) -> bool:
    path = checkpoint_path(dirpath, record)
    try:
        if not path.is_file() or path.stat().st_size != record.size:
            return False
    except OSError:
        return False
    return not verify_sha or sha256_file(path) == record.sha256


def write_committed_record(dirpath: Path, record: CommittedCheckpoint) -> Path:
    path = record_path(dirpath, record.step)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_durably(path, (record.model_dump_json(indent=2) + "\n").encode("utf-8"))
    return path


def write_bytes_durably(path: Path, payload: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
