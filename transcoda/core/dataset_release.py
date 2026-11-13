"""The dataset release contract shared by ``pipelines.datasets`` and ``lab``.

Two identities live here:

* the snapshot digest (``hf_dataset_manifest_v1``): the bytes of one persisted
  Hugging Face dataset directory, through its ``state.json``, ``dataset_info.json``
  and a domain-separated sha256 of every listed shard;
* the release digest: what a release claims about those bytes (role, row schema,
  image size, builder pins, corpus version, members, tier counts). Names, the
  on-disk location and timestamps are not part of it, so the same bytes
  registered under two names are one release.

The module is pure: pydantic, hashlib, json and pathlib only. Neither
``pipelines`` nor ``lab`` owns the identity algorithm; both import it from here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

SNAPSHOT_MANIFEST_KIND = "hf_dataset_manifest_v1"
METADATA_FILENAMES = ("dataset_info.json", "state.json")
RELEASE_FILENAME = "release.json"
SNAPSHOT_MANIFEST_FILENAME = "snapshot.manifest.json"
MEMBERS_FILENAME = "members.jsonl"
IMAGE_SIZE = (1485, 1050)
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PREFIXED_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_CHUNK_SIZE = 1024 * 1024
# Fields that name or place a release, or say when it was written. They are
# stored in release.json but never feed the release digest.
_NON_IDENTITY_FIELDS = frozenset({"name", "created_at", "dataset_path"})
ReleaseRole = Literal["train", "eval", "test"]
# Roles a suite scores on: validation (``eval``) and the frozen held-out
# ``test`` split. Their members are ``EvalMember``s.
HELD_OUT_ROLES: frozenset[str] = frozenset({"eval", "test"})


def canonical_json(value: Any) -> bytes:
    """Sorted keys, no insignificant space, UTF-8, finite floats only."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def validate_dataset_filename(value: str) -> str:
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or PurePosixPath(value).is_absolute()
    ):
        raise ValueError(f"dataset filename must be a safe relative POSIX path: {value!r}")
    return value


def _validate_sha256(value: str) -> str:
    if not _HEX_SHA256.fullmatch(value):
        raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
    return value


def _validate_prefixed(value: str) -> str:
    if not _PREFIXED_SHA256.fullmatch(value):
        raise ValueError("digest must be sha256: followed by 64 lowercase hex characters")
    return value


# --- snapshot digest (hf_dataset_manifest_v1) --------------------------------


class DatasetFile(_FrozenModel):
    filename: str
    sha256: str
    size_bytes: int = Field(ge=0)

    @field_validator("filename")
    @classmethod
    def _safe_filename(cls, value: str) -> str:
        return validate_dataset_filename(value)

    @field_validator("sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        return _validate_sha256(value)


class MetadataFile(_FrozenModel):
    filename: str
    sha256: str
    size_bytes: int = Field(ge=0)

    @field_validator("filename")
    @classmethod
    def _known_filename(cls, value: str) -> str:
        if value not in METADATA_FILENAMES:
            raise ValueError(f"metadata filename must be one of {METADATA_FILENAMES}: {value!r}")
        return value

    @field_validator("sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        return _validate_sha256(value)


class DatasetFileStat(_FrozenModel):
    filename: str
    size_bytes: int = Field(ge=0)

    @field_validator("filename")
    @classmethod
    def _safe_filename(cls, value: str) -> str:
        return validate_dataset_filename(value)


def file_v1_sha256(chunks: Iterable[bytes]) -> str:
    """Return the domain-separated shard digest used by historical lab pins."""

    digest = hashlib.sha256()
    digest.update(b"file-v1\0")
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def raw_sha256(chunks: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def legacy_manifest_payload(
    *,
    state: dict[str, Any],
    dataset_info: dict[str, Any],
    data_files: Sequence[DatasetFile],
) -> dict[str, Any]:
    return {
        "kind": SNAPSHOT_MANIFEST_KIND,
        "state": state,
        "dataset_info": dataset_info,
        "data_files": [
            row.model_dump(mode="json") for row in sorted(data_files, key=lambda row: row.filename)
        ],
    }


def legacy_manifest_digest(
    *,
    state: dict[str, Any],
    dataset_info: dict[str, Any],
    data_files: Sequence[DatasetFile],
) -> str:
    return hashlib.sha256(
        canonical_json(
            legacy_manifest_payload(
                state=state,
                dataset_info=dataset_info,
                data_files=data_files,
            )
        )
    ).hexdigest()


class SnapshotManifest(_FrozenModel):
    kind: str
    state: dict[str, Any]
    dataset_info: dict[str, Any]
    data_files: tuple[DatasetFile, ...]
    metadata_files: tuple[MetadataFile, ...]
    manifest_digest: str

    @classmethod
    def create(
        cls,
        *,
        state: dict[str, Any],
        dataset_info: dict[str, Any],
        data_files: Sequence[DatasetFile],
        metadata_files: Sequence[MetadataFile],
    ) -> Self:
        sorted_data = tuple(sorted(data_files, key=lambda row: row.filename))
        sorted_metadata = tuple(sorted(metadata_files, key=lambda row: row.filename))
        return cls(
            kind=SNAPSHOT_MANIFEST_KIND,
            state=state,
            dataset_info=dataset_info,
            data_files=sorted_data,
            metadata_files=sorted_metadata,
            manifest_digest="sha256:"
            + legacy_manifest_digest(
                state=state,
                dataset_info=dataset_info,
                data_files=sorted_data,
            ),
        )

    @field_validator("kind")
    @classmethod
    def _supported_kind(cls, value: str) -> str:
        if value != SNAPSHOT_MANIFEST_KIND:
            raise ValueError(f"snapshot manifest kind must be {SNAPSHOT_MANIFEST_KIND!r}")
        return value

    @field_validator("manifest_digest")
    @classmethod
    def _valid_manifest_digest(cls, value: str) -> str:
        return _validate_prefixed(value)

    @model_validator(mode="after")
    def _consistent_payload(self) -> Self:
        filenames = [row.filename for row in self.data_files]
        if filenames != sorted(filenames):
            raise ValueError("data_files must be sorted by filename")
        if len(filenames) != len(set(filenames)):
            raise ValueError("data_files filenames must be unique")

        metadata_names = [row.filename for row in self.metadata_files]
        if metadata_names != list(METADATA_FILENAMES):
            raise ValueError(f"metadata_files must be exactly {METADATA_FILENAMES}")

        raw_state_files = self.state.get("_data_files", [])
        if not isinstance(raw_state_files, list):
            raise ValueError("state.json _data_files must be a list")
        state_filenames: list[str] = []
        for item in raw_state_files:
            if not isinstance(item, dict) or not isinstance(item.get("filename"), str):
                raise ValueError("state.json _data_files rows must contain filename strings")
            state_filenames.append(validate_dataset_filename(item["filename"]))
        if len(state_filenames) != len(set(state_filenames)):
            raise ValueError("state.json _data_files filenames must be unique")
        if sorted(state_filenames) != filenames:
            raise ValueError("state.json _data_files must match manifest data_files")

        expected = "sha256:" + legacy_manifest_digest(
            state=self.state,
            dataset_info=self.dataset_info,
            data_files=self.data_files,
        )
        if self.manifest_digest != expected:
            raise ValueError(
                f"manifest_digest does not match canonical payload: "
                f"expected {expected}, got {self.manifest_digest}"
            )
        return self

    def to_bytes(self) -> bytes:
        """The tracked sidecar layout: indented, sorted keys, trailing newline."""
        return (json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode()


class SnapshotInspection(_FrozenModel):
    data_files: tuple[DatasetFileStat, ...]
    metadata_files: tuple[MetadataFile, ...]

    @classmethod
    def from_manifest(cls, manifest: SnapshotManifest) -> Self:
        return cls(
            data_files=tuple(
                DatasetFileStat(filename=row.filename, size_bytes=row.size_bytes)
                for row in manifest.data_files
            ),
            metadata_files=manifest.metadata_files,
        )

    @model_validator(mode="after")
    def _canonical_rows(self) -> Self:
        filenames = [row.filename for row in self.data_files]
        if filenames != sorted(filenames):
            raise ValueError("inspection data_files must be sorted by filename")
        if len(filenames) != len(set(filenames)):
            raise ValueError("inspection data_files filenames must be unique")
        metadata_names = [row.filename for row in self.metadata_files]
        if metadata_names != list(METADATA_FILENAMES):
            raise ValueError(f"inspection metadata_files must be exactly {METADATA_FILENAMES}")
        return self


def snapshot_inspection_mismatches(
    manifest: SnapshotManifest, inspection: SnapshotInspection
) -> tuple[str, ...]:
    mismatches: list[str] = []
    if inspection.metadata_files != manifest.metadata_files:
        mismatches.append("state.json/dataset_info.json hash or size")
    expected_files = tuple(
        DatasetFileStat(filename=row.filename, size_bytes=row.size_bytes)
        for row in manifest.data_files
    )
    if inspection.data_files != expected_files:
        mismatches.append("referenced filename set/count or size")
    return tuple(mismatches)


def is_hf_dataset(path: Path) -> bool:
    return path.is_dir() and (path / "state.json").is_file() and (path / "dataset_info.json").is_file()


def file_chunks(path: Path) -> Iterator[bytes]:
    with path.open("rb") as handle:
        yield from iter(lambda: handle.read(_CHUNK_SIZE), b"")


def read_snapshot_metadata(
    root: Path, *, strict: bool = False
) -> tuple[tuple[MetadataFile, ...], dict[str, Any], dict[str, Any]]:
    """Hash the two metadata files and parse them; both must hold JSON objects.

    Pin hashing has always followed a symlinked ``state.json``/``dataset_info.json``;
    ``strict`` (sidecar writing, release adoption) requires regular files.
    """
    rows: list[MetadataFile] = []
    parsed: dict[str, dict[str, Any]] = {}
    for filename in METADATA_FILENAMES:
        path = root / filename
        if (strict and path.is_symlink()) or not path.is_file():
            raise ValueError(f"HF dataset metadata must be a regular file: {path}")
        raw = path.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"HF dataset metadata must contain a JSON object: {path}")
        parsed[filename] = payload
        rows.append(MetadataFile(filename=filename, sha256=raw_sha256([raw]), size_bytes=len(raw)))
    return tuple(rows), parsed["state.json"], parsed["dataset_info.json"]


def listed_data_files(root: Path, state: Mapping[str, Any]) -> tuple[tuple[str, Path], ...]:
    """The shards ``state.json`` lists, as (filename, path); each must be a regular file."""
    raw_data_files = state.get("_data_files", [])
    if not isinstance(raw_data_files, list):
        raise ValueError(f"HF state.json _data_files must be a list: {root / 'state.json'}")
    rows: list[tuple[str, Path]] = []
    for item in raw_data_files:
        if not isinstance(item, dict) or not isinstance(item.get("filename"), str):
            raise ValueError(f"HF state.json contains an invalid data file row: {item!r}")
        filename = validate_dataset_filename(item["filename"])
        file_path = root.joinpath(*filename.split("/"))
        if file_path.is_symlink() or not file_path.is_file():
            raise ValueError(f"dataset manifest references missing regular file: {file_path}")
        rows.append((filename, file_path))
    return tuple(rows)


def build_snapshot_manifest(dataset_dir: str | Path, *, strict_metadata: bool = False) -> SnapshotManifest:
    """Hash every byte of an HF dataset directory into its snapshot manifest."""
    root = Path(dataset_dir)
    if not root.is_dir():
        raise ValueError(f"HF dataset snapshot must be a directory: {root}")
    metadata_files, state, dataset_info = read_snapshot_metadata(root, strict=strict_metadata)
    data_files = [
        DatasetFile(
            filename=filename,
            sha256=file_v1_sha256(file_chunks(file_path)),
            size_bytes=file_path.stat().st_size,
        )
        for filename, file_path in listed_data_files(root, state)
    ]
    return SnapshotManifest.create(
        state=state,
        dataset_info=dataset_info,
        data_files=data_files,
        metadata_files=metadata_files,
    )


def snapshot_digest(dataset_dir: str | Path) -> str:
    """``sha256:<hex>`` of an HF dataset directory's bytes (hf_dataset_manifest_v1)."""
    return build_snapshot_manifest(dataset_dir).manifest_digest


def inspect_snapshot(dataset_dir: Path, *, strict_metadata: bool = False) -> SnapshotInspection:
    """Metadata hashes and shard sizes only: no shard bytes are read."""
    metadata_files, state, _info = read_snapshot_metadata(dataset_dir, strict=strict_metadata)
    stats = sorted(
        (
            DatasetFileStat(filename=filename, size_bytes=path.stat().st_size)
            for filename, path in listed_data_files(dataset_dir, state)
        ),
        key=lambda row: row.filename,
    )
    return SnapshotInspection(data_files=tuple(stats), metadata_files=metadata_files)


def path_digest(path: Path) -> str:
    """Bare hex digest of a lab input: a file, an HF dataset, or a plain directory."""
    if path.is_file():
        return file_v1_sha256(file_chunks(path))
    if path.is_dir():
        if is_hf_dataset(path):
            return build_snapshot_manifest(path).manifest_digest.removeprefix("sha256:")
        digest = hashlib.sha256()
        digest.update(b"dir-v1\0")
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(file_v1_sha256(file_chunks(child)).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()
    raise ValueError(f"configured input path is neither a file nor a directory: {path}")


def row_schema_digest(dataset_info: Mapping[str, Any]) -> str:
    """Digest of an HF dataset's declared features (the row schema)."""
    features = dataset_info.get("features")
    if not isinstance(features, dict):
        raise ValueError("dataset_info.json has no features object")
    return "sha256:" + hashlib.sha256(canonical_json(features)).hexdigest()


# --- release contract ---------------------------------------------------------


class CorpusVersion(_FrozenModel):
    """Mirrors ``CorpusRepository.version()``: trigger-bumped counters plus MAX(change_id)."""

    reviews: int = Field(ge=0)
    pages: int = Field(ge=0)
    splits: int = Field(ge=0)
    last_change: int = Field(ge=0)


class BuilderPins(_FrozenModel):
    """What built the bytes. ``adopted`` registers an existing snapshot in place.

    ``paired_eval_release`` is the release digest of the eval release a real
    train release was built against, and ``paired_test_release`` that of its
    test release: the members and pieces of both were left out, and lab only
    runs the train release under a suite that pins one of them.
    ``tokenizer_sha256`` is the sha256 of the ``tokenizer.json`` every label
    of a real release was checked to spell (``build-real``'s spelling check);
    lab runs the release only with a model that uses that tokenizer.

    All three are omitted from the serialized form when unset, so releases
    built before they existed keep their digests.
    """

    kind: Literal["adopted", "materialized"]
    code_commit: str | None = None
    kern_parser_sha256: str | None = None
    policy_digest: str | None = None
    paired_eval_release: str | None = None
    paired_test_release: str | None = None
    tokenizer_sha256: str | None = None

    @field_validator("paired_eval_release", "paired_test_release", "tokenizer_sha256")
    @classmethod
    def _prefixed_digest(cls, value: str | None) -> str | None:
        return None if value is None else _validate_prefixed(value)

    @model_serializer(mode="wrap")
    def _omit_unset(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        for key in ("paired_eval_release", "paired_test_release", "tokenizer_sha256"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class ReleaseMember(_FrozenModel):
    """One row of a release. Fields beyond sample_id and tier are optional until the
    real-page builder decides what each release must record."""

    sample_id: str = Field(min_length=1)
    tier: str = Field(min_length=1)
    revision_digest: str | None = None
    document_id: str | None = None
    work_group: str | None = None
    is_no_music: bool | None = None

    @field_validator("revision_digest")
    @classmethod
    def _digest(cls, value: str | None) -> str | None:
        return None if value is None else _validate_prefixed(value)


class TrainMember(ReleaseMember):
    tier: Literal["synthetic", "gold", "silver"]


class EvalMember(ReleaseMember):
    """Evaluation rows are human gold at a pinned revision; nothing else can be built."""

    tier: Literal["gold"]
    revision_digest: str


def members_jsonl_bytes(members: Iterable[ReleaseMember]) -> bytes:
    """Canonical members.jsonl: one canonical JSON object per line, sorted by sample id."""
    rows = sorted(members, key=lambda member: member.sample_id)
    ids = [row.sample_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("release members must have unique sample ids")
    return b"".join(canonical_json(row.model_dump(mode="json", exclude_none=True)) + b"\n" for row in rows)


def members_digest(members: Iterable[ReleaseMember]) -> str:
    return "sha256:" + hashlib.sha256(members_jsonl_bytes(members)).hexdigest()


def parse_members_jsonl(data: bytes) -> tuple[ReleaseMember, ...]:
    members = tuple(
        ReleaseMember.model_validate_json(line) for line in data.splitlines() if line.strip()
    )
    if members_jsonl_bytes(members) != data:
        raise ValueError("members.jsonl is not in canonical form (sorted, canonical JSON lines)")
    return members


class ReleaseManifest(_FrozenModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    role: ReleaseRole
    snapshot_digest: str
    row_count: int = Field(ge=0)
    row_schema_digest: str
    image_size: tuple[int, int] = IMAGE_SIZE
    builder: BuilderPins
    corpus_version: CorpusVersion | None = None
    members_digest: str | None = None
    tier_counts: dict[str, int] = Field(default_factory=dict)
    eligibility: Literal["promotable", "diagnostic"] | None = None
    dataset_path: str
    created_at: datetime | None = None

    @field_validator("snapshot_digest", "row_schema_digest")
    @classmethod
    def _digests(cls, value: str) -> str:
        return _validate_prefixed(value)

    @field_validator("members_digest")
    @classmethod
    def _members_digest(cls, value: str | None) -> str | None:
        return None if value is None else _validate_prefixed(value)

    @field_validator("name")
    @classmethod
    def _plain_name(cls, value: str) -> str:
        if value in {".", ".."} or "/" in value or "\\" in value or value != value.strip():
            raise ValueError(f"release name must be a plain directory name: {value!r}")
        return value

    @field_validator("dataset_path")
    @classmethod
    def _data_path(cls, value: str) -> str:
        validate_dataset_filename(value)
        if not value.startswith("data/"):
            raise ValueError(f"dataset_path must be a repository-relative data/ path: {value!r}")
        return value

    @model_validator(mode="after")
    def _counts(self) -> Self:
        if any(count < 0 for count in self.tier_counts.values()):
            raise ValueError("tier counts must be non-negative")
        if self.tier_counts and sum(self.tier_counts.values()) != self.row_count:
            raise ValueError(
                f"tier counts sum to {sum(self.tier_counts.values())}, not row_count {self.row_count}"
            )
        return self


def release_identity(manifest: ReleaseManifest) -> dict[str, Any]:
    payload = manifest.model_dump(mode="json")
    return {key: value for key, value in payload.items() if key not in _NON_IDENTITY_FIELDS}


def release_digest(manifest: ReleaseManifest) -> str:
    """``sha256:<hex>`` over the release's claims; excludes name, location and time."""
    return "sha256:" + hashlib.sha256(b"release-v1\0" + canonical_json(release_identity(manifest))).hexdigest()


def release_file_bytes(manifest: ReleaseManifest) -> bytes:
    payload = manifest.model_dump(mode="json")
    payload["release_digest"] = release_digest(manifest)
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def parse_release_file(data: bytes) -> tuple[ReleaseManifest, str]:
    """Parse release.json and check its recorded digest; returns (manifest, digest)."""
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("release.json must contain a JSON object")
    recorded = payload.pop("release_digest", None)
    manifest = ReleaseManifest.model_validate(payload)
    digest = release_digest(manifest)
    if recorded != digest:
        raise ValueError(f"release.json records digest {recorded}, its content hashes to {digest}")
    return manifest, digest


def check_release_dir(release_dir: Path) -> tuple[ReleaseManifest, str, SnapshotManifest, tuple[ReleaseMember, ...] | None]:
    """Validate a release directory's tracked files against each other (no dataset bytes)."""
    manifest, digest = parse_release_file((release_dir / RELEASE_FILENAME).read_bytes())
    snapshot = SnapshotManifest.model_validate_json((release_dir / SNAPSHOT_MANIFEST_FILENAME).read_bytes())
    if snapshot.manifest_digest != manifest.snapshot_digest:
        raise ValueError(
            f"{SNAPSHOT_MANIFEST_FILENAME} digest {snapshot.manifest_digest} is not the release's "
            f"snapshot digest {manifest.snapshot_digest}"
        )
    if row_schema_digest(snapshot.dataset_info) != manifest.row_schema_digest:
        raise ValueError("release row_schema_digest does not match the snapshot's features")
    members: tuple[ReleaseMember, ...] | None = None
    members_path = release_dir / MEMBERS_FILENAME
    if members_path.exists():
        members = parse_members_jsonl(members_path.read_bytes())
        if members_digest(members) != manifest.members_digest:
            raise ValueError(f"{MEMBERS_FILENAME} does not hash to the release's members_digest")
    elif manifest.members_digest is not None:
        raise ValueError(f"release pins members_digest but {MEMBERS_FILENAME} is missing")
    return manifest, digest, snapshot, members


def diff_releases(
    a: ReleaseManifest,
    b: ReleaseManifest,
    *,
    members_a: Sequence[ReleaseMember] | None = None,
    members_b: Sequence[ReleaseMember] | None = None,
) -> dict[str, Any]:
    """What changes from release ``a`` to ``b``: claims, rows, tiers and members."""
    ident_a, ident_b = release_identity(a), release_identity(b)
    changed = sorted(key for key in ident_a.keys() | ident_b.keys() if ident_a.get(key) != ident_b.get(key))
    tiers = sorted(a.tier_counts.keys() | b.tier_counts.keys())
    result: dict[str, Any] = {
        "a": {"name": a.name, "digest": release_digest(a)},
        "b": {"name": b.name, "digest": release_digest(b)},
        "same_release": release_digest(a) == release_digest(b),
        "changed_fields": changed,
        "rows": {"a": a.row_count, "b": b.row_count, "delta": b.row_count - a.row_count},
        "tier_counts": {
            tier: {
                "a": a.tier_counts.get(tier, 0),
                "b": b.tier_counts.get(tier, 0),
                "delta": b.tier_counts.get(tier, 0) - a.tier_counts.get(tier, 0),
            }
            for tier in tiers
            if a.tier_counts.get(tier, 0) != b.tier_counts.get(tier, 0)
        },
    }
    if members_a is not None and members_b is not None:
        by_a = {row.sample_id: row for row in members_a}
        by_b = {row.sample_id: row for row in members_b}
        groups_a = {row.work_group for row in members_a if row.work_group}
        groups_b = {row.work_group for row in members_b if row.work_group}
        result["members"] = {
            "added": len(by_b.keys() - by_a.keys()),
            "removed": len(by_a.keys() - by_b.keys()),
            "revision_changed": sum(
                1
                for sample_id in by_a.keys() & by_b.keys()
                if by_a[sample_id].revision_digest != by_b[sample_id].revision_digest
            ),
            "tier_changed": sum(
                1 for sample_id in by_a.keys() & by_b.keys() if by_a[sample_id].tier != by_b[sample_id].tier
            ),
            "work_groups_added": sorted(groups_b - groups_a),
            "work_groups_removed": sorted(groups_a - groups_b),
        }
    return result
