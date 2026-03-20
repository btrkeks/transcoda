#!/usr/bin/env python3
"""Download a compact W&B analysis bundle for a run.

Bundle contents:
- Run metadata (JSON, including config and summary metrics)
- Sampled training/validation metrics history (JSONL + CSV)
- Final qualitative table artifact (latest) and linked images

Auth:
- Requires WANDB_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_METRIC_KEYS = [
    "_step",
    "_timestamp",
    "_runtime",
    "epoch",
    "trainer/global_step",
    # Common training/classification keys (kept for backwards compatibility)
    "train_loss_step",
    "train_loss_epoch",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_precision",
    "val_recall",
    "val_f1",
    "lr-AdamW/head",
    # OMR project keys (current and legacy naming variants)
    "val/aggregate/SER",
    "val/aggregate/CER",
    "val/synth/SER",
    "val/synth/CER",
    "val/synth/loss",
    "val/polish/SER",
    "val/polish/CER",
    "val/polish/CER_no_ties_beams",
    "val/polish/loss",
    "val/aggregate/runaway_rate",
    "val/aggregate/runaway_samples",
    "val/aggregate/runaway_repeat_loop_rate",
    "val/aggregate/runaway_max_length_hit_rate",
    "val/aggregate/runaway_no_eos_at_max_length_rate",
    "val_SER",
    "val_SER/synth",
    "val_SER/polish",
    "val_CER/synth",
    "val_CER/polish",
    "val_loss/synth",
    "val_loss/polish",
    "val_runaway_rate/synth",
    "val_runaway_rate/polish",
    "val/runaway_rate",
]

DEFAULT_ENTITY = "dadra102-heinrich-heine-university-d-sseldorf"
DEFAULT_PROJECT = "SMT-FP"


class WandbHTTPClient:
    def __init__(self, api_key: str) -> None:
        self.auth_header = "Basic " + base64.b64encode(f"api:{api_key}".encode()).decode()

    def gql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            "https://api.wandb.ai/graphql",
            data=json.dumps({"query": query, "variables": variables}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": self.auth_header,
            },
        )
        with urllib.request.urlopen(req) as resp:
            payload = json.loads(resp.read().decode())
        if "errors" in payload:
            raise RuntimeError(f"W&B GraphQL error: {payload['errors']}")
        return payload["data"]

    def download_file(self, url: str, dst: Path, retries: int = 3) -> int:
        dst.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"Authorization": self.auth_header})
                with urllib.request.urlopen(req) as resp:
                    blob = resp.read()
                dst.write_bytes(blob)
                return len(blob)
            except (urllib.error.HTTPError, urllib.error.URLError):
                if attempt < retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise
        raise RuntimeError(f"failed to download {url}")


def flatten_dict_rows(value: Any) -> list[dict[str, Any]]:
    """Flatten arbitrary nested list structure into list[dict]."""
    out: list[dict[str, Any]] = []
    if isinstance(value, dict):
        out.append(value)
        return out
    if isinstance(value, list):
        for item in value:
            out.extend(flatten_dict_rows(item))
    return out


def clean_scalar(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    return json.dumps(value, ensure_ascii=False)


def write_history_files(rows: list[dict[str, Any]], out_jsonl: Path, out_csv: Path) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    all_keys: list[str] = sorted({k for row in rows for k in row})
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: clean_scalar(row.get(k)) for k in all_keys})


def parse_metric_keys(raw: str | None) -> list[str]:
    if raw is None or raw.strip() == "":
        return list(DEFAULT_METRIC_KEYS)
    return [k.strip() for k in raw.split(",") if k.strip()]


def _normalize_key_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _artifact_ref_from_node(node: dict[str, Any]) -> str:
    seq = str(node["artifactSequence"]["name"])
    aliases = [str(a["alias"]) for a in node.get("aliases", []) if isinstance(a, dict)]
    if "latest" in aliases:
        return f"{seq}:latest"
    return f"{seq}:v{int(node['versionIndex'])}"


def fetch_run_output_artifacts(
    client: WandbHTTPClient,
    entity: str,
    project: str,
    run_id: str,
) -> list[dict[str, Any]]:
    query = """query RunArtifacts($entity: String!, $project: String!, $runId: String!, $cursor: String) {
  project(name: $project, entityName: $entity) {
    run(name: $runId) {
      outputArtifacts(first: 200, after: $cursor) {
        edges {
          node {
            versionIndex
            aliases { alias }
            artifactSequence { name }
          }
        }
        pageInfo { endCursor hasNextPage }
      }
    }
  }
}"""
    out: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        data = client.gql(
            query,
            {"entity": entity, "project": project, "runId": run_id, "cursor": cursor},
        )
        run = data["project"]["run"]
        if run is None:
            raise RuntimeError(f"run not found while resolving artifacts: {entity}/{project}/{run_id}")
        artifacts_obj = run["outputArtifacts"]
        out.extend(edge["node"] for edge in artifacts_obj["edges"])
        if not artifacts_obj["pageInfo"]["hasNextPage"]:
            break
        cursor = artifacts_obj["pageInfo"]["endCursor"]
    return out


def fetch_run_metadata(
    client: WandbHTTPClient,
    entity: str,
    project: str,
    run_id: str,
) -> dict[str, Any]:
    query = """query RunMeta($entity: String!, $project: String!, $runId: String!) {
  project(name: $project, entityName: $entity) {
    run(name: $runId) {
      id
      name
      displayName
      state
      createdAt
      heartbeatAt
      config
      summaryMetrics
    }
  }
}"""
    data = client.gql(query, {"entity": entity, "project": project, "runId": run_id})
    run = data["project"]["run"]
    if run is None:
        raise RuntimeError(f"run not found: {entity}/{project}/{run_id}")
    return run


def fetch_history_rows(
    client: WandbHTTPClient,
    entity: str,
    project: str,
    run_id: str,
    metric_keys: list[str],
    max_samples: int,
) -> list[dict[str, Any]]:
    query = """query RunHistorySample($entity: String!, $project: String!, $runId: String!, $specs: [JSONString!]!) {
  project(name: $project, entityName: $entity) {
    run(name: $runId) {
      sampledHistory(specs: $specs)
    }
  }
}"""
    # `sampledHistory` returns rows where queried keys co-occur.
    # Querying too many keys at once can yield empty intersections. Query
    # per key and merge rows by `_step` to build a usable history table.
    keys = [k for k in metric_keys if k]
    if "_step" not in keys:
        keys = ["_step"] + keys

    rows_by_step: dict[int, dict[str, Any]] = {}

    for key in keys:
        spec_keys = ["_step"] if key == "_step" else ["_step", key]
        specs = [json.dumps({"keys": spec_keys, "samples": max_samples})]
        data = client.gql(
            query,
            {
                "entity": entity,
                "project": project,
                "runId": run_id,
                "specs": specs,
            },
        )
        raw = data["project"]["run"]["sampledHistory"]
        rows = flatten_dict_rows(raw)
        for row in rows:
            if not isinstance(row, dict) or "_step" not in row:
                continue
            step_val = row.get("_step")
            try:
                step = int(step_val)
            except (TypeError, ValueError):
                continue
            merged = rows_by_step.setdefault(step, {"_step": step})
            merged.update(row)

    out = list(rows_by_step.values())
    out.sort(key=lambda r: (r.get("_step", -1), r.get("_timestamp", -1)))
    return out


def resolve_qualitative_artifact_name(
    client: WandbHTTPClient,
    entity: str,
    project: str,
    run_id: str,
    table_key: str,
) -> str:
    candidate = f"run-{run_id}-{table_key}:latest"
    probe_query = """query ProbeArtifact($entity: String!, $project: String!, $artifactName: String!) {
  project(name: $project, entityName: $entity) {
    artifact(name: $artifactName) { id }
  }
}"""
    try:
        data = client.gql(
            probe_query,
            {"entity": entity, "project": project, "artifactName": candidate},
        )
        if data["project"]["artifact"] is not None:
            return candidate
    except Exception:
        pass

    nodes = fetch_run_output_artifacts(client, entity, project, run_id)
    run_prefix = f"run-{run_id}-"
    target_seq = f"{run_prefix}{table_key}"
    target_norm = _normalize_key_token(table_key)

    # First pass: sequence-name based matching for common W&B naming patterns
    # like `run-<id>-<table_key>` and `run-<id>-<table_key>-<suffix>`.
    matched: list[dict[str, Any]] = []
    for node in nodes:
        seq = str(node["artifactSequence"]["name"])
        if seq == target_seq or seq.startswith(f"{target_seq}-"):
            matched.append(node)
            continue

        if not seq.startswith(run_prefix):
            continue
        tail = seq[len(run_prefix) :]
        tail_norm = _normalize_key_token(tail)
        if target_norm and (tail_norm == target_norm or tail_norm.startswith(target_norm)):
            matched.append(node)

    if matched:
        matched.sort(
            key=lambda n: (
                "latest" in {str(a["alias"]) for a in n.get("aliases", []) if isinstance(a, dict)},
                int(n["versionIndex"]),
            ),
            reverse=True,
        )
        return _artifact_ref_from_node(matched[0])

    # Final pass: inspect run output artifacts and pick the one that actually
    # contains a W&B table JSON file. This handles dynamic sequence suffixes.
    nodes_sorted = sorted(
        nodes,
        key=lambda n: (
            "latest" in {str(a["alias"]) for a in n.get("aliases", []) if isinstance(a, dict)},
            int(n["versionIndex"]),
        ),
        reverse=True,
    )
    for node in nodes_sorted:
        artifact_ref = _artifact_ref_from_node(node)
        try:
            files = fetch_artifact_files(client, entity, project, artifact_ref)
        except Exception:
            continue
        if any(str(f.get("name", "")).endswith(".table.json") for f in files):
            return artifact_ref

    raise RuntimeError(
        f"could not resolve qualitative artifact for table_key={table_key} in run {run_id}"
    )


def fetch_artifact_files(
    client: WandbHTTPClient,
    entity: str,
    project: str,
    artifact_name: str,
) -> list[dict[str, Any]]:
    query = """query ArtifactFiles($entity: String!, $project: String!, $artifactName: String!, $cursor: String) {
  project(name: $project, entityName: $entity) {
    artifact(name: $artifactName) {
      files(first: 200, after: $cursor) {
        edges { node { name url } }
        pageInfo { endCursor hasNextPage }
      }
    }
  }
}"""
    out: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        data = client.gql(
            query,
            {
                "entity": entity,
                "project": project,
                "artifactName": artifact_name,
                "cursor": cursor,
            },
        )
        files_obj = data["project"]["artifact"]["files"]
        out.extend(edge["node"] for edge in files_obj["edges"])
        if not files_obj["pageInfo"]["hasNextPage"]:
            break
        cursor = files_obj["pageInfo"]["endCursor"]
    return out


def download_qualitative_bundle(
    client: WandbHTTPClient,
    entity: str,
    project: str,
    run_id: str,
    table_key: str,
    out_dir: Path,
    download_images: bool,
) -> dict[str, Any]:
    artifact_name = resolve_qualitative_artifact_name(client, entity, project, run_id, table_key)
    files = fetch_artifact_files(client, entity, project, artifact_name)

    table_candidates = [f for f in files if f["name"].endswith(".table.json")]
    if not table_candidates:
        raise RuntimeError(f"no table json found in artifact: {artifact_name}")
    table_file = None
    target_name = f"{table_key}.table.json"
    for candidate in table_candidates:
        if candidate["name"] == target_name:
            table_file = candidate
            break
    if table_file is None:
        table_file = table_candidates[0]

    qual_dir = out_dir / "qualitative"
    table_dst = qual_dir / table_file["name"]
    bytes_table = client.download_file(table_file["url"], table_dst)

    image_files = [f for f in files if f["name"].startswith("media/images/")]
    image_count = 0
    image_bytes = 0
    if download_images:
        for img in image_files:
            img_dst = qual_dir / img["name"]
            image_bytes += client.download_file(img["url"], img_dst)
            image_count += 1

    return {
        "artifact_name": artifact_name,
        "artifact_file_count": len(files),
        "table_file": str((Path("qualitative") / table_file["name"]).as_posix()),
        "table_bytes": bytes_table,
        "image_file_count": image_count,
        "image_bytes": image_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download W&B run analysis bundle.")
    parser.add_argument(
        "--entity",
        default=DEFAULT_ENTITY,
        help=f"W&B entity (team/user). Default: {DEFAULT_ENTITY}.",
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help=f"W&B project name. Default: {DEFAULT_PROJECT}.",
    )
    parser.add_argument("--run-id", required=True, help="W&B run id (e.g. 6q1zlk47).")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: wandb/run_<run-id>_bundle",
    )
    parser.add_argument(
        "--table-key",
        default="val_qualitative_samples",
        help="Qualitative table key suffix (default: val_qualitative_samples).",
    )
    parser.add_argument(
        "--metric-keys",
        default=None,
        help="Comma-separated metric keys. Default includes train/val loss, acc/f1, lr, step/time.",
    )
    parser.add_argument(
        "--max-history-samples",
        type=int,
        default=50000,
        help="Requested sampled history size (default: 50000).",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip downloading qualitative images (download table only).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("ERROR: WANDB_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("wandb") / f"run_{args.run_id}_bundle"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    client = WandbHTTPClient(api_key=api_key)
    metric_keys = parse_metric_keys(args.metric_keys)

    run_meta = fetch_run_metadata(client, args.entity, args.project, args.run_id)
    history_rows = fetch_history_rows(
        client,
        args.entity,
        args.project,
        args.run_id,
        metric_keys,
        max_samples=args.max_history_samples,
    )
    qual_summary = download_qualitative_bundle(
        client,
        args.entity,
        args.project,
        args.run_id,
        args.table_key,
        output_dir,
        download_images=not args.skip_images,
    )

    run_meta_path = output_dir / "run_metadata.json"
    history_jsonl_path = output_dir / "metrics_history.jsonl"
    history_csv_path = output_dir / "metrics_history.csv"
    summary_path = output_dir / "download_summary.json"

    with run_meta_path.open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    write_history_files(history_rows, history_jsonl_path, history_csv_path)

    summary = {
        "entity": args.entity,
        "project": args.project,
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "history_rows": len(history_rows),
        "metric_keys": metric_keys,
        "files": {
            "run_metadata": str(run_meta_path),
            "metrics_history_jsonl": str(history_jsonl_path),
            "metrics_history_csv": str(history_csv_path),
        },
        "qualitative": qual_summary,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
