"""Prepare meterless continuation variants before render search.

Natural page-split continuations replace their standalone version when selected.
Groups without a natural continuation keep their opening page and gain a paired
synthetic continuation. Selection uses a versioned SHA-256 sample identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_FRACTION = 0.25
DEFAULT_MAX_FAILURE_FRACTION = 0.005


class ContinuationFailures(RuntimeError):
    def __init__(self, summary: dict):
        self.summary = summary
        super().__init__(
            f"{summary['failure_count']}/{summary['input_rows']} continuation rows failed; "
            f"maximum fraction {summary['max_failure_fraction']}"
        )


def failure_class(message: str, returncode: int) -> str:
    if returncode == 101 or "panicked at" in message:
        return "panic"
    if "InvalidSpineToken" in message or "ParseError" in message:
        return "parse"
    if "UnmatchedClose" in message:
        return "unmatched_close"
    if "SpanMismatch" in message:
        return "span_mismatch"
    if "Conflicting" in message:
        return "conflicting_state"
    return "prepare_error"


def selected(sample_id: str, fraction: float = DEFAULT_FRACTION) -> bool:
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("continuation fraction must be finite and in [0, 1]")
    digest = hashlib.sha256(f"continuation-v1:{sample_id}".encode()).digest()
    return int.from_bytes(digest, "big") < int(fraction * (1 << 256))


def prepare(
    *,
    input_dir: Path,
    source_dir: Path,
    output_dir: Path | None = None,
    binary: Path,
    fraction: float = DEFAULT_FRACTION,
    namespace: str = "",
    workers: int = 8,
    manifest: Path | None = None,
    max_failure_fraction: float = DEFAULT_MAX_FAILURE_FRACTION,
    preflight: bool = False,
) -> dict:
    selected("validate", fraction)
    selected("validate", max_failure_fraction)
    if preflight:
        fraction = 1.0
    elif output_dir is None:
        raise ValueError("output_dir is required except in preflight mode")
    if not preflight and input_dir.resolve() == output_dir.resolve():
        raise ValueError("continuations require a separate output directory")
    manifest = manifest or input_dir / "concatenation_manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    expected = set()
    for row in records:
        name = row["output_path"]
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"output_path must be relative: {name}")
        if name in expected:
            raise ValueError(f"duplicate concatenation output: {name}")
        expected.add(name)
    allowed = expected | {
        str(Path(name).with_stem(Path(name).stem + "_continuation")) for name in expected
    }
    if len(allowed) != 2 * len(expected):
        raise ValueError("input names collide with generated continuation variants")
    # Refuse stale extra samples on a rerun with a different input cohort.
    if (
        not preflight
        and output_dir.exists()
        and any(
            p.relative_to(output_dir).as_posix() not in allowed for p in output_dir.rglob("*.krn")
        )
    ):
        raise ValueError("output directory contains unrelated samples")
    if not preflight:
        output_dir.mkdir(parents=True, exist_ok=True)

    def one(row):
        name = row["output_path"]
        first = row["sources"][0]
        natural = int(first.get("start_bar_index", 0)) > 0
        pick = selected(f"{namespace}:{name}", fraction)
        source = input_dir / name
        target = output_dir / name if not preflight else None
        derived = target.with_stem(target.stem + "_continuation") if target else None
        if not preflight:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            # Only remove this stage's stale paired variant when selection changed.
            if derived.exists():
                derived.unlink()
        result = {
            **row,
            "continuation": {"kind": "natural" if natural else "opening", "selected": pick},
        }
        if not pick:
            return [result]
        destination = target if natural else derived
        request = {"action": "page", "input": str(source), "preflight": preflight}
        if not preflight:
            request["output"] = str(destination)
        if natural:
            request.update(
                source=str(source_dir / first["source_path"]),
                start_bar_index=first["start_bar_index"],
                end_bar_index=first.get("end_bar_index"),
            )
        try:
            process = subprocess.run(
                [str(binary)], input=json.dumps(request), text=True, capture_output=True
            )
            if process.returncode:
                error = {
                    "class": failure_class(process.stderr, process.returncode),
                    "returncode": process.returncode,
                    "message": process.stderr.strip()[:1200],
                }
            else:
                receipt = json.loads(process.stdout)
                if receipt.get("status") != "success":
                    raise ValueError("continuation receipt is not successful")
                if not isinstance(receipt["carried_meter_hidden"], bool):
                    raise ValueError("invalid carried_meter_hidden receipt")
                changed = (
                    receipt["changed"]
                    if preflight
                    else destination.read_bytes() != source.read_bytes()
                )
                error = None
        except (OSError, ValueError, KeyError) as exception:
            error = {"class": "receipt_or_io", "message": str(exception)[:1200]}
        if error:
            if not preflight:
                shutil.copyfile(source, target)
                derived.unlink(missing_ok=True)
            result["continuation"].update(status="failed", failure=error)
            return [result]
        if not natural and not receipt["carried_meter_hidden"]:
            if not preflight:
                derived.unlink()
            result["continuation"]["reason"] = "no_opening_meter"
            return [result]
        if preflight:
            result["continuation"].update(**receipt)
            return [result]
        augmented = {
            **row,
            "output_path": destination.relative_to(output_dir).as_posix(),
            "parent_output_path": name,
            "continuation": {
                "kind": "natural" if natural else "synthetic_hidden_header",
                "selected": True,
                "changed": changed,
                **receipt,
            },
        }
        if changed:
            augmented["schema_version"] = "continuation.v1"
            augmented["parent_content_hash"] = row.get("content_hash")
            augmented["parent_provenance_hash"] = row.get("provenance_hash")
            augmented["provenance_hash"] = hashlib.sha256(
                f"continuation-v1:{namespace}:{name}:{row.get('provenance_hash', '')}".encode()
            ).hexdigest()
            augmented["content_hash"] = hashlib.sha256(destination.read_bytes()).hexdigest()
            augmented["parent_output_id"] = row["output_id"]
            augmented["output_id"] = hashlib.sha256(
                f"continuation-v1:{namespace}:{name}".encode()
            ).hexdigest()[:12]
        return [augmented] if natural else [result, augmented]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [row for group in pool.map(one, records) for row in group]
    failures = [
        {"output_path": row["output_path"], **row["continuation"]["failure"]}
        for row in rows
        if "failure" in row["continuation"]
    ]
    summary = {
        "schema_version": "continuations.v1",
        "fraction": fraction,
        "preflight": preflight,
        "failure_count": len(failures),
        "failure_classes": dict(Counter(row["class"] for row in failures)),
        "max_failure_fraction": max_failure_fraction,
        "selection": "SHA256(continuation-v1:NAMESPACE:OUTPUT_PATH) / 2**256 < fraction",
        "input_rows": len(records),
        "output_rows": len(rows),
        "natural_rows": sum(row["continuation"]["kind"] == "natural" for row in rows),
        "natural_selected": sum(
            row["continuation"]["kind"] == "natural" and row["continuation"]["selected"]
            for row in rows
        ),
        "natural_hidden": sum(
            row["continuation"]["kind"] == "natural"
            and row["continuation"].get("carried_meter_hidden", False)
            for row in rows
        ),
        "synthetic_variants": sum(
            row["continuation"]["kind"] == "synthetic_hidden_header" for row in rows
        ),
    }
    summary["threshold_exceeded"] = len(failures) > max_failure_fraction * len(records)
    if preflight:
        summary["failures"] = failures
        if summary["threshold_exceeded"]:
            raise ContinuationFailures(summary)
        return summary
    (output_dir / "concatenation_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    (output_dir / "continuations.json").write_text(json.dumps(summary, indent=2) + "\n")
    subprocess.run(
        [str(binary)],
        input=json.dumps(
            {
                "action": "manifest",
                "output_dir": str(output_dir),
                "input_count": len(records),
                "output_count": len(rows),
                "failure_count": len(failures),
            }
        ),
        text=True,
        capture_output=True,
        check=True,
    )
    if summary["threshold_exceeded"]:
        raise ContinuationFailures(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--binary", default=Path("target/release/continuation_prepare"), type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--fraction", default=DEFAULT_FRACTION, type=float)
    parser.add_argument("--namespace", default="")
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--preflight", action="store_true", help="check all rows without writes")
    parser.add_argument("--max-failure-fraction", default=DEFAULT_MAX_FAILURE_FRACTION, type=float)
    try:
        summary = prepare(**vars(parser.parse_args()))
    except ContinuationFailures as error:
        print(json.dumps(error.summary, indent=2))
        raise SystemExit(1) from None
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
