#!/usr/bin/env python3
"""Profile dataset generation with worker timings, py-spy, and system telemetry."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import fire


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _command_available(command: str) -> bool:
    return shutil.which(command) is not None


def _stop_process(proc: subprocess.Popen[bytes], *, timeout_s: float = 10.0) -> int | None:
    if proc.poll() is not None:
        return proc.returncode
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout_s)
    return proc.returncode


def _start_bg_process(
    cmd: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    env: dict[str, str] | None = None,
) -> tuple[subprocess.Popen[bytes], Any, Any]:
    stdout_handle = stdout_path.open("wb")
    stderr_handle = stderr_path.open("wb")
    proc = subprocess.Popen(cmd, stdout=stdout_handle, stderr=stderr_handle, env=env)
    return proc, stdout_handle, stderr_handle


def _start_ps_sampler(
    *,
    output_path: Path,
    stop_event: threading.Event,
    interval_s: float = 1.0,
) -> threading.Thread:
    def _run() -> None:
        with output_path.open("w", encoding="utf-8") as handle:
            while not stop_event.is_set():
                ts = time.time()
                proc = subprocess.run(
                    ["ps", "-eo", "pid,ppid,pcpu,pmem,rss,state,etime,args"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                lines = proc.stdout.splitlines()
                filtered = [
                    line
                    for line in lines
                    if (
                        "dataset_generation.dataset_generation.main" in line
                        or "py-spy record" in line
                        or (
                            "python" in line
                            and "scripts.dataset_generation" in line
                        )
                    )
                ]
                handle.write(f"\n# timestamp={ts}\n")
                for line in filtered:
                    handle.write(line + "\n")
                handle.flush()
                stop_event.wait(interval_s)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def _maybe_start_vmstat(
    *,
    run_dir: Path,
    interval_s: int,
) -> tuple[subprocess.Popen[bytes] | None, Any | None, Any | None]:
    if not _command_available("vmstat"):
        return None, None, None
    return _start_bg_process(
        ["vmstat", str(interval_s)],
        stdout_path=run_dir / "system_vmstat.log",
        stderr_path=run_dir / "system_vmstat.err.log",
    )


def _maybe_start_top(
    *,
    run_dir: Path,
    interval_s: int,
) -> tuple[subprocess.Popen[bytes] | None, Any | None, Any | None]:
    if not _command_available("top"):
        return None, None, None
    return _start_bg_process(
        ["top", "-b", "-d", str(interval_s), "-w", "512"],
        stdout_path=run_dir / "system_top.log",
        stderr_path=run_dir / "system_top.err.log",
    )


def _auto_kern_dirs() -> list[Path]:
    return sorted(Path("data/interim/train").glob("*/3_normalized"))


def _pyspy_extension(output_format: str) -> str:
    if output_format == "flamegraph":
        return "svg"
    if output_format == "raw":
        return "raw"
    if output_format == "speedscope":
        return "json"
    if output_format == "chrometrace":
        return "json"
    raise ValueError(
        "pyspy_format must be one of: flamegraph, raw, speedscope, chrometrace"
    )


def main(
    *kern_dirs: str,
    output_root: str = "/tmp/dataset_generation_profiles",
    dataset_output_dir: str | None = None,
    num_samples: int = 500,
    num_workers: int = 4,
    variants_per_file: int = 1,
    augment_seed: int | None = None,
    profile_log_every: int = 100,
    profile_capture_per_sample: bool = False,
    failure_policy: str = "throughput",
    quarantine_in: str | None = None,
    quarantine_out: str | None = None,
    start_method: str = "auto",
    vmstat_interval: int = 1,
    pyspy_enabled: bool = True,
    pyspy_format: str = "flamegraph",
    pyspy_rate: int = 100,
    pyspy_native: bool = True,
    quiet: bool = False,
) -> None:
    """Run one reproducible profiling benchmark and emit a summary JSON."""
    started_at = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"profile_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if kern_dirs:
        kern_dir_paths = [Path(p) for p in kern_dirs]
    else:
        kern_dir_paths = _auto_kern_dirs()
        if not kern_dir_paths:
            raise ValueError("No default normalized directories found in data/interim/train/*/3_normalized")

    dataset_out = (
        Path(dataset_output_dir)
        if dataset_output_dir is not None
        else run_dir / "dataset_output"
    )
    profile_out = run_dir / "profile_artifacts"
    profile_out.mkdir(parents=True, exist_ok=True)
    dataset_out.parent.mkdir(parents=True, exist_ok=True)

    generation_cmd = [
        sys.executable,
        "-m",
        "scripts.dataset_generation.dataset_generation.main",
        *[str(path) for path in kern_dir_paths],
        "--output_dir",
        str(dataset_out),
        "--num_samples",
        str(num_samples),
        "--num_workers",
        str(num_workers),
        "--variants_per_file",
        str(variants_per_file),
        "--profile_enabled=True",
        "--profile_out_dir",
        str(profile_out),
        "--profile_log_every",
        str(profile_log_every),
        "--profile_sample_limit",
        str(num_samples),
        "--profile_capture_per_sample",
        str(profile_capture_per_sample),
        "--failure_policy",
        str(failure_policy),
        "--start_method",
        str(start_method),
        "--quiet",
        str(quiet),
    ]
    if quarantine_in is not None:
        generation_cmd.extend(["--quarantine_in", str(quarantine_in)])
    if quarantine_out is not None:
        generation_cmd.extend(["--quarantine_out", str(quarantine_out)])
    if augment_seed is not None:
        generation_cmd.extend(["--augment_seed", str(augment_seed)])

    datasets_cache_dir = run_dir / "hf_datasets_cache"
    datasets_cache_dir.mkdir(parents=True, exist_ok=True)
    generation_env = dict(os.environ)
    generation_env["HF_DATASETS_CACHE"] = str(datasets_cache_dir)

    pyspy_available = _command_available("py-spy")
    pyspy_active = bool(pyspy_enabled and pyspy_available)
    pyspy_output_path: Path | None = None
    run_cmd = list(generation_cmd)
    if pyspy_active:
        output_extension = _pyspy_extension(pyspy_format)
        pyspy_output_path = run_dir / f"pyspy.{output_extension}"
        run_cmd = [
            "py-spy",
            "record",
            "--subprocesses",
            "--rate",
            str(pyspy_rate),
        ]
        if pyspy_native:
            run_cmd.append("--native")
        run_cmd.extend(
            [
                "--format",
                pyspy_format,
                "--output",
                str(pyspy_output_path),
                "--",
                *generation_cmd,
            ]
        )

    generation_proc, generation_stdout, generation_stderr = _start_bg_process(
        run_cmd,
        stdout_path=run_dir / "generation.stdout.log",
        stderr_path=run_dir / "generation.stderr.log",
        env=generation_env,
    )

    stop_event = threading.Event()
    ps_thread = _start_ps_sampler(output_path=run_dir / "system_ps.log", stop_event=stop_event)
    vmstat_proc, vmstat_stdout, vmstat_stderr = _maybe_start_vmstat(
        run_dir=run_dir,
        interval_s=vmstat_interval,
    )
    top_proc, top_stdout, top_stderr = _maybe_start_top(
        run_dir=run_dir,
        interval_s=vmstat_interval,
    )

    generation_returncode = generation_proc.wait()
    finished_at = time.time()

    stop_event.set()
    ps_thread.join(timeout=5.0)

    profiler_returncodes = {"pyspy": generation_returncode if pyspy_active else None}
    telemetry_returncodes = {
        "vmstat": _stop_process(vmstat_proc) if vmstat_proc is not None else None,
        "top": _stop_process(top_proc) if top_proc is not None else None,
    }

    for handle in (
        generation_stdout,
        generation_stderr,
        vmstat_stdout,
        vmstat_stderr,
        top_stdout,
        top_stderr,
    ):
        if handle is not None:
            handle.close()

    benchmark_summary = profile_out / "benchmark_summary.json"
    stage_stats = profile_out / "stage_stats.json"
    orchestrator_stats = profile_out / "orchestrator_stats.json"
    run_manifest = profile_out / "run_manifest.json"

    telemetry_notes: dict[str, str] = {}
    if telemetry_returncodes["vmstat"] == -2:
        telemetry_notes["vmstat"] = "terminated by harness signal; expected on shutdown"

    report = {
        "run_dir": str(run_dir),
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": finished_at - started_at,
        "generation_returncode": generation_returncode,
        "profiler_returncodes": profiler_returncodes,
        "telemetry_returncodes": telemetry_returncodes,
        "telemetry_notes": telemetry_notes,
        "command": generation_cmd,
        "execution_command": run_cmd,
        "pyspy": {
            "enabled": pyspy_enabled,
            "available": pyspy_available,
            "active": pyspy_active,
            "format": pyspy_format if pyspy_active else None,
            "output_path": str(pyspy_output_path) if pyspy_output_path is not None else None,
        },
        "kern_dirs": [str(path) for path in kern_dir_paths],
        "num_samples": num_samples,
        "num_workers": num_workers,
        "variants_per_file": variants_per_file,
        "artifacts": {
            "generation_stdout": str(run_dir / "generation.stdout.log"),
            "generation_stderr": str(run_dir / "generation.stderr.log"),
            "system_vmstat": str(run_dir / "system_vmstat.log"),
            "system_top": str(run_dir / "system_top.log"),
            "system_ps": str(run_dir / "system_ps.log"),
            "pyspy_output": str(pyspy_output_path) if pyspy_output_path is not None else None,
            "stage_stats": str(stage_stats),
            "orchestrator_stats": str(orchestrator_stats),
            "run_manifest": str(run_manifest),
            "benchmark_summary": str(benchmark_summary),
        },
        "available_artifacts": {
            "pyspy_output": pyspy_output_path.exists() if pyspy_output_path is not None else False,
            "stage_stats": stage_stats.exists(),
            "orchestrator_stats": orchestrator_stats.exists(),
            "run_manifest": run_manifest.exists(),
            "benchmark_summary": benchmark_summary.exists(),
        },
    }
    _write_json(run_dir / "summary.json", report)
    print(json.dumps(report, indent=2))

    if generation_returncode != 0:
        raise SystemExit(generation_returncode)


if __name__ == "__main__":
    fire.Fire(main)
