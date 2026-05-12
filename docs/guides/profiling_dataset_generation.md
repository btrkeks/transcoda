# Profiling the dataset-generation pipeline

A practical, step-by-step guide for finding throughput bottlenecks in:

```bash
source .venv/bin/activate && python -m scripts.dataset_generation.dataset_generation.main \
  data/interim/train/pdmx/3_normalized \
  data/interim/train/grandstaff/3_normalized \
  data/interim/train/musetrainer/3_normalized \
  data/interim/train/openscore-lieder/3_normalized \
  data/interim/train/openscore-stringquartets/3_normalized \
  --name test_v2 \
  --target_samples 100 \
  --num_workers 4 \
  --max_attempts 999
```

## Start Here — Use the official profiling harness

For routine profiling runs, prefer the repo's harness instead of assembling `py-spy` + telemetry by hand:

```bash
source .venv/bin/activate
python scripts/dataset_generation/profile_dataset_generation.py \
  data/interim/train/pdmx/3_normalized \
  data/interim/train/grandstaff/3_normalized \
  data/interim/train/musetrainer/3_normalized \
  data/interim/train/openscore-lieder/3_normalized \
  data/interim/train/openscore-stringquartets/3_normalized \
  --target_samples 500 \
  --num_workers 4
```

The harness already:

- runs the production dataset-generation entrypoint
- records a `py-spy` profile when `py-spy` is available on `PATH`
- captures `ps`, `vmstat`, and `top` telemetry
- isolates `HF_DATASETS_CACHE` under the profiling run directory
- writes a structured `summary.json` with the generated dataset run-artifact location

Use the rest of this guide when you need to interpret those artifacts, compare scaling runs, or drop down to manual `py-spy` / `cProfile` / `line_profiler` work.

## What the workload actually looks like

Before profiling, internalize the shape of the work, otherwise you will misread the results:

- The parent process (`scripts.dataset_generation.dataset_generation.executor.run_dataset_generation`) plans samples, schedules them onto a `pebble.ProcessPool`, collects results, writes shards, and updates progress.
- Each worker (`worker.process_sample_plan` / `evaluate_sample_plan`) does the heavy lifting: kern manipulation, Verovio rendering (native C++), PIL/numpy image post-processing, augmentation.
- Workers are recycled every `_MAX_TASKS_PER_WORKER = 200` tasks (see `executor.py`).
- Throughput is therefore dominated by what happens *inside* workers, with tail latency from Verovio crashes/timeouts and serial parts of the parent (shard writing, resume bookkeeping, progress flushing) capping scaling at high `--num_workers`.

A profiling strategy that only watches the parent process will tell you almost nothing useful. Always sample subprocesses or instrument the worker entrypoint.

## Prerequisites — install the profilers

Run these once before you start. On this machine `py-spy` is already installed and available on `PATH`; the rest go into the project venv.

```bash
source .venv/bin/activate
pip install snakeviz line_profiler viztracer scalene
```

What you get:
- **snakeviz** — browser viewer for `cProfile` output (Step 3).
- **line_profiler** — line-level attribution after you've narrowed to one function (Step 4).
- **viztracer** — low-overhead tracer that produces a per-sample timeline you can open in `vizviewer` / Perfetto. Excellent for seeing *why* a particular sample was slow, not just the average.
- **scalene** — combined CPU + GPU + memory profiler with column-wise attribution per line. Useful when you suspect memory churn (PIL/numpy buffers) or want a one-shot look at CPU vs system vs Python time without a separate `iostat` run.

You don't need all of them; the workflow below is built around `py-spy` + `cProfile` + `line_profiler`. The other two are mentioned where they shine.

## Step 0 — Read built-in artifacts before patching code

Before editing the pipeline or adding fresh timers, inspect the structured artifacts the pipeline already emits.

If you used `scripts/dataset_generation/profile_dataset_generation.py`, start with the run's `summary.json`. It points at:

- the harness logs (`generation.stdout.log`, `generation.stderr.log`, `system_ps.log`, `system_vmstat.log`, `system_top.log`)
- the `py-spy` output, if one was captured
- the dataset-generation run summary
- the dataset-generation `run_artifacts_dir`

Inside the dataset-generation `run_artifacts_dir`, check these first:

- `progress.json` — coarse progress, acceptance/rejection counts, and histograms
- `success_events.jsonl` / `failure_events.jsonl` — per-sample success and failure traces
- `verovio_events.jsonl` — native renderer diagnostics
- `augmentation_events.jsonl` — structured augmentation traces, including built-in stage timings

If you ran `scripts.dataset_generation.dataset_generation.main` directly instead of the harness, the same dataset-generation run artifacts still exist under the dataset's `_runs/<dataset>/<run_id>/` directory.

These files often answer "which stage is weird?" before you need invasive instrumentation.

## Step 1 — Establish a baseline

Goal: a single number you can compare every later experiment against.

1. Use a short, repeatable run. `--target_samples 100 --max_attempts 999` is fine; if a run is too short you'll be measuring warmup (worker spawn, Verovio init) instead of steady-state. If you can spare the time, prefer `--target_samples 500`.
2. Pin to the same input dirs each time.
3. Wrap the run with `/usr/bin/time -v` to capture wall time, max RSS, voluntary/involuntary context switches, and CPU%:

   ```bash
   /usr/bin/time -v python -m scripts.dataset_generation.dataset_generation.main \
     data/interim/train/pdmx/3_normalized ... \
     --name profile_baseline_w4 --target_samples 500 --num_workers 4 \
     --max_attempts 9999 2>&1 | tee runs/profile/baseline_w4.log
   ```
4. Record: wall time, accepted_samples, samples/sec (`accepted / wall`), CPU% from `time -v` (a 4-worker run that's CPU-bound should be near 400%).
5. Repeat with `--num_workers 1`, `2`, `8` (or up to your physical core count). The ratio `samples_per_sec(N) / samples_per_sec(1)` is your **scaling efficiency**; if it plateaus well below `N`, the bottleneck is in the parent or in shared resources (disk, kernel locks, GIL in the parent), not in the worker hot path. This single experiment usually decides whether to focus on per-worker hotspots or on the orchestrator.

Save these numbers under `docs/guides/profile_results/` (or wherever you like) so you can A/B optimizations.

## Step 2 — Sampling profile of the whole process tree (py-spy)

`py-spy` samples without modifying the code, follows subprocesses, and can attach to native frames — exactly what this workload needs.

### 2a. Live view (sanity check)

```bash
py-spy top --subprocesses --pid $(pgrep -f dataset_generation.main)
```

Run the pipeline in another terminal first, then attach. Watch which functions dominate. If you see `verovio`/`_vrv`/`PIL`/`numpy` near the top, that's expected. If you see `multiprocessing.connection.recv` or `pebble` IPC in the parent eating most of the time, the parent is starved and the workers are idle.

### 2b. Flamegraph (the main artifact)

Run for the whole job and write an SVG flamegraph plus a speedscope trace:

```bash
mkdir -p runs/profile
py-spy record \
  --subprocesses \
  --native \
  --rate 200 \
  --output runs/profile/flame_w4.svg \
  --format flamegraph \
  -- python -m scripts.dataset_generation.dataset_generation.main \
       data/interim/train/pdmx/3_normalized ... \
       --name profile_pyspy_w4 --target_samples 500 --num_workers 4 \
       --max_attempts 9999
```

Notes:
- `--native` is important — Verovio is C++, and without `--native` you'll just see `process_sample_plan` as a black box.
- `--subprocesses` is mandatory; without it you only profile the parent.
- Depending on the host's ptrace settings, you may need elevated permission to attach. Common options on Linux are: run the command with `sudo`, grant `py-spy` `cap_sys_ptrace` with `setcap cap_sys_ptrace+ep "$(readlink -f "$(which py-spy)")"`, or temporarily relax `kernel.yama.ptrace_scope`. Pick the least invasive option that fits your environment.
- For an interactive trace open in https://www.speedscope.app, also emit a speedscope file:

  ```bash
  py-spy record --subprocesses --native --rate 200 \
    --format speedscope --output runs/profile/trace_w4.speedscope.json \
    -- python -m scripts.dataset_generation.dataset_generation.main ...
  ```

### How to read the flamegraph

- Wide stacks under `worker.process_sample_plan` → per-sample CPU. Look at:
  - `image_generation/rendering/verovio_backend.py` — Verovio invocation and SVG-to-PNG.
  - `image_generation/image_post.py` — PIL/numpy post-processing.
  - `augmentation.py` and the `image_augmentation/` subpackage.
  - kern normalization inside `image_generation/kern_ops.py` and `score_generator.py`.
- Wide stacks under the parent's main thread (no `worker.*` prefix) → orchestration cost. Likely culprits: shard writes (`io.py`, `datasets` library), resume store I/O (`resume_store.py`), `progress_tracking.maybe_flush_and_report`, JSONL appends.
- Wide stacks in `multiprocessing`/`pebble`/`pickle` → IPC overhead. Big return payloads (e.g. PNG bytes) get pickled across the process boundary every sample. If this is non-trivial, consider returning paths instead of bytes.

## Step 3 — Deterministic worker-module profile (cProfile)

Sampling tells you where time is spent on average; deterministic profiling gives exact call counts and is better for comparing two implementations of the same function.

The trick with `pebble`/`multiprocessing` is that you cannot just wrap `python -m cProfile` around the entrypoint — that profiles the parent. You need to start a profiler **inside each worker**.

A minimal, surgical patch (don't commit it; this is for a profiling session only):

1. In `scripts/dataset_generation/dataset_generation/worker.py`, instrument the worker module at top level:

   ```python
   import cProfile, os, atexit, pathlib
   _PROFILE_DIR = pathlib.Path(os.environ.get("DSG_PROFILE_DIR", ""))
   _profiler = None
   if _PROFILE_DIR:
       _PROFILE_DIR.mkdir(parents=True, exist_ok=True)
       _profiler = cProfile.Profile()
       _profiler.enable()
       atexit.register(
           lambda: _profiler.dump_stats(str(_PROFILE_DIR / f"worker_{os.getpid()}.prof"))
       )
   ```

   Place this at module top-level so it runs once per worker import. This intentionally includes worker initialization cost such as `init_generation_worker(...)` and `VerovioRenderer()` setup. That startup time matters because workers are recycled every `_MAX_TASKS_PER_WORKER = 200` tasks.

2. Run with the env var set:

   ```bash
   DSG_PROFILE_DIR=runs/profile/cprofile_w4 \
     python -m scripts.dataset_generation.dataset_generation.main ... --num_workers 4 ...
   ```

3. Merge and inspect:

   ```bash
   python - <<'PY'
   import pstats, glob
   stats = pstats.Stats(*glob.glob("runs/profile/cprofile_w4/worker_*.prof"))
   stats.strip_dirs().sort_stats("cumulative").print_stats(40)
   PY
   ```

   Or open in [snakeviz](https://jiffyclub.github.io/snakeviz/) (`pip install snakeviz; snakeviz runs/profile/cprofile_w4/worker_*.prof`).

Caveat: cProfile adds 10–30% overhead, which compresses native time relative to Python time. Use it for ranking *Python-level* hotspots; trust py-spy with `--native` for native time accounting.

## Step 4 — Line-level profiling on the hot function

Once Steps 2–3 narrow it down to one or two functions, drop in `line_profiler` for line-by-line attribution. Decorate the suspect function with `@profile` (line_profiler's globally injected decorator) — typically something inside `score_generator.py`, `kern_ops.py`, `image_post.py`, or an augmentation step — and run:

```bash
kernprof -l -o runs/profile/lineprof/run.lprof \
  -m scripts.dataset_generation.dataset_generation.main ... --num_workers 1 ...
python -m line_profiler runs/profile/lineprof/run.lprof
```

Use `--num_workers 1` for line profiling; otherwise the per-worker `.lprof` files trample each other.

## Step 5 — Distinguish CPU-bound from I/O-bound

If `time -v` shows CPU% well below `100% × num_workers`, the workers are blocked on something. Quick checks:

- **Disk**: `iostat -xm 2` while the run is in progress. High `%util` on the dataset device → shard/PNG writes or input reads are the wall.
- **Per-process I/O**: `pidstat -d -p $(pgrep -f dataset_generation) 2`.
- **Syscalls**: for one worker, `sudo strace -c -p <worker_pid>` for ~30s shows the syscall histogram. Lots of `write`/`fsync`/`openat` → I/O bound. Lots of `futex` → lock contention (rare here, but possible inside Verovio or in a thread pool).
- **Memory pressure**: `time -v` "Maximum resident set size" combined with `vmstat 2` — if the system is swapping or pages are being reclaimed, throughput collapses non-linearly.

## Step 6 — Find the parent-side ceiling

If scaling efficiency from Step 1 is poor (say `samples/sec(8) / samples/sec(1) < 4`), the parent is the bottleneck. To localize it, run py-spy against the parent only, while the workers are busy:

```bash
py-spy record --pid <parent_pid> --rate 500 --duration 60 \
  --format flamegraph --output runs/profile/parent_only.svg
```

Common offenders, in rough order of likelihood:

1. `pebble`/`multiprocessing` IPC: pickling large worker payloads. If `WorkerSuccess` carries raw image bytes, that crosses the process boundary every sample. Returning a path to a written file is usually faster.
2. Shard finalization in `executor.py` (the `datasets` library serialization, parquet/arrow writes).
3. Frequent JSONL appends in `io.append_jsonl` and `outcome_events.write_outcome_events` — flush cadence may be too high.
4. `progress_tracking.maybe_flush_and_report` printing/serializing too often.
5. The `wait(..., return_when=FIRST_COMPLETED, timeout=...)` loop's timeout granularity.

Each of these is fixable without touching workers, and the wins compound with worker speedups.

## Step 7 — Targeted micro-benchmarks

When you have a candidate hotspot, isolate it from the pipeline before you optimize:

- For Verovio rendering, write a small script that loads a representative `.krn`, builds the same recipe options the pipeline uses, and renders N times in a tight loop. Time it with `perf_counter`. This eliminates IPC, scheduling, and the augmentation pipeline as variables and gives you a clean baseline for comparing rendering options (e.g. SVG → PNG backend choices, page size, font caching).
- For image post-processing, do the same with a fixed input PNG.
- For augmentation, inspect `augmentation_events.jsonl` first because it already includes structured timing fields. Only add `time.perf_counter_ns()` instrumentation when those built-in timings are too coarse for the question you're asking.

## Step 8 — Decide and act

After the above, you should be able to fill in this table — that's the deliverable, not the SVGs:

| Stage                     | Cost (ms/sample) | % of worker wall | Notes / cheapest win |
|---------------------------|------------------|------------------|----------------------|
| Verovio render            |                  |                  |                      |
| SVG → PNG                 |                  |                  |                      |
| Image post-processing     |                  |                  |                      |
| Augmentation (per kind)   |                  |                  |                      |
| Kern normalization/ops    |                  |                  |                      |
| IPC + pickle              |                  |                  |                      |
| Parent shard/I/O          |                  |                  |                      |

Optimize the largest row first and re-run Step 1. Stop when wall time is dominated by stages you've decided not to touch.

## Reproducibility checklist

- Same input dirs, same `--target_samples`, same `--max_attempts`, same git commit. Note the commit SHA in the log filename.
- Disable other GPU/CPU-heavy work on the box; especially other Python processes that might steal cores.
- Warm caches: run once and discard; profile the second run. (Verovio font loading, Python bytecode caches, OS page cache for input `.krn`s all help.)
- Resume artifacts can skew throughput. For clean numbers, use a fresh `--name` each run, or pass `--resume_mode off`.

## Reference: tools and where to install them

| Tool            | Install                              | Use it for                                  |
|-----------------|--------------------------------------|---------------------------------------------|
| `py-spy`        | system install on `PATH`             | Sampling, subprocesses, native frames       |
| `cProfile`      | stdlib                               | Deterministic per-worker profile            |
| `snakeviz`      | `pip install snakeviz`               | Visualize cProfile output                   |
| `line_profiler` | `pip install line_profiler`          | Line-level attribution on one hot function  |
| `viztracer`     | `pip install viztracer`              | Per-sample timeline / Perfetto trace        |
| `scalene`       | `pip install scalene`                | Combined CPU + memory + GPU per-line view   |
| `/usr/bin/time` | coreutils                            | Wall time, RSS, CPU%, context switches      |
| `iostat`/`pidstat` | `sysstat` package                 | Disk and per-process I/O                    |
| `strace`        | system                               | Syscall mix when CPU% is unexpectedly low   |
| speedscope      | https://www.speedscope.app (browser) | Interactive flamegraph viewing              |
