#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
LOG_DIR="${ROOT_DIR}/logs"

COMMAND="submit"
if [ $# -gt 0 ]; then
  case "$1" in
    submit|validate|shell|local|queue|cancel|logs|doctor|resources|help|-h|--help)
      COMMAND="$1"
      shift
      ;;
  esac
fi

CONFIG="${ROOT_DIR}/config/train.json"
PARTITION="gpu"
PARTITION_EXPLICIT=false
GPUS=1
GPU_TYPE=""
CPUS_PER_TASK=8
MEMORY=""
TIME_LIMIT="24:00:00"
JOB_NAME=""
SEED=42
CHECKPOINT_PATH=""
FRESH_RUN=false
GPU_ID=0
NO_SYNC=false
DRY_RUN=false
NO_DOCTOR=false
DOCTOR_WARN_ONLY=false
DOCTOR_JSON=false
SLURM_EXTRA_ARGS=()
FORWARDED_ARGS=()

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command '$1' not found in PATH." >&2
    exit 127
  fi
}

has_forwarded_override() {
  local key="$1"
  local arg
  for arg in "${FORWARDED_ARGS[@]}"; do
    if [[ "${arg}" == "--${key}" ]] || [[ "${arg}" == "--${key}="* ]]; then
      return 0
    fi
  done
  return 1
}

json_escape() {
  local s="$1"
  s=${s//\\/\\\\}
  s=${s//\"/\\\"}
  s=${s//$'\n'/\\n}
  printf '%s' "$s"
}

DOCTOR_PASSES=0
DOCTOR_WARNS=0
DOCTOR_FAILS=0
DOCTOR_RESULTS=()

doctor_reset() {
  DOCTOR_PASSES=0
  DOCTOR_WARNS=0
  DOCTOR_FAILS=0
  DOCTOR_RESULTS=()
}

doctor_add() {
  local level="$1"
  local check="$2"
  local message="$3"

  DOCTOR_RESULTS+=("${level}|${check}|${message}")
  case "${level}" in
    PASS)
      DOCTOR_PASSES=$((DOCTOR_PASSES + 1))
      ;;
    WARN)
      DOCTOR_WARNS=$((DOCTOR_WARNS + 1))
      ;;
    FAIL)
      DOCTOR_FAILS=$((DOCTOR_FAILS + 1))
      ;;
  esac
}

doctor_print() {
  local entry
  local level
  local check
  local message
  local idx=0

  if [ "${DOCTOR_JSON}" = true ]; then
    printf '{'
    printf '"summary":{"pass":%d,"warn":%d,"fail":%d},' "${DOCTOR_PASSES}" "${DOCTOR_WARNS}" "${DOCTOR_FAILS}"
    printf '"results":['
    for entry in "${DOCTOR_RESULTS[@]}"; do
      IFS='|' read -r level check message <<<"${entry}"
      if [ ${idx} -gt 0 ]; then
        printf ','
      fi
      printf '{"level":"%s","check":"%s","message":"%s"}' \
        "$(json_escape "${level}")" \
        "$(json_escape "${check}")" \
        "$(json_escape "${message}")"
      idx=$((idx + 1))
    done
    printf ']}\n'
    return
  fi

  echo "Preflight checks:"
  for entry in "${DOCTOR_RESULTS[@]}"; do
    IFS='|' read -r level check message <<<"${entry}"
    printf '[%s] %s: %s\n' "${level}" "${check}" "${message}"
  done
  printf 'Summary: pass=%d warn=%d fail=%d\n' "${DOCTOR_PASSES}" "${DOCTOR_WARNS}" "${DOCTOR_FAILS}"
}

run_doctor() {
  local strict=true
  if [ "${DOCTOR_WARN_ONLY}" = true ]; then
    strict=false
  fi

  doctor_reset

  local cmd
  for cmd in sbatch squeue scontrol uv; do
    if command -v "${cmd}" >/dev/null 2>&1; then
      doctor_add "PASS" "command:${cmd}" "available"
    else
      doctor_add "FAIL" "command:${cmd}" "not found in PATH"
    fi
  done

  if [ -f "${CONFIG}" ]; then
    doctor_add "PASS" "config" "${CONFIG}"
  else
    doctor_add "FAIL" "config" "missing file: ${CONFIG}"
  fi

  if [ -f "${ROOT_DIR}/.venv/bin/activate" ]; then
    doctor_add "PASS" "venv" ".venv activation script present"
  else
    doctor_add "WARN" "venv" ".venv/bin/activate not found"
  fi

  if [ -n "${WANDB_API_KEY:-}" ]; then
    doctor_add "PASS" "wandb" "WANDB_API_KEY is set"
  elif [ -f "${HOME}/.netrc" ] && grep -q "api.wandb.ai" "${HOME}/.netrc"; then
    doctor_add "PASS" "wandb" "W&B credentials found in ~/.netrc"
  else
    doctor_add "WARN" "wandb" "no WANDB_API_KEY and no ~/.netrc entry for api.wandb.ai"
  fi

  if git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    doctor_add "PASS" "git" "repository detected"
  else
    doctor_add "WARN" "git" "not a git working tree"
  fi

  local gres="gpu:${GPUS}"
  if [ -n "${GPU_TYPE}" ]; then
    gres="gpu:${GPU_TYPE}:${GPUS}"
  fi

  local partition_probe
  if command -v sinfo >/dev/null 2>&1; then
    if partition_probe="$(sinfo -N -p "${PARTITION}" -o '%N %G' -h 2>/dev/null)"; then
      if [ -n "${partition_probe}" ]; then
        doctor_add "PASS" "partition" "partition '${PARTITION}' is visible"
        if [ -n "${GPU_TYPE}" ]; then
          if echo "${partition_probe}" | grep -q "gpu:${GPU_TYPE}:"; then
            doctor_add "PASS" "gpu-type" "gpu:${GPU_TYPE} is available in partition '${PARTITION}'"
          else
            doctor_add "FAIL" "gpu-type" "gpu:${GPU_TYPE} not found in partition '${PARTITION}'"
          fi
        fi
      else
        doctor_add "WARN" "partition" "partition '${PARTITION}' returned no nodes"
      fi
    else
      doctor_add "WARN" "partition" "unable to query partition '${PARTITION}' via sinfo"
    fi
  else
    doctor_add "WARN" "partition" "sinfo not found; skipping partition checks"
    if [ -n "${GPU_TYPE}" ]; then
      doctor_add "WARN" "gpu-type" "sinfo unavailable; unable to verify gpu:${GPU_TYPE}"
    fi
  fi

  if [ "${COMMAND}" = "validate" ] || [ -n "${CHECKPOINT_PATH}" ]; then
    if [ -n "${CHECKPOINT_PATH}" ]; then
      if [ -f "${CHECKPOINT_PATH}" ]; then
        doctor_add "PASS" "checkpoint" "${CHECKPOINT_PATH}"
      else
        doctor_add "FAIL" "checkpoint" "missing file: ${CHECKPOINT_PATH}"
      fi
    else
      doctor_add "FAIL" "checkpoint" "validate mode requires --checkpoint_path"
    fi
  fi

  if [ -n "${MEMORY}" ]; then
    local test_output
    local test_status=0
    set +e
    test_output="$(sbatch --test-only \
      "--partition=${PARTITION}" \
      "--gres=${gres}" \
      "--cpus-per-task=${CPUS_PER_TASK}" \
      "--time=${TIME_LIMIT}" \
      "--mem=${MEMORY}" \
      --wrap="hostname" 2>&1)"
    test_status=$?
    set -e

    if [ ${test_status} -eq 0 ]; then
      doctor_add "PASS" "memory" "memory request '${MEMORY}' accepted by --test-only"
    else
      doctor_add "FAIL" "memory" "sbatch --test-only failed: ${test_output}"
    fi
  fi

  doctor_print

  if [ "${strict}" = true ] && [ ${DOCTOR_FAILS} -gt 0 ]; then
    return 1
  fi
  return 0
}

show_resources() {
  require_cmd sinfo

  local sinfo_format="%P|%N|%t|%C|%m|%G"
  local sinfo_cmd=(
    sinfo
    -N
    -h
    -o
    "${sinfo_format}"
  )
  if [ "${PARTITION_EXPLICIT}" = true ]; then
    sinfo_cmd=(
      sinfo
      -N
      -h
      "-p=${PARTITION}"
      -o
      "${sinfo_format}"
    )
  fi

  echo "Cluster resource snapshot:"
  if [ "${PARTITION_EXPLICIT}" = true ]; then
    echo "Partition filter: ${PARTITION}"
  else
    echo "Partition filter: all"
  fi
  "${sinfo_cmd[@]}" | awk -F'|' '
    BEGIN {
      printf "%-18s %-20s %-8s %-20s %-10s %-30s\n", "PARTITION", "NODE", "STATE", "CPUS(A/I/O/T)", "MEM(MB)", "GRES";
    }
    NF >= 6 {
      printf "%-18s %-20s %-8s %-20s %-10s %-30s\n", $1, $2, $3, $4, $5, $6;
    }
  '

  if command -v scontrol >/dev/null 2>&1; then
    echo
    echo "GPU allocation by node:"

    local scontrol_cmd=(scontrol show nodes -o)
    local squeue_gpu_csv=""
    declare -A squeue_gpu_used=()
    if [ "${PARTITION_EXPLICIT}" = true ]; then
      mapfile -t resource_nodes < <(sinfo -N -h "-p=${PARTITION}" -o "%N")
      if [ ${#resource_nodes[@]} -eq 0 ]; then
        echo "No nodes found for partition '${PARTITION}'."
        return 0
      fi
      scontrol_cmd+=( "${resource_nodes[@]}" )
    fi

    if command -v squeue >/dev/null 2>&1; then
      while IFS='|' read -r nodelist job_gres; do
        [ -z "${nodelist}" ] && continue
        [ "${nodelist}" = "(null)" ] && continue
        [ -z "${job_gres}" ] && continue
        [ "${job_gres}" = "N/A" ] && continue

        local gpu_count=0
        local part=""
        IFS=',' read -r -a _gres_parts <<<"${job_gres}"
        for part in "${_gres_parts[@]}"; do
          part="${part%%(*}"
          if [[ "${part}" =~ ^(gres/)?gpu(:|$) ]]; then
            if [[ "${part}" =~ :([0-9]+)$ ]]; then
              gpu_count=$((gpu_count + BASH_REMATCH[1]))
            else
              gpu_count=$((gpu_count + 1))
            fi
          fi
        done
        [ ${gpu_count} -gt 0 ] || continue

        local expanded_nodes=()
        if mapfile -t expanded_nodes < <(scontrol show hostnames "${nodelist}" 2>/dev/null); then
          :
        else
          expanded_nodes=("${nodelist}")
        fi
        [ ${#expanded_nodes[@]} -gt 0 ] || continue

        local per_node=$((gpu_count / ${#expanded_nodes[@]}))
        local remainder=$((gpu_count % ${#expanded_nodes[@]}))
        local idx=0
        local node=""
        for node in "${expanded_nodes[@]}"; do
          local add=${per_node}
          if [ ${idx} -lt ${remainder} ]; then
            add=$((add + 1))
          fi
          squeue_gpu_used["${node}"]=$(( ${squeue_gpu_used["${node}"]:-0} + add ))
          idx=$((idx + 1))
        done
      done < <(squeue -h -t R -o "%N|%b")

      local n=""
      for n in "${!squeue_gpu_used[@]}"; do
        if [ -n "${squeue_gpu_csv}" ]; then
          squeue_gpu_csv="${squeue_gpu_csv},"
        fi
        squeue_gpu_csv="${squeue_gpu_csv}${n}=${squeue_gpu_used[${n}]}"
      done
    fi

    "${scontrol_cmd[@]}" | awk -v squeue_gpu_csv="${squeue_gpu_csv}" '
      function extract_tres_gpu(tres,    n, i, arr, kv, key, val, total, found) {
        if (tres == "") {
          return ""
        }
        n = split(tres, arr, ",")
        total = 0
        found = 0
        for (i = 1; i <= n; i++) {
          split(arr[i], kv, "=")
          key = kv[1]
          val = kv[2]
          if (key ~ /^gres\/gpu($|:)/ && val ~ /^[0-9]+$/) {
            total += (val + 0)
            found = 1
          }
        }
        if (found) {
          return total
        }
        return ""
      }
      function extract_gres_gpu(gres,    n, i, arr, parts, count, total, found) {
        if (gres == "" || gres == "(null)") {
          return ""
        }
        n = split(gres, arr, ",")
        total = 0
        found = 0
        for (i = 1; i <= n; i++) {
          gsub(/\(.*/, "", arr[i])
          if (arr[i] ~ /^gpu(:|$)/) {
            split(arr[i], parts, ":")
            count = parts[length(parts)]
            if (count ~ /^[0-9]+$/) {
              total += (count + 0)
            } else {
              total += 1
            }
            found = 1
          }
        }
        if (found) {
          return total
        }
        return ""
      }
      function parse_squeue_map(raw,    n, i, pairs, kv) {
        if (raw == "") {
          return
        }
        n = split(raw, pairs, ",")
        for (i = 1; i <= n; i++) {
          split(pairs[i], kv, "=")
          if (kv[1] != "" && kv[2] ~ /^[0-9]+$/) {
            squeue_gpu[kv[1]] = kv[2] + 0
          }
        }
      }
      BEGIN {
        parse_squeue_map(squeue_gpu_csv)
        printf "%-20s %-18s %-12s %-10s\n", "NODE", "STATE", "GPUs(U/T)", "SOURCE"
      }
      {
        node = ""; state = ""; cfg = ""; alloc = ""; gres = ""; gres_used = "";
        for (i = 1; i <= NF; i++) {
          if ($i ~ /^NodeName=/) {
            node = substr($i, 10)
          } else if ($i ~ /^State=/) {
            state = substr($i, 7)
            sub(/\+.*/, "", state)
          } else if ($i ~ /^CfgTRES=/) {
            cfg = substr($i, 9)
          } else if ($i ~ /^AllocTRES=/) {
            alloc = substr($i, 11)
          } else if ($i ~ /^Gres=/) {
            gres = substr($i, 6)
          } else if ($i ~ /^GresUsed=/) {
            gres_used = substr($i, 10)
          }
        }
        if (node != "") {
          total_gpu = extract_tres_gpu(cfg)
          if (total_gpu == "") {
            total_gpu = extract_gres_gpu(gres)
          }

          used_gpu = extract_tres_gpu(alloc)
          source = "AllocTRES"
          if (used_gpu == "") {
            used_gpu = extract_gres_gpu(gres_used)
            if (used_gpu != "") {
              source = "GresUsed"
            } else if (node in squeue_gpu) {
              used_gpu = squeue_gpu[node]
              source = "squeue"
            } else {
              used_gpu = "?"
              source = "-"
            }
          }

          if (total_gpu == "") {
            total_gpu = "?"
          }
          printf "%-20s %-18s %-12s %-10s\n", node, state, used_gpu "/" total_gpu, source
        }
      }
    '

    echo
    echo "Configured vs allocated TRES:"
    "${scontrol_cmd[@]}" | awk '
      BEGIN {
        printf "%-20s %-18s %-42s %s\n", "NODE", "STATE", "CfgTRES", "AllocTRES"
      }
      {
        node = ""; state = ""; cfg = ""; alloc = "";
        for (i = 1; i <= NF; i++) {
          if ($i ~ /^NodeName=/) {
            node = substr($i, 10);
          } else if ($i ~ /^State=/) {
            state = substr($i, 7);
            sub(/\+.*/, "", state);
          } else if ($i ~ /^CfgTRES=/) {
            cfg = substr($i, 9);
          } else if ($i ~ /^AllocTRES=/) {
            alloc = substr($i, 11);
          }
        }
        if (node != "") {
          printf "%-20s %-18s %-42s %s\n", node, state, cfg, alloc;
        }
      }
    '
  else
    echo
    echo "scontrol not found; skipping CfgTRES/AllocTRES details."
  fi
}

usage() {
  cat <<'EOF'
Usage:
  ./train.sh [submit] [options] -- [train.py overrides]
  ./train.sh validate --checkpoint_path PATH [options] -- [train.py overrides]
  ./train.sh shell [options]
  ./train.sh local [--gpu-id N] [options] -- [train.py overrides]
  ./train.sh queue
  ./train.sh cancel JOB_ID
  ./train.sh logs [JOB_ID] [--lines N] [--no-follow]
  ./train.sh doctor [options]
  ./train.sh resources [--partition NAME]

Commands:
  submit      Submit a training job with sbatch (default).
  validate    Submit a validate-only job (requires checkpoint, enables example-image logging by default).
  shell       Open an interactive Slurm shell (srun --pty bash).
  local       Run train.py directly on this machine.
  queue       Show your queued/running jobs.
  cancel      Cancel a job by job ID.
  logs        Tail Slurm logs for a job (auto-detects active job by default).
  doctor      Run preflight checks for Slurm training/validation.
  resources   Show Slurm node resources (cluster-wide by default).

Common options:
  --config PATH            Config JSON (default: config/train.json)
  --debug                  Shortcut for config/debug.json
  --profile                Shortcut for config/profile.json
  --partition NAME         Slurm partition (default: gpu)
  --gpus N                 Number of GPUs (default: 1)
  --gpu-type TYPE          GPU type for --gres (e.g. rtx4090, rtx5090)
  --cpus-per-task N        CPUs per task (default: 8)
  --mem SIZE               Memory request (default: unset)
  --time HH:MM:SS          Time limit (default: 24:00:00)
  --job-name NAME          Slurm job name
  --seed N                 Seed passed to train.py (default: 42)
  --checkpoint_path PATH   Checkpoint path
  --fresh-run              Disable auto-resume and force fresh training
  --no-sync                Skip 'uv sync --group omr-ned' in job command
  --no-doctor              Skip auto preflight checks for submit/validate
  --doctor-warn-only       Do not block job submission on doctor failures
  --strict                 Enforce fail-fast doctor behavior
  --warn-only              Doctor mode: return success even if checks fail
  --json                   Doctor mode: print machine-readable JSON
  --sbatch-arg ARG         Additional sbatch arg (repeatable)
  --dry-run                Print command and exit
  -h, --help               Show this help

Local-only options:
  --gpu-id N               CUDA_VISIBLE_DEVICES for local mode (default: 0)

Examples:
  ./train.sh submit --gpu-type rtx5090 --gpus 2 --time 16:00:00 -- --training.max_epochs=40
  ./train.sh submit --fresh-run -- --checkpoint.run_name=fresh-ablation
  ./train.sh validate --checkpoint_path weights/GrandStaff/smt-model.ckpt --gpu-type rtx4090
  ./train.sh shell --gpu-type rtx5090 --gpus 1
  ./train.sh queue
  ./train.sh logs
  ./train.sh doctor --checkpoint_path weights/GrandStaff/smt-model.ckpt --gpu-type rtx4090
  ./train.sh resources
  ./train.sh resources --partition gpu
EOF
}

if [ "${COMMAND}" = "-h" ] || [ "${COMMAND}" = "--help" ] || [ "${COMMAND}" = "help" ]; then
  usage
  exit 0
fi

if [ "${COMMAND}" = "queue" ]; then
  require_cmd squeue
  squeue -u "${USER}"
  exit 0
fi

if [ "${COMMAND}" = "cancel" ]; then
  require_cmd scancel
  JOB_ID="${1:-}"
  if [ -z "${JOB_ID}" ]; then
    echo "Missing job id. Usage: ./train.sh cancel JOB_ID" >&2
    exit 2
  fi
  scancel "${JOB_ID}"
  exit 0
fi

if [ "${COMMAND}" = "logs" ]; then
  LOG_JOB_ID=""
  LOG_LINES=50
  LOG_FOLLOW=true

  if [ $# -gt 0 ] && [[ "${1}" != --* ]]; then
    LOG_JOB_ID="$1"
    shift
  fi

  while [ $# -gt 0 ]; do
    case "$1" in
      --lines)
        LOG_LINES="$2"
        shift 2
        ;;
      --no-follow)
        LOG_FOLLOW=false
        shift
        ;;
      -h|--help)
        echo "Usage: ./train.sh logs [JOB_ID] [--lines N] [--no-follow]"
        echo "If JOB_ID is omitted, auto-detects newest active job (running first, then pending)."
        exit 0
        ;;
      *)
        echo "Unknown logs option: $1" >&2
        echo "Usage: ./train.sh logs [JOB_ID] [--lines N] [--no-follow]" >&2
        exit 2
        ;;
    esac
  done

  require_cmd tail

  if [ -z "${LOG_JOB_ID}" ]; then
    require_cmd squeue
    LOG_JOB_ID="$(
      squeue -u "${USER}" -h -o "%i %t %j" | awk '$2=="R" && $3 ~ /^smt-/ {print $1}' | sort -nr | head -n1
    )"
  fi
  if [ -z "${LOG_JOB_ID}" ]; then
    LOG_JOB_ID="$(
      squeue -u "${USER}" -h -o "%i %t %j" | awk '$2=="R" {print $1}' | sort -nr | head -n1
    )"
  fi
  if [ -z "${LOG_JOB_ID}" ]; then
    LOG_JOB_ID="$(
      squeue -u "${USER}" -h -o "%i %t %j" | awk '$2=="PD" && $3 ~ /^smt-/ {print $1}' | sort -nr | head -n1
    )"
  fi
  if [ -z "${LOG_JOB_ID}" ]; then
    LOG_JOB_ID="$(
      squeue -u "${USER}" -h -o "%i %t %j" | awk '$2=="PD" {print $1}' | sort -nr | head -n1
    )"
  fi

  if [ -z "${LOG_JOB_ID}" ]; then
    echo "No active jobs found for ${USER}. Pass a job id explicitly: ./train.sh logs <JOB_ID>" >&2
    exit 1
  fi

  JOB_INFO=""
  if command -v scontrol >/dev/null 2>&1; then
    JOB_INFO="$(scontrol show job -o "${LOG_JOB_ID}" 2>/dev/null || true)"
  fi
  STDOUT_PATH=""
  STDERR_PATH=""
  if [ -n "${JOB_INFO}" ]; then
    STDOUT_PATH="$(echo "${JOB_INFO}" | tr ' ' '\n' | awk -F= '$1=="StdOut" {print $2}')"
    STDERR_PATH="$(echo "${JOB_INFO}" | tr ' ' '\n' | awk -F= '$1=="StdErr" {print $2}')"
  fi

  if [ -z "${STDOUT_PATH}" ]; then
    STDOUT_PATH="$(compgen -G "${LOG_DIR}/*-${LOG_JOB_ID}.out" | head -n1 || true)"
  fi
  if [ -z "${STDERR_PATH}" ]; then
    STDERR_PATH="$(compgen -G "${LOG_DIR}/*-${LOG_JOB_ID}.err" | head -n1 || true)"
  fi

  if [ -z "${STDOUT_PATH}" ] || [ -z "${STDERR_PATH}" ]; then
    echo "Could not resolve log files for job ${LOG_JOB_ID} under ${LOG_DIR}." >&2
    echo "Expected patterns: ${LOG_DIR}/*-${LOG_JOB_ID}.out and ${LOG_DIR}/*-${LOG_JOB_ID}.err" >&2
    exit 1
  fi

  echo "Job: ${LOG_JOB_ID}"
  echo "Out: ${STDOUT_PATH}"
  echo "Err: ${STDERR_PATH}"

  if [ "${STDOUT_PATH}" = "${STDERR_PATH}" ]; then
    if [ "${LOG_FOLLOW}" = true ]; then
      exec tail -n "${LOG_LINES}" -F "${STDOUT_PATH}"
    fi
    exec tail -n "${LOG_LINES}" "${STDOUT_PATH}"
  fi

  if [ "${LOG_FOLLOW}" = true ]; then
    exec tail -n "${LOG_LINES}" -F "${STDOUT_PATH}" "${STDERR_PATH}"
  fi
  exec tail -n "${LOG_LINES}" "${STDOUT_PATH}" "${STDERR_PATH}"
fi

if [ "${COMMAND}" != "submit" ] && [ "${COMMAND}" != "validate" ] && [ "${COMMAND}" != "shell" ] && [ "${COMMAND}" != "local" ] && [ "${COMMAND}" != "doctor" ] && [ "${COMMAND}" != "resources" ]; then
  echo "Unknown command: ${COMMAND}" >&2
  usage
  exit 2
fi

while [ $# -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --debug)
      CONFIG="${ROOT_DIR}/config/debug.json"
      shift
      ;;
    --profile)
      CONFIG="${ROOT_DIR}/config/profile.json"
      shift
      ;;
    --partition)
      PARTITION="$2"
      PARTITION_EXPLICIT=true
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --gpu-type)
      GPU_TYPE="$2"
      shift 2
      ;;
    --cpus-per-task)
      CPUS_PER_TASK="$2"
      shift 2
      ;;
    --mem)
      MEMORY="$2"
      shift 2
      ;;
    --time)
      TIME_LIMIT="$2"
      shift 2
      ;;
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --checkpoint_path|--checkpoint)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --fresh-run)
      FRESH_RUN=true
      shift
      ;;
    --gpu-id)
      GPU_ID="$2"
      shift 2
      ;;
    --no-sync)
      NO_SYNC=true
      shift
      ;;
    --no-doctor)
      NO_DOCTOR=true
      shift
      ;;
    --doctor-warn-only)
      DOCTOR_WARN_ONLY=true
      shift
      ;;
    --strict)
      DOCTOR_WARN_ONLY=false
      shift
      ;;
    --warn-only)
      DOCTOR_WARN_ONLY=true
      shift
      ;;
    --json)
      DOCTOR_JSON=true
      shift
      ;;
    --sbatch-arg)
      SLURM_EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --)
      shift
      FORWARDED_ARGS=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [ ! -f "${CONFIG}" ]; then
  if [[ "${CONFIG}" != /* ]]; then
    CONFIG="${ROOT_DIR}/${CONFIG#./}"
  fi
fi

if [ -n "${CHECKPOINT_PATH}" ] && [[ "${CHECKPOINT_PATH}" != /* ]]; then
  CHECKPOINT_PATH="${ROOT_DIR}/${CHECKPOINT_PATH#./}"
fi

if [ "${COMMAND}" = "doctor" ]; then
  if run_doctor; then
    exit 0
  fi
  exit 1
fi

if [ "${COMMAND}" = "resources" ]; then
  show_resources
  exit 0
fi

if { [ "${COMMAND}" = "submit" ] || [ "${COMMAND}" = "validate" ]; } && [ "${NO_DOCTOR}" = false ]; then
  if ! run_doctor; then
    echo "Preflight failed. Fix issues or rerun with --doctor-warn-only / --no-doctor." >&2
    exit 1
  fi
fi

if [ ! -f "${CONFIG}" ]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"

GRES="gpu:${GPUS}"
if [ -n "${GPU_TYPE}" ]; then
  GRES="gpu:${GPU_TYPE}:${GPUS}"
fi

if [ "${COMMAND}" = "shell" ]; then
  require_cmd srun
  SRUN_CMD=(
    srun
    "--partition=${PARTITION}"
    "--gres=${GRES}"
    "--cpus-per-task=${CPUS_PER_TASK}"
    "--time=${TIME_LIMIT}"
    --pty
    bash
    -lc
    "cd $(printf '%q' "${ROOT_DIR}") && if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi; exec bash -i"
  )

  if [ "${DRY_RUN}" = true ]; then
    printf 'Dry run: '
    printf '%q ' "${SRUN_CMD[@]}"
    echo
    exit 0
  fi

  exec "${SRUN_CMD[@]}"
fi

PY_CMD=(uv run train.py "${CONFIG}" "--seed=${SEED}")
if [ -n "${CHECKPOINT_PATH}" ]; then
  PY_CMD+=("--checkpoint_path=${CHECKPOINT_PATH}")
fi
if [ "${FRESH_RUN}" = true ]; then
  PY_CMD+=("--fresh_run=true")
fi
if [ "${COMMAND}" = "validate" ]; then
  PY_CMD+=("--validate_only=true")
  if ! has_forwarded_override "training.log_example_images"; then
    PY_CMD+=("--training.log_example_images=true")
  fi
fi
if [ ${#FORWARDED_ARGS[@]} -gt 0 ]; then
  PY_CMD+=("${FORWARDED_ARGS[@]}")
fi

if [ "${COMMAND}" = "local" ]; then
  if [ "${DRY_RUN}" = true ]; then
    printf 'Dry run: CUDA_VISIBLE_DEVICES=%q ' "${GPU_ID}"
    printf '%q ' "${PY_CMD[@]}"
    echo
    exit 0
  fi
  if [ "${NO_SYNC}" = false ]; then
    uv sync --group omr-ned
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PY_CMD[@]}"
  exit 0
fi

if [ "${COMMAND}" = "validate" ] && [ -z "${CHECKPOINT_PATH}" ]; then
  echo "validate mode requires --checkpoint_path PATH" >&2
  exit 2
fi

if [ -z "${JOB_NAME}" ]; then
  if [ "${COMMAND}" = "validate" ]; then
    JOB_NAME="smt-validate"
  else
    JOB_NAME="smt-train"
  fi
fi

SBATCH_CMD=(
  sbatch
  "--job-name=${JOB_NAME}"
  "--partition=${PARTITION}"
  "--gres=${GRES}"
  "--cpus-per-task=${CPUS_PER_TASK}"
  "--time=${TIME_LIMIT}"
  "--output=${LOG_DIR}/%x-%j.out"
  "--error=${LOG_DIR}/%x-%j.err"
  "--chdir=${ROOT_DIR}"
)

require_cmd sbatch

if [ -n "${MEMORY}" ]; then
  SBATCH_CMD+=("--mem=${MEMORY}")
fi

if [ ${#SLURM_EXTRA_ARGS[@]} -gt 0 ]; then
  SBATCH_CMD+=("${SLURM_EXTRA_ARGS[@]}")
fi

WRAP_SEGMENTS=("set -eu")
WRAP_SEGMENTS+=("cd $(printf '%q' "${ROOT_DIR}")")
WRAP_SEGMENTS+=("if [ -f .venv/bin/activate ]; then . .venv/bin/activate; fi")
if [ "${NO_SYNC}" = false ]; then
  WRAP_SEGMENTS+=("uv sync --group omr-ned")
fi
printf -v PY_ESCAPED '%q ' "${PY_CMD[@]}"
WRAP_SEGMENTS+=("${PY_ESCAPED% }")

WRAP_CMD=""
for segment in "${WRAP_SEGMENTS[@]}"; do
  if [ -n "${WRAP_CMD}" ]; then
    WRAP_CMD="${WRAP_CMD} && "
  fi
  WRAP_CMD="${WRAP_CMD}${segment}"
done

# sbatch --wrap defaults to /bin/sh on some clusters, so run through bash
# explicitly before sourcing the venv activation script.
printf -v WRAP_CMD_BASH 'bash -lc %q' "${WRAP_CMD}"
SBATCH_CMD+=("--wrap=${WRAP_CMD_BASH}")

if [ "${DRY_RUN}" = true ]; then
  printf 'Dry run: '
  printf '%q ' "${SBATCH_CMD[@]}"
  echo
  exit 0
fi

"${SBATCH_CMD[@]}"
