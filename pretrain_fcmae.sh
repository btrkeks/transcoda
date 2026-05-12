#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
LOG_DIR="${ROOT_DIR}/logs"
LAST_RUN_STATE_FILE="${LOG_DIR}/pretrain_fcmae-last-run.env"

COMMAND="submit"
if [ $# -gt 0 ]; then
  case "$1" in
    submit|local|queue|cancel|logs|export|help|-h|--help)
      COMMAND="$1"
      shift
      ;;
  esac
fi

CONFIG="${ROOT_DIR}/config/pretrain_fcmae_base.json"
PARTITION="gpu"
GPUS=1
GPU_TYPE=""
DEFAULT_CPUS_PER_TASK=8
CPUS_PER_TASK=${DEFAULT_CPUS_PER_TASK}
MEMORY=""
TIME_LIMIT="24:00:00"
JOB_NAME="fcmae-pretrain"
RESUME_PATH=""
NO_SYNC=false
DRY_RUN=false
SLURM_EXTRA_ARGS=()
FORWARDED_ARGS=()
AUTO_FORWARDED_ARGS=()
CHECKPOINT_DIR=""
EXPORT_DIR=""
RESOLVED_RUN_ID=""

usage() {
  cat <<'EOF'
Usage:
  ./pretrain_fcmae.sh submit [options] -- [config overrides]
  ./pretrain_fcmae.sh local [options] -- [config overrides]
  ./pretrain_fcmae.sh queue
  ./pretrain_fcmae.sh cancel JOB_ID
  ./pretrain_fcmae.sh logs [JOB_ID] [--lines N] [--no-follow]
  ./pretrain_fcmae.sh export CHECKPOINT_PATH OUTPUT_DIR [--overwrite] [--validate]
  ./pretrain_fcmae.sh help

Options:
  --config PATH            Config JSON (default: config/pretrain_fcmae_base.json)
  --partition NAME         Slurm partition (default: gpu)
  --gpus N                 Number of GPUs (default: 1)
  --gpu-type TYPE          GPU type for --gres
  --cpus-per-task N        CPUs per task (default: 8)
  --mem SIZE               Memory request
  --time HH:MM:SS          Time limit (default: 24:00:00)
  --job-name NAME          Slurm job name (default: fcmae-pretrain)
  --sbatch-arg ARG         Additional sbatch arg (repeatable)
  --resume PATH            Forward training.resume_from_checkpoint=PATH
  --no-sync                Skip 'uv sync --group omr-ned' in the Slurm job body
  --dry-run                Print rendered command/job script and exit
  -h, --help               Show this help

Examples:
  ./pretrain_fcmae.sh submit --time 48:00:00 --mem 64G -- data.manifest_path=data/fcmae.txt checkpoint.dirpath=weights/fcmae-real
  ./pretrain_fcmae.sh local -- training.max_steps=10 logging.wandb_enabled=false
  ./pretrain_fcmae.sh export weights/fcmae-real/last.ckpt weights/fcmae-real/exported_encoder --validate
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command '$1' not found in PATH." >&2
    exit 127
  fi
}

normalize_path_to_root() {
  local path="$1"
  if [[ "${path}" != /* ]]; then
    path="${ROOT_DIR}/${path#./}"
  fi
  printf '%s' "${path}"
}

resolve_config_value() {
  local config_path="$1"
  local key_path="$2"
  local py_bin="python3"
  if ! command -v "${py_bin}" >/dev/null 2>&1; then
    py_bin="python"
  fi
  "${py_bin}" - "${config_path}" "${key_path}" <<'PY'
import json
import sys

config_path, key_path = sys.argv[1], sys.argv[2]
with open(config_path) as fh:
    value = json.load(fh)
for part in key_path.split("."):
    if not isinstance(value, dict) or part not in value:
        raise SystemExit(2)
    value = value[part]
if value is None or isinstance(value, (dict, list)):
    raise SystemExit(3)
if isinstance(value, bool):
    print("true" if value else "false", end="")
else:
    print(value, end="")
PY
}

get_forwarded_override_value() {
  local key="$1"
  local arg
  for arg in "${FORWARDED_ARGS[@]}"; do
    arg="${arg#--}"
    if [[ "${arg}" == "${key}="* ]]; then
      printf '%s' "${arg#${key}=}"
      return 0
    fi
  done
  return 1
}

get_auto_forwarded_override_value() {
  local key="$1"
  local arg
  for arg in "${AUTO_FORWARDED_ARGS[@]}"; do
    arg="${arg#--}"
    if [[ "${arg}" == "${key}="* ]]; then
      printf '%s' "${arg#${key}=}"
      return 0
    fi
  done
  return 1
}

resolve_effective_config_value() {
  local key="$1"
  local value=""
  if value="$(get_forwarded_override_value "${key}")"; then
    printf '%s' "${value}"
    return 0
  fi
  if value="$(get_auto_forwarded_override_value "${key}")"; then
    printf '%s' "${value}"
    return 0
  fi
  resolve_config_value "${CONFIG}" "${key}"
}

slugify_identifier() {
  local s="$1"
  s="$(printf '%s' "${s}" | tr '[:upper:]' '[:lower:]')"
  s="$(printf '%s' "${s}" | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')"
  if [ -z "${s}" ]; then
    s="run"
  fi
  printf '%s' "${s}"
}

has_forwarded_override() {
  local key="$1"
  local arg
  for arg in "${FORWARDED_ARGS[@]}"; do
    arg="${arg#--}"
    if [[ "${arg}" == "${key}="* ]]; then
      return 0
    fi
  done
  for arg in "${AUTO_FORWARDED_ARGS[@]}"; do
    arg="${arg#--}"
    if [[ "${arg}" == "${key}="* ]]; then
      return 0
    fi
  done
  return 1
}

append_override() {
  local arg="$1"
  FORWARDED_ARGS+=("${arg#--}")
}

prepare_submit_run_identity() {
  local configured_checkpoint_dir=""
  local configured_run_name=""
  local checkpoint_basename=""
  local checkpoint_parent=""
  local run_seed=""
  local timestamp=""
  local export_on_train_end=""

  if [ "${COMMAND}" != "submit" ]; then
    return 0
  fi

  if configured_checkpoint_dir="$(get_forwarded_override_value "checkpoint.dirpath")"; then
    CHECKPOINT_DIR="$(normalize_path_to_root "${configured_checkpoint_dir}")"
    RESOLVED_RUN_ID="${CHECKPOINT_DIR##*/}"
  else
    configured_checkpoint_dir="$(resolve_config_value "${CONFIG}" "checkpoint.dirpath")"
    CHECKPOINT_DIR="$(normalize_path_to_root "${configured_checkpoint_dir}")"
    checkpoint_basename="${CHECKPOINT_DIR##*/}"
    checkpoint_parent="${CHECKPOINT_DIR%/*}"
    if [ -z "${checkpoint_parent}" ] || [ "${checkpoint_parent}" = "${CHECKPOINT_DIR}" ]; then
      checkpoint_parent="${ROOT_DIR}"
    fi

    if configured_run_name="$(resolve_effective_config_value "logging.run_name")" && [ -n "${configured_run_name}" ]; then
      run_seed="$(slugify_identifier "${configured_run_name}")"
    else
      run_seed="$(slugify_identifier "${checkpoint_basename}")"
    fi
    timestamp="$(date +%Y%m%d-%H%M%S)"
    RESOLVED_RUN_ID="${run_seed}-${timestamp}"
    CHECKPOINT_DIR="${checkpoint_parent}/${RESOLVED_RUN_ID}"
    AUTO_FORWARDED_ARGS+=("checkpoint.dirpath=${CHECKPOINT_DIR}")

    if ! get_forwarded_override_value "logging.run_name" >/dev/null; then
      AUTO_FORWARDED_ARGS+=("logging.run_name=${RESOLVED_RUN_ID}")
    fi
  fi

  if EXPORT_DIR="$(get_forwarded_override_value "export.output_dir")"; then
    EXPORT_DIR="$(normalize_path_to_root "${EXPORT_DIR}")"
    return 0
  fi

  if export_on_train_end="$(resolve_effective_config_value "export.export_on_train_end")" \
    && [ "${export_on_train_end}" = "true" ]; then
    EXPORT_DIR="${CHECKPOINT_DIR}/exported_encoder"
    AUTO_FORWARDED_ARGS+=("export.output_dir=${EXPORT_DIR}")
  fi
}

write_last_run_state() {
  local job_id="$1"
  mkdir -p "${LOG_DIR}"
  {
    printf 'LAST_RUN_JOB_ID=%q\n' "${job_id}"
    printf 'LAST_RUN_CONFIG=%q\n' "${CONFIG}"
    printf 'LAST_RUN_CHECKPOINT_DIR=%q\n' "${CHECKPOINT_DIR}"
    printf 'LAST_RUN_EXPORT_DIR=%q\n' "${EXPORT_DIR}"
  } >"${LAST_RUN_STATE_FILE}"
}

load_last_run_state() {
  if [ ! -f "${LAST_RUN_STATE_FILE}" ]; then
    return 1
  fi
  source "${LAST_RUN_STATE_FILE}"
}

if [ "${COMMAND}" = "help" ] || [ "${COMMAND}" = "-h" ] || [ "${COMMAND}" = "--help" ]; then
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
    echo "Missing job id. Usage: ./pretrain_fcmae.sh cancel JOB_ID" >&2
    exit 2
  fi
  scancel "${JOB_ID}"
  exit 0
fi

if [ "${COMMAND}" = "logs" ]; then
  LOG_JOB_ID=""
  LOG_LINES=50
  LOG_FOLLOW=true
  if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
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
        echo "Usage: ./pretrain_fcmae.sh logs [JOB_ID] [--lines N] [--no-follow]"
        exit 0
        ;;
      *)
        echo "Unknown logs option: $1" >&2
        exit 2
        ;;
    esac
  done
  if [ -z "${LOG_JOB_ID}" ] && load_last_run_state; then
    LOG_JOB_ID="${LAST_RUN_JOB_ID:-}"
  fi
  if [ -z "${LOG_JOB_ID}" ]; then
    require_cmd squeue
    LOG_JOB_ID="$(squeue -u "${USER}" -h -o "%i %j" | awk '$2 ~ /^fcmae/ {print $1}' | sort -nr | head -n1)"
  fi
  if [ -z "${LOG_JOB_ID}" ]; then
    echo "No FCMAE job found. Pass a job id explicitly." >&2
    exit 1
  fi
  STDOUT_PATH="$(compgen -G "${LOG_DIR}/*-${LOG_JOB_ID}.out" | head -n1 || true)"
  STDERR_PATH="$(compgen -G "${LOG_DIR}/*-${LOG_JOB_ID}.err" | head -n1 || true)"
  if [ -z "${STDOUT_PATH}" ] || [ -z "${STDERR_PATH}" ]; then
    echo "Could not resolve log files for job ${LOG_JOB_ID} under ${LOG_DIR}." >&2
    exit 1
  fi
  echo "Job: ${LOG_JOB_ID}"
  echo "Out: ${STDOUT_PATH}"
  echo "Err: ${STDERR_PATH}"
  if [ "${LOG_FOLLOW}" = true ]; then
    exec tail -n "${LOG_LINES}" -F "${STDOUT_PATH}" "${STDERR_PATH}"
  fi
  exec tail -n "${LOG_LINES}" "${STDOUT_PATH}" "${STDERR_PATH}"
fi

if [ "${COMMAND}" = "export" ]; then
  CHECKPOINT_PATH="${1:-}"
  OUTPUT_DIR="${2:-}"
  if [ -z "${CHECKPOINT_PATH}" ] || [ -z "${OUTPUT_DIR}" ]; then
    echo "Usage: ./pretrain_fcmae.sh export CHECKPOINT_PATH OUTPUT_DIR [--overwrite] [--validate]" >&2
    exit 2
  fi
  shift 2
  EXPORT_ARGS=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --overwrite|--validate)
        EXPORT_ARGS+=("$1")
        shift
        ;;
      *)
        echo "Unknown export option: $1" >&2
        exit 2
        ;;
    esac
  done
  python -m scripts.export_fcmae_encoder "${CHECKPOINT_PATH}" "${OUTPUT_DIR}" "${EXPORT_ARGS[@]}"
  exit 0
fi

if [ "${COMMAND}" != "submit" ] && [ "${COMMAND}" != "local" ]; then
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
    --partition)
      PARTITION="$2"
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
    --sbatch-arg)
      SLURM_EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --resume)
      RESUME_PATH="$2"
      shift 2
      ;;
    --no-sync)
      NO_SYNC=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --)
      shift
      while [ $# -gt 0 ]; do
        append_override "$1"
        shift
      done
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

CONFIG="$(normalize_path_to_root "${CONFIG}")"
if [ -n "${RESUME_PATH}" ]; then
  append_override "training.resume_from_checkpoint=${RESUME_PATH}"
fi

# Multi-GPU defaults: mirror the main SFT launcher by selecting Lightning DDP
# unless the caller explicitly forwards trainer settings.
if [ "${GPUS}" -gt 1 ]; then
  if ! has_forwarded_override "training.devices"; then
    AUTO_FORWARDED_ARGS+=("training.devices=${GPUS}")
  fi
  if ! has_forwarded_override "training.strategy"; then
    AUTO_FORWARDED_ARGS+=("training.strategy=ddp")
  fi
  if [ "${CPUS_PER_TASK}" -eq "${DEFAULT_CPUS_PER_TASK}" ]; then
    CPUS_PER_TASK=$((DEFAULT_CPUS_PER_TASK * GPUS))
    echo "[pretrain_fcmae.sh] auto-scaled --cpus-per-task to ${CPUS_PER_TASK} for ${GPUS} GPUs (DDP)" >&2
  fi
  if ! has_forwarded_override "training.accumulate_grad_batches"; then
    ACCUMULATE_GRAD_BATCHES="$(resolve_effective_config_value "training.accumulate_grad_batches")"
    SCALED_ACCUMULATE_GRAD_BATCHES=$((ACCUMULATE_GRAD_BATCHES / GPUS))
    if [ "${SCALED_ACCUMULATE_GRAD_BATCHES}" -lt 1 ]; then
      SCALED_ACCUMULATE_GRAD_BATCHES=1
    fi
    AUTO_FORWARDED_ARGS+=("training.accumulate_grad_batches=${SCALED_ACCUMULATE_GRAD_BATCHES}")
    if [ $((ACCUMULATE_GRAD_BATCHES % GPUS)) -eq 0 ]; then
      echo "[pretrain_fcmae.sh] auto-scaled training.accumulate_grad_batches to ${SCALED_ACCUMULATE_GRAD_BATCHES} for ${GPUS} GPUs" >&2
    else
      echo "[pretrain_fcmae.sh] auto-scaled training.accumulate_grad_batches to ${SCALED_ACCUMULATE_GRAD_BATCHES} for ${GPUS} GPUs (rounded down from ${ACCUMULATE_GRAD_BATCHES}/${GPUS})" >&2
    fi
  fi
fi

prepare_submit_run_identity

if [ "${COMMAND}" = "local" ]; then
  if ! CHECKPOINT_DIR="$(get_forwarded_override_value "checkpoint.dirpath")"; then
    CHECKPOINT_DIR="$(resolve_config_value "${CONFIG}" "checkpoint.dirpath" || true)"
  fi
  if [ -n "${CHECKPOINT_DIR}" ]; then
    CHECKPOINT_DIR="$(normalize_path_to_root "${CHECKPOINT_DIR}")"
  fi
  if EXPORT_DIR="$(get_forwarded_override_value "export.output_dir")"; then
    EXPORT_DIR="$(normalize_path_to_root "${EXPORT_DIR}")"
  fi
fi

PY_CMD=(python -m scripts.pretrain_fcmae "${CONFIG}")
if [ ${#AUTO_FORWARDED_ARGS[@]} -gt 0 ]; then
  PY_CMD+=("${AUTO_FORWARDED_ARGS[@]}")
fi
if [ ${#FORWARDED_ARGS[@]} -gt 0 ]; then
  PY_CMD+=("${FORWARDED_ARGS[@]}")
fi

mkdir -p "${LOG_DIR}"

if [ "${COMMAND}" = "local" ]; then
  if [ "${DRY_RUN}" = true ]; then
    printf 'Dry run: '
    printf '%q ' "${PY_CMD[@]}"
    echo
    exit 0
  fi
  "${PY_CMD[@]}"
  exit 0
fi

GRES="gpu:${GPUS}"
if [ -n "${GPU_TYPE}" ]; then
  GRES="gpu:${GPU_TYPE}:${GPUS}"
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
if [ -n "${MEMORY}" ]; then
  SBATCH_CMD+=("--mem=${MEMORY}")
fi
if [ ${#SLURM_EXTRA_ARGS[@]} -gt 0 ]; then
  SBATCH_CMD+=("${SLURM_EXTRA_ARGS[@]}")
fi

WRAP_SEGMENTS=("set -eu")
WRAP_SEGMENTS+=("cd $(printf '%q' "${ROOT_DIR}")")
WRAP_SEGMENTS+=("source .venv/bin/activate")
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
SBATCH_CMD+=(--wrap "bash -lc $(printf '%q' "${WRAP_CMD}")")

if [ "${DRY_RUN}" = true ]; then
  echo "Dry run sbatch command:"
  printf '%q ' "${SBATCH_CMD[@]}"
  echo
  echo
  echo "Rendered job body:"
  echo "${WRAP_CMD}"
  echo "bash -lc $(printf '%q' "${WRAP_CMD}")"
  exit 0
fi

require_cmd sbatch
SBATCH_OUTPUT="$("${SBATCH_CMD[@]}")"
echo "${SBATCH_OUTPUT}"
JOB_ID="$(printf '%s\n' "${SBATCH_OUTPUT}" | awk '{print $NF}' | tail -n1)"
write_last_run_state "${JOB_ID}"
