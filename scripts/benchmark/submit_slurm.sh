#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"

COMMAND="submit"
if [ $# -gt 0 ]; then
  case "$1" in
    submit|queue|cancel|logs|help|-h|--help)
      COMMAND="$1"
      shift
      ;;
  esac
fi

PARTITION="gpu"
GPUS=1
GPU_TYPE=""
CPUS_PER_TASK=8
MEMORY=""
TIME_LIMIT="24:00:00"
JOB_NAME="smt-bench"
NO_SYNC=false
DRY_RUN=false
SBATCH_ARGS=()
BENCHMARK_ARGS=()

DATASET_ROOT="data/datasets/benchmark"
DATASETS=""
MODELS="ours"
METRICS="omr_ned,tedn,cer"
OUTPUT_ROOT="outputs/benchmark"
OURS_CHECKPOINT="weights/GrandStaff/smt-model.ckpt"
OURS_STRATEGY=""
OURS_NUM_BEAMS=""
SMTPP_MODEL_ID="PRAIG/smt-fp-grandstaff"
SMTPP_MAX_LENGTH=""
LEGATO_MODEL_ID="guangyangmusic/legato"
LEGATO_ENCODER_PATH=""
LEGATO_MAX_LENGTH=2048
LEGATO_NUM_BEAMS=10
DEVICE="cuda"
LIMIT=""
BATCH_SIZE="auto"
METRIC_WORKERS="auto"
HUM2XML_PATH="hum2xml"
ABC2XML_PATH="abc2xml"
RESUME=false
SKIP_INFERENCE=false
SKIP_INVALID_GOLD=false
OURS_NORMALIZE_LAYOUT=false
DISABLE_CONSTRAINTS=false

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command '$1' not found in PATH." >&2
    exit 127
  fi
}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/benchmark/submit_slurm.sh [submit] [options] -- [extra benchmark args]
  ./scripts/benchmark/submit_slurm.sh queue
  ./scripts/benchmark/submit_slurm.sh cancel JOB_ID
  ./scripts/benchmark/submit_slurm.sh logs [JOB_ID] [--lines N] [--no-follow]

Slurm options:
  --partition NAME         Slurm partition (default: gpu)
  --gpus N                 Number of GPUs (default: 1)
  --gpu-type TYPE          GPU type for --gres (e.g. rtx4090)
  --cpus-per-task N        CPUs per task (default: 8)
  --mem SIZE               Memory request (default: unset)
  --time HH:MM:SS          Time limit (default: 24:00:00)
  --job-name NAME          Slurm job name (default: smt-bench)
  --sbatch-arg ARG         Additional sbatch arg (repeatable)
  --no-sync                Skip 'uv sync --group omr-ned'
  --dry-run                Print commands and exit

Benchmark options:
  --dataset-root PATH      Benchmark dataset root
  --datasets CSV           Dataset subset (e.g. synth)
  --models CSV             Model subset (default: ours)
  --metrics CSV            Metric subset (default: omr_ned,tedn,cer)
  --output-root PATH       Benchmark output root
  --ours-checkpoint PATH   Local checkpoint path for our model
  --ours-strategy MODE     Our adapter decode strategy override (greedy|beam)
  --ours-num-beams N       Our adapter beam width override
  --smtpp-model-id ID      SMT++ HF model id
  --smtpp-max-length N     SMT++ max output length
  --legato-model-id ID     LEGATO HF model id
  --legato-encoder-path P  LEGATO vision encoder path or HF id override
  --legato-max-length N    LEGATO max output length
  --legato-num-beams N     LEGATO beam size
  --device DEVICE          Benchmark device (default: cuda)
  --limit N                Limit dataset rows
  --batch-size VALUE       Benchmark batch size or 'auto'
  --metric-workers VALUE   Metric worker count or 'auto'
  --hum2xml-path PATH      hum2xml executable path
  --abc2xml-path PATH      abc2xml executable path
  --resume                 Resume latest benchmark output root
  --skip-inference         Reuse cached raw predictions
  --skip-invalid-gold      Skip invalid gold conversions
  --ours-normalize-layout  Enable layout normalization for our adapter
  --disable-constraints    Disable all constrained decoding for our adapter

Examples:
  ./scripts/benchmark/submit_slurm.sh \
    --job-name smt-bench-synth-omrned \
    --datasets synth \
    --models ours \
    --metrics omr_ned \
    --ours-checkpoint weights/train_full_page_aug_finetune_v1/smt-model.ckpt \
    --output-root outputs/benchmark_train_full_page_aug_finetune_v1_synth_omr_ned

  ./scripts/benchmark/submit_slurm.sh queue
  ./scripts/benchmark/submit_slurm.sh logs
EOF
}

if [ "${COMMAND}" = "-h" ] || [ "${COMMAND}" = "--help" ] || [ "${COMMAND}" = "help" ]; then
  usage
  exit 0
fi

if [ "${COMMAND}" = "queue" ]; then
  require_cmd squeue
  exec squeue -u "${USER}" -o "%.18i %.9P %.8j %.2t %.10M %.6D %R"
fi

if [ "${COMMAND}" = "cancel" ]; then
  require_cmd scancel
  JOB_ID="${1:-}"
  if [ -z "${JOB_ID}" ]; then
    echo "Missing job id. Usage: ./scripts/benchmark/submit_slurm.sh cancel JOB_ID" >&2
    exit 2
  fi
  exec scancel "${JOB_ID}"
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
        echo "Usage: ./scripts/benchmark/submit_slurm.sh logs [JOB_ID] [--lines N] [--no-follow]"
        exit 0
        ;;
      *)
        echo "Unknown logs option: $1" >&2
        exit 2
        ;;
    esac
  done

  require_cmd tail
  require_cmd squeue

  if [ -z "${LOG_JOB_ID}" ]; then
    LOG_JOB_ID="$(
      squeue -u "${USER}" -h -o "%i %t %j" \
        | awk '$2=="R" && $3 ~ /^smt-bench/ {print $1}' \
        | sort -nr \
        | head -n1
    )"
  fi
  if [ -z "${LOG_JOB_ID}" ]; then
    LOG_JOB_ID="$(
      squeue -u "${USER}" -h -o "%i %t %j" \
        | awk '$2=="PD" && $3 ~ /^smt-bench/ {print $1}' \
        | sort -nr \
        | head -n1
    )"
  fi

  if [ -z "${LOG_JOB_ID}" ]; then
    echo "No active benchmark jobs found for ${USER}. Pass a job id explicitly." >&2
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

while [ $# -gt 0 ]; do
  case "$1" in
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
      SBATCH_ARGS+=("$2")
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
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    --models)
      MODELS="$2"
      shift 2
      ;;
    --metrics)
      METRICS="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --ours-checkpoint)
      OURS_CHECKPOINT="$2"
      shift 2
      ;;
    --ours-strategy)
      OURS_STRATEGY="$2"
      shift 2
      ;;
    --ours-num-beams)
      OURS_NUM_BEAMS="$2"
      shift 2
      ;;
    --smtpp-model-id)
      SMTPP_MODEL_ID="$2"
      shift 2
      ;;
    --smtpp-max-length)
      SMTPP_MAX_LENGTH="$2"
      shift 2
      ;;
    --legato-model-id)
      LEGATO_MODEL_ID="$2"
      shift 2
      ;;
    --legato-encoder-path)
      LEGATO_ENCODER_PATH="$2"
      shift 2
      ;;
    --legato-max-length)
      LEGATO_MAX_LENGTH="$2"
      shift 2
      ;;
    --legato-num-beams)
      LEGATO_NUM_BEAMS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --metric-workers)
      METRIC_WORKERS="$2"
      shift 2
      ;;
    --hum2xml-path)
      HUM2XML_PATH="$2"
      shift 2
      ;;
    --abc2xml-path)
      ABC2XML_PATH="$2"
      shift 2
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --skip-inference)
      SKIP_INFERENCE=true
      shift
      ;;
    --skip-invalid-gold)
      SKIP_INVALID_GOLD=true
      shift
      ;;
    --ours-normalize-layout)
      OURS_NORMALIZE_LAYOUT=true
      shift
      ;;
    --disable-constraints)
      DISABLE_CONSTRAINTS=true
      shift
      ;;
    --)
      shift
      BENCHMARK_ARGS=("$@")
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

mkdir -p "${LOG_DIR}"

GRES="gpu:${GPUS}"
if [ -n "${GPU_TYPE}" ]; then
  GRES="gpu:${GPU_TYPE}:${GPUS}"
fi

RUN_CMD=(
  uv run python scripts/benchmark/run.py
  "--dataset-root=${DATASET_ROOT}"
  "--datasets=${DATASETS}"
  "--models=${MODELS}"
  "--metrics=${METRICS}"
  "--output-root=${OUTPUT_ROOT}"
  "--ours-checkpoint=${OURS_CHECKPOINT}"
  "--smtpp-model-id=${SMTPP_MODEL_ID}"
  "--legato-model-id=${LEGATO_MODEL_ID}"
  "--legato-max-length=${LEGATO_MAX_LENGTH}"
  "--legato-num-beams=${LEGATO_NUM_BEAMS}"
  "--device=${DEVICE}"
  "--batch-size=${BATCH_SIZE}"
  "--metric-workers=${METRIC_WORKERS}"
  "--hum2xml-path=${HUM2XML_PATH}"
  "--abc2xml-path=${ABC2XML_PATH}"
)

if [ -n "${OURS_STRATEGY}" ]; then
  RUN_CMD+=("--ours-strategy=${OURS_STRATEGY}")
fi
if [ -n "${OURS_NUM_BEAMS}" ]; then
  RUN_CMD+=("--ours-num-beams=${OURS_NUM_BEAMS}")
fi
if [ -n "${SMTPP_MAX_LENGTH}" ]; then
  RUN_CMD+=("--smtpp-max-length=${SMTPP_MAX_LENGTH}")
fi
if [ -n "${LEGATO_ENCODER_PATH}" ]; then
  RUN_CMD+=("--legato-encoder-path=${LEGATO_ENCODER_PATH}")
fi
if [ -n "${LIMIT}" ]; then
  RUN_CMD+=("--limit=${LIMIT}")
fi
if [ "${RESUME}" = true ]; then
  RUN_CMD+=("--resume")
fi
if [ "${SKIP_INFERENCE}" = true ]; then
  RUN_CMD+=("--skip-inference")
fi
if [ "${SKIP_INVALID_GOLD}" = true ]; then
  RUN_CMD+=("--skip-invalid-gold")
fi
if [ "${OURS_NORMALIZE_LAYOUT}" = true ]; then
  RUN_CMD+=("--ours-normalize-layout")
fi
if [ "${DISABLE_CONSTRAINTS}" = true ]; then
  RUN_CMD+=("--disable-constraints")
fi
if [ ${#BENCHMARK_ARGS[@]} -gt 0 ]; then
  RUN_CMD+=("${BENCHMARK_ARGS[@]}")
fi

RUN_CMD_STRING=""
printf -v RUN_CMD_STRING '%q ' "${RUN_CMD[@]}"
RUN_CMD_STRING="${RUN_CMD_STRING% }"

TIMESTAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
JOB_SCRIPT="${LOG_DIR}/${JOB_NAME}-${TIMESTAMP_UTC}.sbatch"

cat > "${JOB_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOG_DIR}/%x-%j.out
#SBATCH --error=${LOG_DIR}/%x-%j.err
#SBATCH --chdir=${ROOT_DIR}
EOF

if [ -n "${MEMORY}" ]; then
  printf '#SBATCH --mem=%s\n' "${MEMORY}" >> "${JOB_SCRIPT}"
fi

if [ ${#SBATCH_ARGS[@]} -gt 0 ]; then
  for arg in "${SBATCH_ARGS[@]}"; do
    printf '#SBATCH %s\n' "${arg}" >> "${JOB_SCRIPT}"
  done
fi

cat >> "${JOB_SCRIPT}" <<EOF

set -euo pipefail
cd $(printf '%q' "${ROOT_DIR}")
export PATH="\$HOME/.local/bin:\$PATH"
set +u
source .venv/bin/activate
set -u
EOF

if [ "${NO_SYNC}" = false ]; then
  printf 'uv sync --group omr-ned\n' >> "${JOB_SCRIPT}"
fi

printf 'exec %s\n' "${RUN_CMD_STRING}" >> "${JOB_SCRIPT}"
chmod +x "${JOB_SCRIPT}"

SBATCH_CMD=(sbatch --parsable "${JOB_SCRIPT}")

if [ "${DRY_RUN}" = true ]; then
  printf 'Dry run job script: %s\n' "${JOB_SCRIPT}"
  sed -n '1,220p' "${JOB_SCRIPT}"
  printf 'Dry run submit command: '
  printf '%q ' "${SBATCH_CMD[@]}"
  echo
  exit 0
fi

require_cmd sbatch
JOB_ID="$("${SBATCH_CMD[@]}")"
echo "Submitted benchmark job ${JOB_ID}"
echo "Job script: ${JOB_SCRIPT}"
echo "Stdout: ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.out"
echo "Stderr: ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.err"
