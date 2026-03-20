#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

RUNPOD_HOST="${RUNPOD_HOST:-runpod}"
RUNPOD_REMOTE_DIR="${RUNPOD_REMOTE_DIR:-/workspace/omr}"
RUNPOD_REPO_BRANCH="${RUNPOD_REPO_BRANCH:-master}"
RUNPOD_REPO_TRANSFER_MODE="${RUNPOD_REPO_TRANSFER_MODE:-auto}"
RUNPOD_DATASET_NAME="${RUNPOD_DATASET_NAME:-benchmark}"
RUNPOD_LOCAL_DATASET_DIR="${RUNPOD_LOCAL_DATASET_DIR:-${REPO_ROOT}/data/datasets/${RUNPOD_DATASET_NAME}}"
RUNPOD_REMOTE_DATASETS_DIR="${RUNPOD_REMOTE_DATASETS_DIR:-${RUNPOD_REMOTE_DIR}/data/datasets}"
RUNPOD_BENCHMARK_SUPPORT_PATHS="${RUNPOD_BENCHMARK_SUPPORT_PATHS:-scripts/benchmark,src/benchmark,models/external/legato,models/external/praig-smt,tools/abc2xml}"
RUNPOD_CHECKPOINT_NAME="${RUNPOD_CHECKPOINT_NAME:-long-data-augment-finetune.ckpt}"
RUNPOD_LOCAL_CHECKPOINT_PATH="${RUNPOD_LOCAL_CHECKPOINT_PATH:-${REPO_ROOT}/weights/${RUNPOD_CHECKPOINT_NAME}}"
RUNPOD_REMOTE_WEIGHTS_DIR="${RUNPOD_REMOTE_WEIGHTS_DIR:-${RUNPOD_REMOTE_DIR}/weights}"
RUNPOD_TRANSFER_CHECKPOINT="${RUNPOD_TRANSFER_CHECKPOINT:-auto}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

require_command() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
}

resolve_repo_url() {
  local remote_name
  for remote_name in private origin; do
    if git remote get-url "${remote_name}" >/dev/null 2>&1; then
      git remote get-url "${remote_name}"
      return 0
    fi
  done
  return 1
}

inject_github_token() {
  local repo_url="$1"
  local token="$2"

  if [[ -z "${token}" ]]; then
    printf '%s\n' "${repo_url}"
    return 0
  fi

  if [[ "${repo_url}" =~ ^https://github\.com/ ]]; then
    printf 'https://%s@%s\n' "${token}" "${repo_url#https://}"
    return 0
  fi

  printf '%s\n' "${repo_url}"
}

normalize_boolean() {
  local value="${1,,}"
  case "${value}" in
    1|true|yes|on)
      printf 'true\n'
      ;;
    0|false|no|off)
      printf 'false\n'
      ;;
    *)
      echo "Expected boolean-like value, got '${1}'" >&2
      exit 1
      ;;
  esac
}

resolve_repo_transfer_mode() {
  case "${RUNPOD_REPO_TRANSFER_MODE}" in
    auto)
      if [[ -n "${GITHUB_TOKEN}" ]]; then
        printf 'git\n'
      else
        printf 'snapshot\n'
      fi
      ;;
    git|snapshot)
      printf '%s\n' "${RUNPOD_REPO_TRANSFER_MODE}"
      ;;
    *)
      echo "Unsupported RUNPOD_REPO_TRANSFER_MODE='${RUNPOD_REPO_TRANSFER_MODE}'. Use auto, git, or snapshot." >&2
      exit 1
      ;;
  esac
}

should_transfer_checkpoint() {
  case "${RUNPOD_TRANSFER_CHECKPOINT}" in
    auto)
      [[ -f "${RUNPOD_LOCAL_CHECKPOINT_PATH}" ]]
      ;;
    *)
      [[ "$(normalize_boolean "${RUNPOD_TRANSFER_CHECKPOINT}")" == "true" ]]
      ;;
  esac
}

create_repo_snapshot() {
  local snapshot_path
  snapshot_path="$(mktemp "${TMPDIR:-/tmp}/runpod-repo-snapshot-XXXXXX.tgz")"
  tar \
    --exclude='./.git' \
    --exclude='./.venv' \
    --exclude='./data/datasets' \
    --exclude='./weights' \
    --exclude='./outputs' \
    --exclude='./logs' \
    --exclude='./analysis' \
    --exclude='./__pycache__' \
    --exclude='*.pyc' \
    -czf "${snapshot_path}" \
    -C "${REPO_ROOT}" \
    .
  printf '%s\n' "${snapshot_path}"
}

wait_for_transfer_code() {
  local log_path="$1"
  local sender_pid="$2"
  local transfer_code=""

  while [[ -z "${transfer_code}" ]]; do
    if ! kill -0 "${sender_pid}" >/dev/null 2>&1; then
      cat "${log_path}" >&2 || true
      echo "runpodctl send exited before emitting a transfer code." >&2
      exit 1
    fi

    transfer_code="$(sed -n 's/^code is: //p' "${log_path}" | tail -n1)"
    sleep 1
  done

  printf '%s\n' "${transfer_code}"
}

run_remote_script() {
  local remote_script="$1"

  {
    printf '%s\n' 'set -euo pipefail'
    printf '%s\n' "${remote_script}"
  } | ssh -T "${RUNPOD_HOST}" 'bash -s'
}

transfer_with_runpodctl() {
  local local_path="$1"
  local remote_dir="$2"
  local remote_cleanup_path="$3"
  local remote_verify_path="$4"
  local verify_type="$5"
  local label="$6"
  local safe_label
  local send_log
  local transfer_code

  echo "==> Starting local ${label} transfer"
  safe_label="$(printf '%s' "${label}" | sed 's/[^[:alnum:]. _-]/-/g')"
  safe_label="${safe_label// /-}"
  send_log="$(mktemp "${TMPDIR:-/tmp}/runpod-transfer-${safe_label}-XXXXXX.log")"
  SEND_LOG="${send_log}"
  runpodctl send "${local_path}" >"${SEND_LOG}" 2>&1 &
  SENDER_PID="$!"
  transfer_code="$(wait_for_transfer_code "${SEND_LOG}" "${SENDER_PID}")"
  echo "Transfer code received for ${label}: ${transfer_code}"

  echo "==> Receiving ${label} on ${RUNPOD_HOST}:${remote_dir}"
  run_remote_script "$(cat <<EOF
remote_dir=$(printf '%q' "${remote_dir}")
cleanup_path=$(printf '%q' "${remote_cleanup_path}")
transfer_code=$(printf '%q' "${transfer_code}")

mkdir -p "\${remote_dir}"
cd "\${remote_dir}"
rm -rf "\${cleanup_path}"
runpodctl receive "\${transfer_code}"
EOF
)"

  wait "${SENDER_PID}"
  SENDER_PID=""
  rm -f "${SEND_LOG}"
  SEND_LOG=""

  echo "==> Verifying remote ${label}"
  run_remote_script "$(cat <<EOF
remote_verify_path=$(printf '%q' "${remote_verify_path}")
verify_type=$(printf '%q' "${verify_type}")

if [[ "\${verify_type}" == "dir" ]]; then
  [[ -d "\${remote_verify_path}" ]] || {
    echo "Remote directory missing: \${remote_verify_path}" >&2
    exit 1
  }
else
  [[ -f "\${remote_verify_path}" ]] || {
    echo "Remote file missing: \${remote_verify_path}" >&2
    exit 1
  }
fi

du -sh "\${remote_verify_path}"
EOF
)"
}

require_command git
require_command runpodctl
require_command ssh
require_command tar

if [[ ! -d "${RUNPOD_LOCAL_DATASET_DIR}" ]]; then
  echo "Local dataset directory not found: ${RUNPOD_LOCAL_DATASET_DIR}" >&2
  exit 1
fi

REPO_TRANSFER_MODE="$(resolve_repo_transfer_mode)"
TRANSFER_CHECKPOINT=false
if should_transfer_checkpoint; then
  TRANSFER_CHECKPOINT=true
  if [[ ! -f "${RUNPOD_LOCAL_CHECKPOINT_PATH}" ]]; then
    echo "Local checkpoint file not found: ${RUNPOD_LOCAL_CHECKPOINT_PATH}" >&2
    exit 1
  fi
fi

LOCAL_REPO_URL="$(resolve_repo_url || true)"
if [[ -z "${LOCAL_REPO_URL}" ]] && [[ "${REPO_TRANSFER_MODE}" == "git" ]]; then
  echo "Unable to determine the GitHub remote URL from this checkout." >&2
  exit 1
fi

AUTH_REPO_URL="$(inject_github_token "${LOCAL_REPO_URL}" "${GITHUB_TOKEN}")"

if [[ "${REPO_TRANSFER_MODE}" == "git" ]] && [[ "${AUTH_REPO_URL}" == "${LOCAL_REPO_URL}" ]] && [[ ! -n "${GITHUB_TOKEN}" ]]; then
  echo "Using repo URL '${LOCAL_REPO_URL}' without an injected GitHub token."
  echo "Fresh Runpod instances may fail to clone if the repository is private."
fi

SENDER_PID=""
SEND_LOG=""
REPO_SNAPSHOT_PATH=""
cleanup() {
  if [[ -n "${SENDER_PID}" ]] && kill -0 "${SENDER_PID}" >/dev/null 2>&1; then
    kill "${SENDER_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${SEND_LOG}" ]]; then
    rm -f "${SEND_LOG}"
  fi
  if [[ -n "${REPO_SNAPSHOT_PATH}" ]]; then
    rm -f "${REPO_SNAPSHOT_PATH}"
  fi
}
trap cleanup EXIT

if [[ "${REPO_TRANSFER_MODE}" == "git" ]]; then
  echo "==> Preparing repository checkout on ${RUNPOD_HOST}:${RUNPOD_REMOTE_DIR} via git"
  REMOTE_REPO_SCRIPT="$(cat <<EOF
remote_dir=$(printf '%q' "${RUNPOD_REMOTE_DIR}")
repo_url=$(printf '%q' "${LOCAL_REPO_URL}")
auth_repo_url=$(printf '%q' "${AUTH_REPO_URL}")
repo_branch=$(printf '%q' "${RUNPOD_REPO_BRANCH}")

mkdir -p "\$(dirname -- "\${remote_dir}")"
if [[ -d "\${remote_dir}/.git" ]]; then
  cd "\${remote_dir}"
  if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "\${auth_repo_url}"
  else
    git remote add origin "\${auth_repo_url}"
  fi
  git fetch origin "\${repo_branch}"
  if git show-ref --verify --quiet "refs/heads/\${repo_branch}"; then
    git checkout "\${repo_branch}"
  else
    git checkout -B "\${repo_branch}" "origin/\${repo_branch}"
  fi
  git pull --ff-only origin "\${repo_branch}"
else
  rm -rf "\${remote_dir}"
  git clone -b "\${repo_branch}" "\${auth_repo_url}" "\${remote_dir}"
fi
cd "\${remote_dir}"
git remote set-url origin "\${repo_url}"
EOF
)"
  run_remote_script "${REMOTE_REPO_SCRIPT}"
else
  echo "==> Preparing repository snapshot from local checkout"
  REPO_SNAPSHOT_PATH="$(create_repo_snapshot)"
  transfer_with_runpodctl \
    "${REPO_SNAPSHOT_PATH}" \
    "/tmp" \
    "$(basename "${REPO_SNAPSHOT_PATH}")" \
    "/tmp/$(basename "${REPO_SNAPSHOT_PATH}")" \
    "file" \
    "repository snapshot"
  echo "==> Extracting repository snapshot on ${RUNPOD_HOST}:${RUNPOD_REMOTE_DIR}"
  run_remote_script "$(cat <<EOF
remote_dir=$(printf '%q' "${RUNPOD_REMOTE_DIR}")
snapshot_path=$(printf '%q' "/tmp/$(basename "${REPO_SNAPSHOT_PATH}")")

rm -rf "\${remote_dir}"
mkdir -p "\${remote_dir}"
tar -xzf "\${snapshot_path}" -C "\${remote_dir}"
rm -f "\${snapshot_path}"
[[ -f "\${remote_dir}/scripts/benchmark/run.py" ]] || {
  echo "Repository snapshot extraction failed at \${remote_dir}" >&2
  exit 1
}
EOF
)"
fi

if [[ "${REPO_TRANSFER_MODE}" == "git" ]] && [[ -n "${RUNPOD_BENCHMARK_SUPPORT_PATHS}" ]]; then
  OLD_IFS="${IFS}"
  IFS=','
  read -r -a SUPPORT_PATHS <<< "${RUNPOD_BENCHMARK_SUPPORT_PATHS}"
  IFS="${OLD_IFS}"
  for relative_path in "${SUPPORT_PATHS[@]}"; do
    relative_path="${relative_path#"${relative_path%%[![:space:]]*}"}"
    relative_path="${relative_path%"${relative_path##*[![:space:]]}"}"
    [[ -n "${relative_path}" ]] || continue
    local_path="${REPO_ROOT}/${relative_path}"
    if [[ -d "${local_path}" ]]; then
      parent_rel="$(dirname -- "${relative_path}")"
      remote_parent="${RUNPOD_REMOTE_DIR}"
      if [[ "${parent_rel}" != "." ]]; then
        remote_parent="${RUNPOD_REMOTE_DIR}/${parent_rel}"
      fi
      transfer_with_runpodctl \
        "${local_path}" \
        "${remote_parent}" \
        "$(basename -- "${relative_path}")" \
        "${RUNPOD_REMOTE_DIR}/${relative_path}" \
        "dir" \
        "support path ${relative_path}"
    elif [[ -f "${local_path}" ]]; then
      parent_rel="$(dirname -- "${relative_path}")"
      remote_parent="${RUNPOD_REMOTE_DIR}"
      if [[ "${parent_rel}" != "." ]]; then
        remote_parent="${RUNPOD_REMOTE_DIR}/${parent_rel}"
      fi
      transfer_with_runpodctl \
        "${local_path}" \
        "${remote_parent}" \
        "$(basename -- "${relative_path}")" \
        "${RUNPOD_REMOTE_DIR}/${relative_path}" \
        "file" \
        "support path ${relative_path}"
    else
      echo "Skipping missing support path: ${relative_path}" >&2
    fi
  done
fi

transfer_with_runpodctl \
  "${RUNPOD_LOCAL_DATASET_DIR}" \
  "${RUNPOD_REMOTE_DATASETS_DIR}" \
  "${RUNPOD_DATASET_NAME}" \
  "${RUNPOD_REMOTE_DATASETS_DIR}/${RUNPOD_DATASET_NAME}" \
  "dir" \
  "dataset ${RUNPOD_DATASET_NAME}"

if [[ "${TRANSFER_CHECKPOINT}" == "true" ]]; then
  transfer_with_runpodctl \
    "${RUNPOD_LOCAL_CHECKPOINT_PATH}" \
    "${RUNPOD_REMOTE_WEIGHTS_DIR}" \
    "${RUNPOD_CHECKPOINT_NAME}" \
    "${RUNPOD_REMOTE_WEIGHTS_DIR}/${RUNPOD_CHECKPOINT_NAME}" \
    "file" \
    "checkpoint ${RUNPOD_CHECKPOINT_NAME}"
else
  echo "==> Skipping checkpoint transfer"
fi

echo "Runpod bootstrap complete."
