#!/usr/bin/env bash
set -euo pipefail

REPO_BRANCH="${REPO_BRANCH:-master}"
REPO_URL="${REPO_URL:-https://github.com/btrkeks/thesis-omr.git}"

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  REPO_URL="https://${GITHUB_TOKEN}@github.com/btrkeks/thesis-omr.git"
fi

git clone -b "${REPO_BRANCH}" "${REPO_URL}" omr
cd omr || exit
wandb login
