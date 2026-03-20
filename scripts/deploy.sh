#!/bin/bash
set -e

SERVER="tujestpolin"
REMOTE_PATH="~/thesis-omr"
DATASET_NAME="${1:-augmented_dataset_50k_new}"

echo "==> Syncing $DATASET_NAME to server..."
rsync -avz --progress "data/$DATASET_NAME/" "$SERVER:$REMOTE_PATH/data/$DATASET_NAME/"

echo "==> Committing and pushing..."
git add -A
git commit -m "Update dataset: $DATASET_NAME" || echo "Nothing to commit"
git push

echo "==> Pulling on server..."
ssh "$SERVER" "cd $REMOTE_PATH && git pull"

echo "==> Done!"
