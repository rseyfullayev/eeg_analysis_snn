#!/bin/bash

# ============================================================
# CI/CD Auto-Runner: Monitors git for START/TEST trigger files
#
# START file format (one per line):
#   config/config.yaml
#   --phase 1a --loso 1
#
# TEST file format (one per line):
#   config/config.yaml
#   phase1a/MobileNet_20260410/checkpoint_best.pt
#   --loso 1
# ============================================================

# --- FILL THESE IN ---
VENV_PATH="/home/temporaryuser3/Documents/ERO/eeg_analysis_snn/venv/bin/activate"
REPO_DIR="/home/temporaryuser3/Documents/ERO/eeg_analysis_snn/"
BRANCH="features/odconv"
TEST_CHECKPOINT_BASE="/media/temporaryuser3/STORAGE/ERO/saved_models/"               # e.g. "/home/temporaryuser3/checkpoints"

POLL_INTERVAL=60
# ============================================================

source "$VENV_PATH"

cd "$REPO_DIR" || { echo "REPO_DIR not found: $REPO_DIR"; exit 1; }

echo "Starting auto-runner. Monitoring branch: $BRANCH..."

while true; do
    git fetch origin > /dev/null 2>&1
    git reset --hard origin/$BRANCH > /dev/null 2>&1

    CURRENT_COMMIT=$(git rev-parse HEAD)

    # --- BLOCK 1: Handle START (Training) ---
    if [ -f "START" ]; then
        if [ ! -f ".last_run_commit" ] || [ "$(cat .last_run_commit)" != "$CURRENT_COMMIT" ]; then
            echo "$(date): New START file detected on commit $CURRENT_COMMIT."
            echo "$CURRENT_COMMIT" > .last_run_commit

            # Read config and args from START file
            START_CONFIG=$(sed -n '1p' START)
            START_ARGS=$(sed -n '2p' START)

            echo "  Config: $START_CONFIG"
            echo "  Args:   $START_ARGS"

            python main.py --config "$START_CONFIG" $START_ARGS
            echo "$(date): Training finished."
        fi
    fi

    # --- BLOCK 2: Handle TEST (Evaluation) ---
    if [ -f "TEST" ]; then
        if [ ! -f ".last_test_commit" ] || [ "$(cat .last_test_commit)" != "$CURRENT_COMMIT" ]; then
            echo "$(date): New TEST file detected on commit $CURRENT_COMMIT."
            echo "$CURRENT_COMMIT" > .last_test_commit

            # Read config, checkpoint rel path, and args from TEST file
            TEST_CONFIG=$(sed -n '1p' TEST)
            TEST_CHECKPOINT_REL=$(sed -n '2p' TEST)
            TEST_ARGS=$(sed -n '3p' TEST)

            FULL_CHECKPOINT="$TEST_CHECKPOINT_BASE/$TEST_CHECKPOINT_REL"

            echo "  Config:     $TEST_CONFIG"
            echo "  Checkpoint: $FULL_CHECKPOINT"
            echo "  Args:       $TEST_ARGS"

            python main.py --config "$TEST_CONFIG" --test --checkpoint "$FULL_CHECKPOINT" $TEST_ARGS
            echo "$(date): Test finished."
        fi
    fi

    sleep $POLL_INTERVAL
done