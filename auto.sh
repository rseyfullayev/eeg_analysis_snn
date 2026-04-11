#!/bin/bash

# --- Configuration ---
REPO_DIR="/home/temporaryuser3/Documents/ERO/"
BRANCH="features/odconv" # Change this if you use 'master' or another branch
SCRIPT_TO_RUN="python main.py --config configs/config.yaml --loso 1" # Replace with your actual command
TEST_SCRIPT="python main.py --config configs/config.yaml --loso 1" # Replace with your actual command
POLL_INTERVAL=60 # Checks every 60 seconds
# ---------------------

cd "$REPO_DIR" || exit

echo "Starting auto-runner. Monitoring branch: $BRANCH..."

while true; do
    git fetch origin > /dev/null 2>&1
    git reset --hard origin/$BRANCH > /dev/null 2>&1

    CURRENT_COMMIT=$(git rev-parse HEAD)

    # --- BLOCK 1: Handle START ---
    if [ -f "START" ]; then
        if [ ! -f ".last_run_commit" ] || [ "$(cat .last_run_commit)" != "$CURRENT_COMMIT" ]; then
            echo "$(date): New START file detected on commit $CURRENT_COMMIT."
            echo "$CURRENT_COMMIT" > .last_run_commit
            
            eval $SCRIPT_TO_RUN > run_log.txt 2>&1
            echo "$(date): Run finished."
        fi
    fi

    # --- BLOCK 2: Handle TEST ---
    if [ -f "TEST" ]; then
        if [ ! -f ".last_test_commit" ] || [ "$(cat .last_test_commit)" != "$CURRENT_COMMIT" ]; then
            echo "$(date): New TEST file detected on commit $CURRENT_COMMIT."
            echo "$CURRENT_COMMIT" > .last_test_commit
            
            echo "Executing test script..."
            eval $TEST_SCRIPT > test_log.txt 2>&1
            
            echo "Pushing PNG results back to repository..."
            # Stage any newly generated PNG files
            git add *.png
            
            # Commit and push. The || true prevents the script from crashing if no PNG was made
            git commit -m "Auto-generated test plot from lab computer" || true
            git push origin $BRANCH || true
            
            echo "$(date): Test finished and pushed."
        fi
    fi

    sleep $POLL_INTERVAL
done