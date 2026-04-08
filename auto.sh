#!/bin/bash

# --- Configuration ---
REPO_DIR="/path/to/your/git/repository"
BRANCH="main" # Change this if you use 'master' or another branch
SCRIPT_TO_RUN="python train_sweep_net.py" # Replace with your actual command
POLL_INTERVAL=60 # Checks every 60 seconds
# ---------------------

cd "$REPO_DIR" || exit

echo "Starting auto-runner. Monitoring branch: $BRANCH..."

while true; do
    # 1. Fetch the latest changes from the remote
    git fetch origin > /dev/null 2>&1
    
    # 2. Reset hard to ensure the lab PC perfectly matches your remote repository
    # This overwrites any local changes on the lab PC, preventing merge conflicts
    git reset --hard origin/$BRANCH > /dev/null 2>&1

    # 3. Get the latest commit hash
    CURRENT_COMMIT=$(git rev-parse HEAD)

    # 4. Check if the trigger file exists
    if [ -f "START" ]; then
        
        # 5. Check if we have already run the script for this specific commit
        if [ ! -f ".last_run_commit" ] || [ "$(cat .last_run_commit)" != "$CURRENT_COMMIT" ]; then
            echo "---------------------------------------------------"
            echo "$(date): New START file detected on commit $CURRENT_COMMIT."
            echo "Initiating run..."
            
            # Save the commit hash so it doesn't trigger again for this exact commit
            echo "$CURRENT_COMMIT" > .last_run_commit

            # 6. Execute your script and log the output
            # Using 'eval' so you can easily swap out the SCRIPT_TO_RUN variable
            eval $SCRIPT_TO_RUN > run_log.txt 2>&1
            
            echo "$(date): Run finished. Logs saved to run_log.txt."
            echo "---------------------------------------------------"
        fi
    fi

    # Wait before checking again
    sleep $POLL_INTERVAL
done