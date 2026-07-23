#!/bin/bash

# Ensure the script fails on any error
set -euo pipefail

# Define the program with all the parameters
PROGRAM=$@

# Define the module and virtual environment activation commands
export VENV="$HOME/github/Sa2VA/.env"

# Execute sbatch command
sbatch --nodes=2 --time="0-01:00:00" --job-name="inference" --output="logs/misc/%j_%n_%x_%a_.out" "$HOME/github/vision-utils/scripts/juelich_run_v3.bash" "$PROGRAM"

# Return to the original folder
cd -
