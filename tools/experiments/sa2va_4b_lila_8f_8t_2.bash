#!/bin/bash

# Ensure the script fails on any error
set -euo pipefail

# Get the name of the experiment from the script name
EXPERIMENT_NAME=$(basename "$0" .bash)
# EXPERIMENT_NAME=sa2va_4b_lila_8f_8t_2

# Define the log folder and create it if it doesn't exist
LOG_FOLDER="$HOME/github/Sa2VA/work_dirs/$EXPERIMENT_NAME"
CODE_FOLDER=$LOG_FOLDER/code
mkdir -p "$CODE_FOLDER"

# Move files not ignored by git to the log folder
rsync -arx --exclude-from=.gitignore ./ "$CODE_FOLDER"

# Change directory to the log folder
cd "$CODE_FOLDER"

# Define the program with all the parameters
PROGRAM="tools/train.py projects/llava_sam2/configs/${EXPERIMENT_NAME}.py --launcher pytorch --deepspeed deepspeed_zero2"

# Define the module and virtual environment activation commands
export VENV="$HOME/github/Sa2VA/.env"

export TMPDIR=/p/scratch/llmvidseg/alexey/tmp
export TRITON_CACHE_DIR=/p/scratch/llmvidseg/alexey/tmp/triton

# Execute sbatch command
sbatch --chdir="$CODE_FOLDER" --job-name="$EXPERIMENT_NAME" --output="$LOG_FOLDER/train-%j_%n_%x_%a_.out" "$HOME/github/vision-utils/scripts/juelich_run_v3.bash" "$PROGRAM"

# Return to the original folder
cd -
