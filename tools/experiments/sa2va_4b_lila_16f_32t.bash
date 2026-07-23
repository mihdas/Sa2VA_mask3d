#!/bin/bash

# Ensure the script fails on any error
set -euo pipefail

# Get the name of the experiment from the script name
# EXPERIMENT_NAME=$(basename "$0" .bash)
EXPERIMENT_NAME=sa2va_4b_lila_16f_32t
EXPERIMENT_NOTES=""

# Define the log folder and create it if it doesn't exist
LOG_FOLDER="$HOME/github/Sa2VA/work_dirs/$EXPERIMENT_NAME"
mkdir -p "$LOG_FOLDER/code"

# Move files not ignored by git to the log folder
rsync -arx --exclude-from=.gitignore ./ "$LOG_FOLDER/code"

# Change directory to the log folder
cd "$LOG_FOLDER/code"

# Define the program with all the parameters
export PROGRAM="tools/train.py projects/llava_sam2/configs/${EXPERIMENT_NAME}.py --launcher pytorch --deepspeed deepspeed_zero2"

# Define the module and virtual environment activation commands
export MODULE_CMD="module load Stages/2024 GCCcore/.12.3.0 Python Rust CUDA PyTorch Pillow-SIMD matplotlib scikit-learn scikit-image numba imageio git CMake torchvision ImageMagick SciPy-bundle GCC"
export VENV="source /p/project1/llmvidseg/alexey/code/Sa2VA/.venv/bin/activate"
export TMPDIR=/p/scratch/llmvidseg/alexey/tmp
export TRITON_CACHE_DIR=/p/scratch/llmvidseg/alexey/tmp/triton

export PYTHONPATH="${LOG_FOLDER}/code${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="/p/project1/llmvidseg/alexey/code/Sa2VA/.venv/lib/python3.11/site-packages:${PYTHONPATH}"

# Execute sbatch command
sbatch --chdir="$LOG_FOLDER/code" --job-name="$EXPERIMENT_NAME" --output="$LOG_FOLDER/slurm-%j.out" "$HOME/github/vision-utils/scripts/juelich_run_v2.bash"

# Return to the original folder
cd -
