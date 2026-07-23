#!/bin/bash
# Ensure the script fails on any error
set -euo pipefail

# Get the name of the experiment from the script name
EXPERIMENT_NAME=$(basename "$0" .bash)
export EXPERIMENT_NOTES=$EXPERIMENT_NAME

HF_EXPERIMENT_NAME="${EXPERIMENT_NAME}_hf"  # Corrected HF_EXPERIMENT_NAME

# Finding the log folder
LOG_FOLDER="$HOME/github/Sa2VA/work_dirs/$EXPERIMENT_NAME"
if [ ! -d "$LOG_FOLDER" ]; then
    echo "Log folder does not exist: $LOG_FOLDER"
    exit 1
fi

# Change directory to the code directory (inside the log folder)
# CODE_FOLDER="$LOG_FOLDER/code"
export CODE_FOLDER="${HOME}/github/Sa2VA"
if [ ! -d "$CODE_FOLDER" ]; then
    echo "Code folder does not exist: $CODE_FOLDER"
    exit 1
fi
cd "$CODE_FOLDER"

# Define the HF save path
HF_SAVE_PATH="work_dirs/${HF_EXPERIMENT_NAME}"

# needed to save predictions
export VENV="$HOME/github/Sa2VA/.env"
export TMPDIR=/p/scratch/llmvidseg/alexey/tmp
export TRITON_CACHE_DIR=/p/scratch/llmvidseg/alexey/tmp/triton

DATA_PATH="${SCRATCH_llmvidseg}/alexey/data/language-data"

# **1. HF Conversion Job**
HF_CONVERSION_COMMAND="projects/llava_sam2/hf/convert_to_hf.py \
    work_dirs/${EXPERIMENT_NAME}/${EXPERIMENT_NAME}.py \
    --pth-model work_dirs/${EXPERIMENT_NAME}/iter_23686.pth \
    --save-path ${HF_SAVE_PATH} \
    --launcher pytorch"

if [ ! -d "$HF_SAVE_PATH" ]; then
    echo "HF folder does not exist: $HF_SAVE_PATH"
    # Capture the HF_JOB_ID correctly
    HF_JOB_ID=$(sbatch --chdir="$CODE_FOLDER" \
           --job-name="${EXPERIMENT_NAME}_convert" \
           --nodes=1 \
           --array="1-1%1" \
           --time=0-00:30:00 \
           --output="$LOG_FOLDER/slurm-convert-%j.out" \
           "$HOME/github/vision-utils/scripts/juelich_run_v3.bash" "${HF_CONVERSION_COMMAND}" | awk '{print $4}')
    echo "HF Conversion job submitted with Job ID: $HF_JOB_ID"
    DEPENDENCY="afterok:$HF_JOB_ID" # Set dependency on HF conversion
else
    echo "HF folder exists: $HF_SAVE_PATH"  # Corrected log message
    HF_JOB_ID=0  # Set to 0 because HF conversion is skipped
    echo "HF conversion skipped."
    DEPENDENCY="" # Remove the dependency
fi

# Check if HF_JOB_ID was successfully captured
if [ -z "$HF_JOB_ID" ]; then
  echo "Error: Failed to capture HF_JOB_ID. sbatch command may have failed."
  exit 1
fi

# **2. Evaluation Job Array**
EVAL_COMMANDS=(
  "projects/llava_sam2/evaluation/refcoco_eval.py --launcher pytorch --data_path ${DATA_PATH} --dataset refcocog ${HF_SAVE_PATH}"
  "projects/llava_sam2/evaluation/refcoco_eval.py --launcher pytorch --data_path ${DATA_PATH} --dataset refcoco_plus ${HF_SAVE_PATH}"
  "projects/llava_sam2/evaluation/refcoco_eval.py --launcher pytorch --data_path ${DATA_PATH} --dataset refcoco_plus ${HF_SAVE_PATH}"
)

# Loop through the evaluation commands and schedule a separate job for each
for i in "${!EVAL_COMMANDS[@]}"; do
  EVAL_COMMAND="${EVAL_COMMANDS[$i]}"
  JOB_NAME="$i"

  sbatch --chdir="$CODE_FOLDER" \
         --nodes=1 \
         --time=0-03:00:00 \
         --job-name="$JOB_NAME" \
         --array="0" \
         --output="$LOG_FOLDER/slurm-eval-$JOB_NAME-%j.out" \
         ${DEPENDENCY:+--dependency=$DEPENDENCY} \
         "$HOME/github/vision-utils/scripts/juelich_run_v3.bash" "${EVAL_COMMAND}"

  echo "Submitted evaluation job: $JOB_NAME with command: $EVAL_COMMAND"
done

# Return to the original folder
cd -
