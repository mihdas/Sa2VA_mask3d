#!/bin/bash
# Ensure the script fails on any error
set -euo pipefail

# Get the name of the experiment from the script name
# EXPERIMENT_NAME=$(basename "$0" .bash)
# EXPERIMENT_NAME=$1
# export EXPERIMENT_NOTES=$EXPERIMENT_NAME
EXPERIMENT_NAME=$(basename "$(dirname "$(pwd)")")

HF_EXPERIMENT_NAME="${EXPERIMENT_NAME}_hf"  # Corrected HF_EXPERIMENT_NAME

# Finding the log folder
# LOG_FOLDER="$HOME/github/Sa2VA/work_dirs/$EXPERIMENT_NAME"
LOG_FOLDER="$(dirname "$(pwd)")"
if [ ! -d "$LOG_FOLDER" ]; then
    echo "Log folder does not exist: $LOG_FOLDER"
    exit 1
fi

# Change directory to the code directory (inside the log folder)
export CODE_FOLDER="$(pwd)"
# export CODE_FOLDER="${HOME}/github/Sa2VA"
# if [ ! -d "$CODE_FOLDER" ]; then
#     echo "Code folder does not exist: $CODE_FOLDER"
#     exit 1
# fi
# cd "$CODE_FOLDER"

# Define the HF save path
HF_SAVE_PATH="${LOG_FOLDER}/${HF_EXPERIMENT_NAME}"

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
IMAGE_EVAL_COMMANDS=(
  "projects/llava_sam2/evaluation/refcoco_eval.py --launcher pytorch --data_path ${DATA_PATH} --dataset refcocog ${HF_SAVE_PATH} --split val"
#  "projects/llava_sam2/evaluation/refcoco_eval.py --launcher pytorch --data_path ${DATA_PATH} --dataset refcocog ${HF_SAVE_PATH} --split test"
#  "projects/llava_sam2/evaluation/refcoco_eval.py --launcher pytorch --data_path ${DATA_PATH} --dataset refcoco ${HF_SAVE_PATH} --split val"
#  "projects/llava_sam2/evaluation/refcoco_eval.py --launcher pytorch --data_path ${DATA_PATH} --dataset refcoco ${HF_SAVE_PATH} --split testA"
#  "projects/llava_sam2/evaluation/refcoco_eval.py --launcher pytorch --data_path ${DATA_PATH} --dataset refcoco ${HF_SAVE_PATH} --split testB"
#  "projects/llava_sam2/evaluation/refcoco_eval.py --launcher pytorch --data_path ${DATA_PATH} --dataset refcoco_plus ${HF_SAVE_PATH} --split val"
#  "projects/llava_sam2/evaluation/refcoco_eval.py --launcher pytorch --data_path ${DATA_PATH} --dataset refcoco_plus ${HF_SAVE_PATH} --split testA"
#  "projects/llava_sam2/evaluation/refcoco_eval.py --launcher pytorch --data_path ${DATA_PATH} --dataset refcoco_plus ${HF_SAVE_PATH} --split testB"
)
IMAGE_DATASET_NAMES=(
  "refcocog_val"
#  "refcocog_test"
#  "refcoco_val"
#  "refcoco_testA"
#  "refcoco_testB"
#  "refcoco_plus_val"
#  "refcoco_plus_testA"
#  "refcoco_plus_testB"
)

VIDEO_EVAL_COMMANDS=(
#  "projects/llava_sam2/evaluation/ref_vos_eval.py --launcher pytorch --data_path ${DATA_PATH} --dataset MEVIS_U --submit --save_path ${LOG_FOLDER}/video_prediction/mevis_u/prediction ${HF_SAVE_PATH}"
#  "projects/llava_sam2/evaluation/ref_vos_eval.py --launcher pytorch --data_path ${DATA_PATH} --dataset MEVIS --submit --save_path ${LOG_FOLDER}/video_prediction/mevis/prediction ${HF_SAVE_PATH}"
  "projects/llava_sam2/evaluation/ref_vos_eval.py --launcher pytorch --data_path ${DATA_PATH} --dataset DAVIS --submit --save_path ${LOG_FOLDER}/video_prediction/davis17/prediction ${HF_SAVE_PATH}"
#  "projects/llava_sam2/evaluation/ref_vos_eval.py --launcher pytorch --data_path ${DATA_PATH} --dataset REVOS --submit --save_path ${LOG_FOLDER}/video_prediction/revos/prediction ${HF_SAVE_PATH}"
#  "projects/llava_sam2/evaluation/ref_vos_eval.py --launcher pytorch --data_path ${DATA_PATH} --dataset REFYTVOS --submit --save_path ${LOG_FOLDER}/video_prediction/youtube-vos/prediction ${HF_SAVE_PATH}"
)

VIDEO_POSTEVAL_COMMANDS=(
#  "video_postprocess.py --base_video_path ${DATA_PATH}/mevis/valid_u/ --prediction_path ${LOG_FOLDER}/video_prediction/mevis_u/prediction --output_path ${LOG_FOLDER}/video_prediction/mevis_u/xmem_prediction"
#  "video_postprocess.py --base_video_path ${DATA_PATH}/mevis/valid/ --prediction_path ${LOG_FOLDER}/video_prediction/mevis/prediction --output_path ${LOG_FOLDER}/video_prediction/mevis/xmem_prediction"
  "video_postprocess.py --base_video_path ${DATA_PATH}/davis17/valid/ --prediction_path ${LOG_FOLDER}/video_prediction/davis17/prediction --output_path ${LOG_FOLDER}/video_prediction/davis17/xmem_prediction"
#  "video_postprocess.py --base_video_path ${DATA_PATH}/revos/REVOS/ --prediction_path ${LOG_FOLDER}/video_prediction/revos/prediction --output_path ${LOG_FOLDER}/video_prediction/revos/xmem_prediction"
#  "video_postprocess.py --base_video_path ${DATA_PATH}/youtube-vos/valid/ --prediction_path ${LOG_FOLDER}/video_prediction/youtube-vos/prediction --output_path ${LOG_FOLDER}/video_prediction/youtube-vos/Annotation"
)
VIDEO_DATASET_NAMES=(
#  "mevis_u"
#  "mevis"
  "davis"
#  "revos"
#  "refytvos"
)

# FINAL_VIDEO_EVAL_COMMANDS=(
#   "tools/eval_davis17.py --mevis_exp_path ${DATA_PATH}/davis17/meta_expressions/valid/meta_expressions.json --mevis_mask_path ${DATA_PATH}/davis17/valid/mask_dict.pkl --save_name davis.json ${LOG_FOLDER}/video_prediction/davis17/xmem_prediction"
#   "tools/eval_revos.py --visa_exp_path ${DATA_PATH}/revos/REVOS/meta_expressions_valid_.json --visa_mask_path ${DATA_PATH}/revos/REVOS/mask_dict.json --visa_foreground_mask_path ${DATA_PATH}/revos/ReVOS/mask_dict_foreground.json --save_json_name revos_valid.json --save_csv_name revos_valid.csv ${LOG_FOLDER}/video_prediction/revos/xmem_prediction"
# )

JOB_NAME_PREFIX="eval"
# Process image evaluation commands
for i in $(seq 0 $(( ${#IMAGE_EVAL_COMMANDS[@]} - 1 ))); do
  command="${IMAGE_EVAL_COMMANDS[$i]}"
  dataset_name="${IMAGE_DATASET_NAMES[$i]}"

  sbatch --chdir="$CODE_FOLDER" \
      --nodes=1 \
      --time=0-04:00:00 \
      --array="0" \
      --job-name="${JOB_NAME_PREFIX}_${dataset_name}" \
      --output="$LOG_FOLDER/slurm-eval-${dataset_name}-%j.out" \
      ${DEPENDENCY:+--dependency=$DEPENDENCY} \
      "$HOME/github/vision-utils/scripts/juelich_run_v3.bash" "$command"
  echo "Image evaluation job ${dataset_name} submitted"
done

export VENV="$HOME/github/Sa2VA/.env"
# Process video evaluation commands with post-processing
for i in $(seq 0 $(( ${#VIDEO_EVAL_COMMANDS[@]} - 1 ))); do
  command="${VIDEO_EVAL_COMMANDS[$i]}"
  post_eval_command="${VIDEO_POSTEVAL_COMMANDS[$i]}"
  dataset_name="${VIDEO_DATASET_NAMES[$i]}"

  job_id=$(sbatch --chdir="${CODE_FOLDER}" \
      --nodes=2 \
      --time=0-02:00:00 \
      --array="0" \
      --job-name="${dataset_name}_${JOB_NAME_PREFIX}" \
      --output="$LOG_FOLDER/slurm-eval-${dataset_name}-%j.out" \
      ${DEPENDENCY:+--dependency=$DEPENDENCY} \
      "$HOME/github/vision-utils/scripts/juelich_run_v3.bash" "$command" | awk '{print $4}')

  sbatch --chdir="${PROJECT_llmvidseg}/alexey/code/XMem2" \
      --nodes=2 \
      --time=0-04:00:00 \
      --array="0" \
      --job-name="post_${dataset_name}_${JOB_NAME_PREFIX}" \
      --dependency=afterany:$job_id \
      --output="$LOG_FOLDER/post-${dataset_name}-%j.out" \
      --export=ALL,EXPERIMENT_PATH,DATA_PATH,save_path,CODE_FOLDER,dataset_index \
      "$HOME/github/vision-utils/scripts/juelich_run_v3.bash" "$post_eval_command"
  echo "Video evaluation job ${dataset_name} submitted with post-processing"
done

# Return to the original folder
cd -
