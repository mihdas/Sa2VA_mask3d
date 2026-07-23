#!/bin/bash
# Ensure the script fails on any error
set -euo pipefail

# Get the name of the experiment from the script name
EXPERIMENT_NAME=$1

# Finding the log folder
LOG_FOLDER="$HOME/github/Sa2VA/work_dirs/$EXPERIMENT_NAME"

# Create video prediction folder if it doesn't exist
VIDEO_PREDICTION_FOLDER="${LOG_FOLDER}/video_prediction"
mkdir -p "$VIDEO_PREDICTION_FOLDER"  # Create it if it doesn't exist

# Logging function
log() {
  timestamp=$(date +%Y-%m-%d_%H:%M:%S)
  echo "[$timestamp] $@"
}

if [ ! -d "$LOG_FOLDER" ]; then
    log "Log folder does not exist: $LOG_FOLDER"
    exit 1
fi

# Change directory to the code directory (inside the log folder)
# CODE_FOLDER="$LOG_FOLDER/code"

# if [ ! -d "$CODE_FOLDER" ]; then
#     log "Code folder does not exist: $CODE_FOLDER"
#     exit 1
# fi
#
# cd "$CODE_FOLDER"

DATA_PATH="${SCRATCH_llmvidseg}/alexey/data/language-data"

# Function to encapsulate each task
run_mevis_verification_zipping() {
    log "Starting MeVis verification and zipping..."
    MEVIS_PREDICTION_DIR="${LOG_FOLDER}/video_prediction/mevis/xmem_prediction"
    MEVIS_ZIP_FILE="${LOG_FOLDER}/video_prediction/${EXPERIMENT_NAME}_mevis.zip"
    MEVIS_ANNOTATION_FILE="${DATA_PATH}/mevis/valid/meta_expressions.json"

    log "Verifying MeVis submission..."
    python tools/verify_submission.py --prediction_dir "${MEVIS_PREDICTION_DIR}" --annotation_file "${MEVIS_ANNOTATION_FILE}"  | tee "${VIDEO_PREDICTION_FOLDER}/mevis_verify.log"

    log "Zipping MeVis predictions..."
    pushd "${MEVIS_PREDICTION_DIR}" > /dev/null
    zip -r "${MEVIS_ZIP_FILE}" . | tee "${VIDEO_PREDICTION_FOLDER}/mevis_zip.log"
    popd > /dev/null
}

run_ytvos_verification_zipping() {
    log "Starting YouTube-VOS verification and zipping..."
    YTVOS_PREDICTION_DIR="${LOG_FOLDER}/video_prediction/youtube-vos/Annotations"
    YTVOS_ZIP_FILE="${LOG_FOLDER}/video_prediction/${EXPERIMENT_NAME}_ytvos.zip"
    YTVOS_ANNOTATION_FILE="${DATA_PATH}/youtube-vos/valid/meta.json"

    log "Verifying YouTube-VOS submission..."
    python tools/verify_submission.py --prediction_dir "${YTVOS_PREDICTION_DIR}" --annotation_file "${YTVOS_ANNOTATION_FILE}" | tee "${VIDEO_PREDICTION_FOLDER}/ytvos_verify.log"

    log "Zipping YouTube-VOS predictions..."
    pushd "${YTVOS_PREDICTION_DIR}/.." > /dev/null
    zip -r "${YTVOS_ZIP_FILE}" ./Annotations | tee "${VIDEO_PREDICTION_FOLDER}/ytvos_zip.log"
    popd > /dev/null
}

run_davis17_evaluation() {
    log "Starting DAVIS-17 evaluation..."
    DAVIS17_MEVIS_EXP_PATH="${DATA_PATH}/davis17/meta_expressions/valid/meta_expressions.json"
    DAVIS17_MEVIS_MASK_PATH="${DATA_PATH}/davis17/valid/mask_dict.pkl"
    DAVIS17_PREDICTION_DIR="${LOG_FOLDER}/video_prediction/davis17/xmem_prediction"
    DAVIS17_SAVE_NAME="${LOG_FOLDER}/video_prediction/davis.json"

    log "Evaluating DAVIS-17..."
    python tools/eval_davis17.py --mevis_exp_path "${DAVIS17_MEVIS_EXP_PATH}" --mevis_mask_path "${DAVIS17_MEVIS_MASK_PATH}" --save_name "${DAVIS17_SAVE_NAME}" "${DAVIS17_PREDICTION_DIR}" | tee "${VIDEO_PREDICTION_FOLDER}/davis17_eval.log"
}

run_mevisu_evaluation() {
    log "Starting MeViS-U evaluation..."
    MEVISU_PATH="${DATA_PATH}/mevis/valid_u"
    MEVISU_PREDICTION_PATH="${LOG_FOLDER}/video_prediction/mevis_u/xmem_prediction"

    log "Evaluating MeViS-U..."
    python tools/eval_mevis.py --mevis_path "${MEVISU_PATH}" --prediction_path "${MEVISU_PREDICTION_PATH}" | tee "${VIDEO_PREDICTION_FOLDER}/mevis_u_eval.log"
}

run_revos_evaluation() {
    log "Starting ReVOS evaluation..."
    REVOS_VISA_EXP_PATH="${DATA_PATH}/revos/REVOS/meta_expressions_valid_.json"
    REVOS_VISA_MASK_PATH="${DATA_PATH}/revos/REVOS/mask_dict.json"
    REVOS_VISA_FOREGROUND_MASK_PATH="${DATA_PATH}/revos/REVOS/mask_dict_foreground.json"
    REVOS_PREDICTION_DIR="${LOG_FOLDER}/video_prediction/revos/xmem_prediction"
    REVOS_SAVE_JSON_NAME="${LOG_FOLDER}/video_prediction/revos/revos_valid.json"
    REVOS_SAVE_CSV_NAME="${LOG_FOLDER}/video_prediction/revos/revos_valid.csv"

    log "Evaluating ReVOS..."
    python tools/eval_revos.py --visa_exp_path "${REVOS_VISA_EXP_PATH}" --visa_mask_path "${REVOS_VISA_MASK_PATH}" --visa_foreground_mask_path "${REVOS_VISA_FOREGROUND_MASK_PATH}" --save_json_name "${REVOS_SAVE_JSON_NAME}" --save_csv_name "${REVOS_SAVE_CSV_NAME}" "${REVOS_PREDICTION_DIR}" | tee "${VIDEO_PREDICTION_FOLDER}/revos_eval.log"
}

# Run tasks in parallel using background processes
run_mevis_verification_zipping &
run_ytvos_verification_zipping &
run_davis17_evaluation &
run_mevisu_evaluation &
run_revos_evaluation &

# Wait for all background processes to finish
wait

log "All tasks completed."
