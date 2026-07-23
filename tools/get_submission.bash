#!/usr/bin/env bash

# Check arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 EXP_NAME"
    exit 1
fi

EXP_NAME=$1
ORIGINAL_DIR=$(pwd)

# Define or source these paths as needed
export WS_SAVE_PATH="/nodes/cristal/fastwork/nekrasov/saved/sa2va_lila"

# Make sure the local workspace directories exist
mkdir -p "${WS_SAVE_PATH}/${EXP_NAME}/val"

# 1. Check if the zip file already exists on the server
# 2. Use rsync to copy the zip file to the workspace
# 3. Unzip in the workspace

############################################
# Functions for generation and verification
############################################

generate_masks() {
    # Submit mask generation job
    # Assume generate_masks_xmempp.bash is located in $HOME/github/XMem2
    cd "$HOME/github/XMem2" || { echo "Failed to cd into XMem2 directory"; exit 1; }
    echo "Submitting generate_masks_xmempp job..."
    gjobid=$(sbatch generate_masks_xmempp.bash "${WS_SAVE_PATH}/${EXP_NAME}/val" | awk '{print $4}')
    if [ -z "$gjobid" ]; then
        echo "Failed to submit generate_masks_xmempp job."
        exit 1
    fi
    echo "Submitted generate_masks_xmempp job with ID: $gjobid"

    # Wait for the generation job to complete
    echo "Waiting for generate_masks_xmempp job [$gjobid] to complete..."
    while [ "$(squeue -j "$gjobid" | wc -l)" -gt 1 ]; do
        sleep 30
    done
    echo "Generation job completed."
}

verify_masks() {
    # Run verification script locally
    # Assuming verify_submission.py is located in $HOME/github/lila/tools/
    cd "$HOME/github/Sa2VA" || { echo "Failed to cd into sa2va directory"; exit 1; }
    echo "Verifying submission..."
    python tools/verify_submission.py \
        --prediction_dir "${WS_SAVE_PATH}/${EXP_NAME}/val/xmem_prediction" \
        --annotation_file /nodes/cristal/work/nekrasov/data/language-data/mevis/valid/meta_expressions.json
    return $?  # Return the exit code of the verification
}

zip_results() {
    echo "Zipping xmem_prediction directory..."
    # Note: Removing the extra ${EXP_NAME} in the path before xmem_prediction
    cd "${WS_SAVE_PATH}/${EXP_NAME}/val/xmem_prediction"
    zip -r "${WS_SAVE_PATH}/${EXP_NAME}/val/${EXP_NAME}_xmem_prediction.zip" "./"
    if [ $? -ne 0 ]; then
        echo "Failed to zip xmem_prediction directory."
        exit 1
    fi
}

######################################
# Main logic with retry if verification fails
######################################

# First generation attempt
generate_masks

# First verification attempt
verify_masks
if [ $? -ne 0 ]; then
    echo "Verification failed. Re-submitting the generation job..."

    # (Optional) You might want to remove or rename the old prediction directory
    # rm -rf "${WS_SAVE_PATH}/${EXP_NAME}/val/xmem_prediction"

    # Second generation attempt
    generate_masks

    # Second verification attempt
    verify_masks
    if [ $? -ne 0 ]; then
        echo "Verification failed again. Exiting."
        exit 1
    fi
fi

# If we reach here, verification succeeded
echo "Verification succeeded."

# Zip results
zip_results

# Return to original directory
cd "$ORIGINAL_DIR"
echo "All steps completed successfully."
