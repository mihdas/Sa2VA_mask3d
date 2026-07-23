#!/bin/bash
#SBATCH --partition=3090-lo
#SBATCH --job-name=mevis_inference
#SBATCH --output=logs/%A_%a.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --exclusive

EXPERIMENT_NAME=$1

# Load environment and modules
# source .venv/bin/activate
[[ -e .env ]] && source .env
export OMP_NUM_THREADS=4
export NUMBA_NUM_THREADS=4
export PET_NPROC_PER_NODE=1

export PYTHONPATH=$PYTHONPATH:$(pwd)

# Print paths for debugging
echo "Current Path: $(pwd)"

# Run the main processing
srun python demo/demo.py \
    /nodes/cristal/work/nekrasov/data/language-data/mevis/valid \
    --model_path ./work_dirs/$EXPERIMENT_NAME \
    --work-dir ./saved/$EXPERIMENT_NAME/val/masks/
