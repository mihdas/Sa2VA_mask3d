#!/bin/bash
#SBATCH --cpus-per-task=48
#SBATCH --mem=450G
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --account=llmvidseg
#SBATCH --exclusive
#SBATCH --job-name=basic
#SBATCH --time=0-04:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --partition=booster
#SBATCH --begin=now
#SBATCH --mail-user=mihirvipradas@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=1-10%1  # Run tasks sequentially
#SBATCH --output=logs/slurm_logs/%j_%n_%x_%a_out.txt

set -eo pipefail
set -x

echo "START TIME: $(date)"

PROGRAM="$1"
echo "Training program: $PROGRAM"

# Check if VENV is empty
if [ -z "$VENV" ]; then
  echo "Error: VENV is not set. Please define it in your environment."
  exit 1
fi

echo $VENV
source $VENV
echo $PYTHONPATH

echo $TMPDIR
echo $TRITON_CACHE_DIR

# Print code version and changes
echo "CODE FOR COMMIT: $(git rev-parse HEAD)"
git diff HEAD | cat

# Get host address for distributed training
PARSED_HOSTLIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)
DDP_HOST_ADDRESS=$(echo $PARSED_HOSTLIST | awk '{print $1}' | sed 's/$/i/')
echo "DDP_HOST_ADDRESS: $DDP_HOST_ADDRESS"
echo "PARSED_HOSTLIST: $(echo "$PARSED_HOSTLIST" | tr '\n' ' ')"

# Distributed training configuration
GPUS_PER_NODE=4
NNODES=$SLURM_NNODES
MASTER_PORT=29500

# Environment variables for offline mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Threading settings
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Prevent NCCL not figuring out how to initialize.
export NCCL_SOCKET_IFNAME=ib0
# Prevent Gloo not being able to communicate.
export GLOO_SOCKET_IFNAME=ib0

# EDIT: useful for debug if needed
# to debug NCCL issues
# export NCCL_DEBUG=INFO
# to unravel async errors w/o the correct traceback - potentially makes everything very slower
# export CUDA_LAUNCH_BLOCKING=1
# to force crashing on nccl issues like hanging broadcast
# export NCCL_ASYNC_ERROR_HANDLING=1

# Distributed launcher command
LAUNCHER="python -m torchrun_jsc \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_endpoint $DDP_HOST_ADDRESS:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role \$(hostname -s | tr -dc '0-9'): \
    --tee 3 \
    "

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --hint=nomultithread \
    --cpus-per-task=$SLURM_CPUS_PER_TASK \
    --jobid $SLURM_JOB_ID \
    "

# Launch the training job
srun $SRUN_ARGS bash -c "$LAUNCHER $PROGRAM" 2>&1

echo "END TIME: $(date)"

