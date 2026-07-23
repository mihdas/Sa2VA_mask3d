#!/bin/bash
#SBATCH --output=/home/vipradas/Thesis/slurm_%j.out
#SBATCH --mail-user=mihirgrad@gmail.com
#SBATCH --mail-type=END,FAIL


export CUDA_LAUNCH_BLOCKING=1
source /home/vipradas/Thesis/envsa2va/bin/activate
cd /home/vipradas/Thesis/Sa2VA_p
bash tools/dist.sh train projects/llava_sam2/configs/sa2va_8b.py 4 --work-dir /nodes/faxe/work/vipradas/feb


