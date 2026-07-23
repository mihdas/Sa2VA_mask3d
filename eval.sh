#!/bin/bash
#SBATCH --output=/home/vipradas/Thesis/slurm_%j.out

source /home/vipradas/Thesis/envsa2va/bin/activate
python /home/vipradas/Thesis/Sa2VA_p/evaluate.py 

