#!/bin/bash
#SBATCH --output=/home/vipradas/Thesis/slurm_%j.out
#SBATCH --mail-user=mihirgrad@gmail.com
#SBATCH --mail-type=BEGIN,FAIL,END

source /home/vipradas/Thesis/envsa2va/bin/activate
cd /home/vipradas/Thesis/Sa2VA_p
python bbox.py