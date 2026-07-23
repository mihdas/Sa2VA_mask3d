#!/bin/bash
#SBATCH --output=/home/vipradas/Thesis/slurm_%j.out

source /home/vipradas/Thesis/envsa2va/bin/activate
python /home/vipradas/Thesis/Sa2VA_p/generate_scanrefer_masks.py --model="4B" --flash=True --start=0    --end=7485
# python /home/vipradas/Thesis/Sa2VA_p/scannet_validate.py --model="4B" --flash=True --start=1190 --end=2377
# python /home/vipradas/Thesis/Sa2VA_p/scannet_validate.py --model="4B" --flash=True --start=2378 --end=3566
# python /home/vipradas/Thesis/Sa2VA_p/scannet_validate.py --model="4B" --flash=True --start=3567 --end=4755
# python /home/vipradas/Thesis/Sa2VA_p/scannet_validate.py --model="4B" --flash=True --start=4756 --end=5944
# python /home/vipradas/Thesis/Sa2VA_p/scannet_validate.py --model="4B" --flash=True --start=5945 --end=7131
# python /home/vipradas/Thesis/Sa2VA_p/scannet_validate.py --model="4B" --flash=True --start=7132 --end=8321
# python /home/vipradas/Thesis/Sa2VA_p/scannet_validate.py --model="4B" --flash=True --start=8322 --end=9508