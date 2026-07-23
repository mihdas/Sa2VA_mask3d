
import argparse
import os, sys
from tqdm import tqdm

from SensorData import SensorData



def main(filename, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sd = SensorData(filename)
    sd.export_color_images(os.path.join(output_path, 'color'))
    sd.export_depth_images(os.path.join(output_path, 'depth'))
    sd.export_intrinsics(os.path.join(output_path, 'intrinsics'))
    sd.export_poses(os.path.join(output_path, 'pose'))


if __name__ == '__main__':
    scenes=[]
    with open('raw_jsons/ScanNet.txt', 'r') as file:
        for line in file:
            scenes.append(line.strip())
    # idx = 0
    for scene in tqdm(scenes):
        main(f"/globalwork/datasets/scannet/scannet/scans/{scene}/{scene}.sens", f"/globalwork/vipradas/scannet_images/{scene}")
       

