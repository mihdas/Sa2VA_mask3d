import json
import zipfile
import os
import torch
from tqdm import tqdm
import numpy as np
from plyfile import PlyData

# if __name__ == '__main__':
#     #scenes=os.listdir("/globalwork/datasets/matterport/v1/scans/")
#     scenes=[]
#     with open('raw_jsons/Matterport.txt', 'r') as file:
#         for line in file:
#             scenes.append(line.strip())
#         file.close()
#     for scene in tqdm(scenes):
#         with zipfile.ZipFile(f"/globalwork/datasets/matterport/v1/scans/{scene}/matterport_color_images.zip", 'r') as zip_ref:
#             zip_ref.extractall(f"/globalwork/vipradas/matterport3d/")


# scenes=[]

    
# scenes=os.listdir("/globalwork/vipradas/scannet")
# images=os.listdir("/globalwork/vipradas/scannet_images")
# missing_scenes={}
# count=0
# with open("/home/vipradas/Thesis/Sa2VA_p/raw_jsons/sr3d_test.json") as file:
#     json_data=json.load(file)
#     for prompt in json_data:
#         count+=1
#         sceneid=prompt['scene_id']
#         if sceneid+".pth" not in scenes:
#             missing_scenes[sceneid]=1
#             #print(sceneid, end="")
#         if sceneid not in images:
#             print("images needed for ", sceneid)
#           #  scenes.append(f"/globalwork/datasets/scannetpp/data/{sceneid}/dslr/resized_images/")
#        # descriptions.append(prompt['description'])
            
#     file.close()
# print(count)
# print(len(missing_scenes.keys()))


# for scene_id in tqdm(missing_scenes.keys()):
#     with open(f'/globalwork/datasets/scannet/scannet/scans/{scene_id}/{scene_id}.aggregation.json') as f:
#         data = json.load(f)
#         f.close()

#     objects = data['segGroups']
#     segments_file=data["segmentsFile"][8:]

#     print(f"Loading {segments_file}")

#     with open(f"/globalwork/datasets/scannet/scannet/scans/{scene_id}/{segments_file}") as f:
#         data=json.load(f)
#         f.close()

#     mapping=np.array(data['segIndices'])
#     instanceids=np.zeros((len(mapping),),dtype=int)
#     print(f"Number of vertices: {len(mapping)}")

#     for i,object in enumerate(objects):
#         vertex_ids=np.isin(mapping,object['segments'])
#         instanceids[vertex_ids]=object['objectId']

#     plydata = PlyData.read(f'/globalwork/datasets/scannet/scannet/scans/{scene_id}/{scene_id}_vh_clean_2.ply')
#     vertices = np.vstack([plydata['vertex'].data['x'], plydata['vertex'].data['y'], plydata['vertex'].data['z']]).T
#     colors = np.vstack([plydata['vertex'].data['red'], plydata['vertex'].data['green'], plydata['vertex'].data['blue']]).T

#     cloud={
#         "vertices": vertices,
#         "colors": colors,
#         "object_id": instanceids
#     }
#     print (f"Saving {scene_id}.pth to globalwork/vipradas/scannet/")
#     torch.save(cloud,f'/globalwork/vipradas/scannet/{scene_id}.pth')