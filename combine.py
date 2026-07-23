# import json
# from tqdm import tqdm

# file_list = ['results_0_1189.json',
#  'results_1190_2317.json',
#   'results_2318_3565.json',
#   'results_3566_4754.json',
#   'results_4755_5943.json',
#  'results_5944_7132.json',
#   'results_7133_8321.json',
#   'results_8322_9508.json',
#   ]  # add your files here
# merged_data = {}

# for json_file in tqdm(file_list):
#     with open(json_file, 'r') as file:
#         data = json.load(file)
#         merged_data={**merged_data, **data}
        
# with open('results_combined.json', 'w') as out:
#     json.dump(merged_data, out, indent=4)

# import torch
# data=torch.load("/home/vipradas/Thesis/iouDict12gt.pth")
# sm=0
# s25=0
# s50=0
# for k,v in data.items():
#     sm+=v
#     if v>=0.25:
#         s25+=1
#     if v>=0.5:
#         s50+=1
# print(sm/len(data), s25/len(data), s50/len(data))

# import json
# from tqdm import tqdm

# scenes=[]
# with open('/home/vipradas/Thesis/slurm_490800.out', 'r') as file:
#     for line in file:
#         words=line.split()
#         print(words[-1])
#         scenes.append(float(words[-1]))
#     file.close()



# sm=0
# s25=0
# s50=0
# for iou in scenes:
#     sm+=iou
#     if iou>=0.25:
#         s25+=1
#     if iou>=0.5:
#         s50+=1
# print(sm/len(scenes), s25/len(scenes), s50/len(scenes))


# with open('/globalwork/vipradas/ScanRefer_maskDict.json', 'r') as file:
#     data = json.load(file)
#     file.close()

# for scene in tqdm(scenes):
#     scene_data={}
#     for k,v in data.items():
#         if k.startswith(scene):
#             scene_data[k]=v
#     with open(f'/globalwork/vipradas/ScanRefer_maskDict/{scene}_maskDict.json', 'w') as out:
#         json.dump(scene_data, out, indent=4)


# import json
# #from tqdm import tqdm   
# with open(f'mask_dict_ScanRefer.json', 'r') as file:
#     data = json.load(file)
#     file.close()
# for k,v in data.items():

#     print(k, len(v))


# scenes=[]
# with open('/globaldata/scanrefer/ScanRefer_filtered_train.txt', 'r') as file:
#     for line in file:
#         scenes.append(line.strip())
#     file.close()    

# scenes=['scene0000_00']  # for testing  

# for scene in tqdm(scenes):
#     with open(f'/globalwork/vipradas/ScanRefer_maskDict/{scene}_maskDict.json', 'r') as file:
#         data = json.load(file)
#         file.close()
#     new_data={
#         'masklet':[[] for _ in range(len(data["scene0000_00_39"]))]
#     }
#     print(data.keys())
#     for k,v in data.items():
#         print(k, len(v)) 
#     print(len(data["scene0000_00_39"]))
#     print(len(new_data['masklet']))
#     for i in range(len(data["scene0000_00_39"])):
#         for k,v in data.items():
#             new_data['masklet'][i].append(data[k][i])

#     with open(f'{scene}_maskletDict.json', 'w') as out:
#         json.dump(new_data, out, indent=4)  

# import json 
# import os
# from tqdm import tqdm
# with open('/home/vipradas/Thesis/LLaVA-3D-Data/LLaVA-3D-Instruct-860K.json', 'r') as file:
# #with open('/home/vipradas/Thesis/MMScan/mmscan_data/MMScan-beta/MMScan_samples/MMScan_QA.json', 'r') as file:
#     data=json.load(file)
#     file.close()

# # dct={}
# # for item in tqdm(data['train']):
# #     if item['sub_class'] not in dct:
# #         dct[item['sub_class']]=1
# #     else:
# #         dct[item['sub_class']]+=1

# # for k,v in dct.items():
# #     print(k,v)
# scan_datasets={}
# caption_datasets={}
# sc=[]
# ms=[]
# rs=[]
# keys=[]
# new=0
# for item in tqdm(data):
#     frm=item['video'].split("/")[0]
#     scn=item['video'].split("/")[-1]
#     if frm not in scan_datasets:
#         scan_datasets[frm]=1
#     else:
#         scan_datasets[frm]+=1
#     caption_from= item['metadata']['dataset']
#     if caption_from not in caption_datasets:
#         caption_datasets[caption_from]=1
#     else:
#         caption_datasets[caption_from]+=1
#     if 'MMScan_QA' in caption_from and 'scannet' not in frm:
#         new+=1
#     if caption_from == 'scanQA':
#         if item.get('box', None) is not None:
#             print('lol', end="")



        
# print(len(scan_datasets), scan_datasets)
# print(len(caption_datasets), caption_datasets)
# print(new)
# print(list(keys))

# llava_3d=[]
# for item in data:
#     if 'scannet' in item['video']:
#         if 'object_id' in item['metadata'].keys():
#             llava_3d.append({'scene_id': item['video'].split("/")[-1], 'description': item['conversations'][0]['value'][8:], 'object_id': item['metadata']['object_id']})


# with open('/home/vipradas/Thesis/Sa2VA_p/Llava3d.json', 'w') as file:
#     json.dump(llava_3d, file, indent=4)

# print(len(data))
# for item in data:
#     if 'scene' not in item['scene_id']:
#         print(item['scene_id'])
    
   
# import json 
# with open('/home/vipradas/Thesis/Sa2VA_p/mask_dict_LLava3D.json', 'r') as file:
#     data=json.load(file)
#     file.close()

# test_mask_dict={}
# for k,v in data.items():
#     if k.startswith('scene0323_00'):
#         test_mask_dict[k]=v
# with open('/home/vipradas/Thesis/Sa2VA_p/mask_dict_test.json', 'w') as out:
#     json.dump(test_mask_dict, out, indent=4)

# import csv
# import json
# with open('raw_jsons/sr3d_test.json','r') as file:
#     data=json.load(file)
#     file.close()
# scenes={}
# for item in data:
#     scene_id=item['scene_id']
#     if scene_id not in scenes:
#         scenes[scene_id]=1
# print(len(scenes))
# from tqdm import tqdm

# jsonlist=[]

# with open('/home/vipradas/Downloads/nr3d_test.csv', newline='') as file:
#     reader = csv.reader(file)
#     next(reader)  # Skip the header row if there is one
#     tmp_ann_id_count = {}
#     for row in reader:
#         scene_id= row[5]
#         object_id=int(row[7])
#         scene_obj_key = (scene_id, object_id)
#         if scene_obj_key not in tmp_ann_id_count:
#             tmp_ann_id_count[scene_obj_key] = 0
#         else:
#             tmp_ann_id_count[scene_obj_key] += 1
        
#         jsonlist.append({'scene_id': scene_id, 'object_id': object_id, 'description': row[1], 'ann_id': tmp_ann_id_count[scene_obj_key], 'eval_type': 'unique'})
        

# with open('/home/vipradas/Thesis/Sa2VA_p/raw_jsons/nr3d_test.json', 'w') as out:
#     json.dump(jsonlist, out, indent=4)

# import os
# import json
# from tqdm import tqdm

# with open("raw_jsons/nr3d_test.json","r") as f:
#     data = json.load(f)
#     f.close()

# dct={}
# for prompt in data:
#     if  prompt['scene_id'] not in dct:
#         dct[prompt['scene_id']]=1

# scenes=os.listdir("/globalwork/vipradas/scannet")
# for i in range(len(scenes)):
#     scenes[i]=scenes[i][:-4]
# sm=0
# t=0
# lst=[]
# for k in dct.keys():
#     t+=1
#     if k not in scenes:
#         #print(k)
#         lst.append(k)
#         sm+=1
# with open("raw_jsons/missing_scenes_nr3d_test.txt","w") as f:
#     for item in lst:
#         f.write("%s\n" % item)
#     f.close()


# import os
# import pickle

# with open(f'/home/vipradas/Thesis/embodiedscan_infos_val.pkl', 'rb') as f:
#     pc_data = pickle.load(f)
# ks=['metainfo', 'data_list']
# print(pc_data['data_list'][0].keys())

# pc_avl=os.listdir('/globalwork/vipradas/scannet_images')[141:]

# scenes=[]
# with open('raw_jsons/ScanNet.txt', 'r') as file:
#     for line in file:
#         scenes.append(line.strip())
#     file.close()    
# a=set(pc_data.keys())
# b=set(pc_avl)
# c=set(scenes)

# print(len(a), len(b), len(c))
# print(len(b-(a.union(b)))) # 1619 1619 1619

# import json

# with open('mask_dicts/mask_dict_ScanRefer.json','r') as file:
#     data=json.load(file)
#     file.close

# with open('mask_dicts/mask_dict_ScanRefer_noindent.json', 'w') as file:
#     json.dump(data,file)

# import jsonlines
# from tqdm import tqdm
# import os
# matterport_scenes={}
# trscan={}
# scannet={}
# with jsonlines.open("/home/vipradas/Thesis/MMScan/mmscan_tool/MMScan_QA_train.jsonl","r") as file:
#     for obj in tqdm(file):
#         images=obj['images']
#         for image in images:
#             dataset=image.split("/")[0]
#             scene=image.split("/")[1]
#             if scene not in matterport_scenes and dataset=='matterport3d':
#                 matterport_scenes[scene]=1
#             elif scene not in trscan and dataset=='3rscan':
#                 trscan[scene]=1
#                 #print(image)
#             # if scene not in scannet and dataset=='scannet':
#             #     scannet[scene]=1
#             #     #print(scene)
#             else:
#                 break
            
# extracted= os.listdir("/globalwork/vipradas/3rscan")
# print('3rscan')
# for scene in trscan.keys():
#     if scene not in extracted:
#         print(scene)


# import json
# import cv2
# import os
# from pycocotools import mask as cocomask
# from tqdm import tqdm


# from mmengine.visualization import Visualizer
# imgpath="/globalwork/vipradas/scannet_images/scene0639_00/color"
# images= os.listdir(imgpath)
# mask_dict=json.load(open("/home/vipradas/Thesis/Sa2VA_p/mask_dicts/mask_dict_ScanRefer.json"))

# masks=mask_dict['scene0639_00_0']

# print(len(masks),len(images))

# def visualize(pred_mask, image_path, work_dir):
#     visualizer = Visualizer()
#     img = cv2.imread(image_path)
#     visualizer.set_image(img)
#     visualizer.draw_binary_masks(pred_mask, colors='g', alphas=0.4)
#     visual_result = visualizer.get_image()

#     output_path = os.path.join(work_dir, os.path.basename(image_path))
#     cv2.imwrite(output_path, visual_result)


# for img,mask in tqdm(zip(images[::30],masks[::30])):
#     mask=cocomask.decode(mask)
#     mask=mask>0
#     img=imgpath+"/"+img
#     visualize(mask,img, "/home/vipradas/Thesis/Sa2VA_p/vdbug")

# import os
# import json
# from tqdm import tqdm
# with open("raw_jsons/ScanRefer_filtered_val.json","r") as file:
#     data=json.load(file)
# # with open("raw_jsons/scannet_select_frames_16.json","r") as file:
# #     samples=json.load(file)

# #samples={}
# #print(len(data))
# scenes={}
# for prompt in tqdm(data):
#     scene=prompt['scene_id']
#     if scene not in scenes:
#         scenes[scene]=1
#         #print("Missing ",scene)
# import json
# from tqdm import tqdm
# import os

# scenes={}
# with open("raw_jsons/sr3d_test.json","r") as file:
#     data=json.load(file)

# with open("raw_jsons/scannet_select_frames_16.json","r") as file:
#     scenes=json.load(file)

# ms={}
# for prompt in tqdm(data):
#     scene=prompt['scene_id']
#     if scene not in scenes:
#         ms[scene]=1

# print(ms.keys(), len(ms))
#         images=os.listdir(f"/globalwork/vipradas/scannet_images/{scene}/color")
        
#         images=images[::len(images)//12][:12]
#         #print(images)
#         for i in range(len(images)):
#             images[i]=f"/globalwork/vipradas/scannet_images/{scene}/color/{images[i]}"
#         scenes[scene]=images

# with open("raw_jsons/sr3d_uniform_samples.json","w") as f:
#     json.dump(scenes, f, indent=4)

# with open("scannet_val_scenes.txt", "a") as f:
#     for scen in scenes.keys():
#         print(scen, file =f)
#     else:
#         images=os.listdir(f"/globalwork/vipradas/scannet_images/{scene}/color")
        
#         images=images[::len(images)//12][:12]
#         #print(images)
#         for i in range(len(images)):
#             images[i]=f"/globalwork/vipradas/scannet_images/{scene}/color/{images[i]}"
#         samples[scene]=images

# with open("raw_jsons/nr3d_samples.json","w") as f:
#     json.dump(samples, f, indent=4)


# import h5py
# import tqdm
# keys=[]

# with h5py.File("/globalwork/vipradas/enet_feats_maxpool.hdf5", "r") as f:

#     print((f.keys()))


# import json
# with open ("raw_jsons/sr3d_frames_16.json","r") as file:
#     dict1=json.load(file)
#     file.close()
# with open ("raw_jsons/scannet_select_frames_16.json","r") as file:
#     dict2=json.load(file)
#     file.close()

# merged = {**dict1, **dict2}

# with open("raw_jsons/scannet_select_frames_16.json","w") as out:
#     json.dump(merged, out, indent=4)

import json 

with open("expressions/scanrefer_val.json","r") as f:
    data=json.load(f)

prompts=data['videos']

print(len(prompts.keys()))