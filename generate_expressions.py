import os 
import json
from tqdm import tqdm

with open("raw_jsons/nr3d_train.json","r") as f:
    data = json.load(f)
    f.close()
    
expressions={
    'videos': {}
}

def remove_ext(file_list):
    return [os.path.splitext(file)[0] for file in file_list]

dct={}
scene_dct={}

for prompt in tqdm(data):
    if prompt['scene_id']=='scene0531_00' or prompt['scene_id']=="scene0546_00":
        continue
    if prompt['scene_id'] not in scene_dct:
        scene_dct[prompt['scene_id']]=0
    else:
        scene_dct[prompt['scene_id']]+=1
    
    
    scene_id = f"{prompt['scene_id']}/color/{prompt['object_id']}/{scene_dct[prompt['scene_id']]}"
    # scene_id = f"{prompt['scene_id']}/color"
    
    
    if scene_id not in dct:
        dct[scene_id]=0
        
        frames=(os.listdir("/globalwork/vipradas/scannet_images/"+scene_id.split("/")[0]+"/color"))
        #print(frames)
        frames=frames[::len(frames)//24]
        frames=frames[:24]
        #frames=[]
        
        for i in range(len(frames)):
            frames[i]=frames[i][:-4]
        expressions['videos'][scene_id] = {
            'expressions': {
                '0': {
                    'exp': prompt['description'],
                    'anno_id' :  [str(scene_dct[prompt['scene_id']])],
                    #'anno_id' : [f"{prefix}_{x}" for x in prompt['object_id']],
                    'obj_id': [int(prompt['object_id'])]
                }
            },
            'frames': frames         
        }
    else:
        dct[scene_id]+=1
    
        #print(scene_id.split("/")[0])
        
    
        
        expressions['videos'][scene_id]['expressions'][dct[scene_id]] = {
            'exp': prompt['description'],
            # 'anno_id' : prompt['ann_id'],
            #'anno_id' : [f"{prefix}_{x}" for x in prompt['object_id']],
            'obj_id': [ int(prompt['object_id'])]
        }

with open("/home/vipradas/Thesis/Sa2VA_p/expressions/nr3d_train_24.json", "w") as f:
    json.dump(expressions, f, indent=4)
