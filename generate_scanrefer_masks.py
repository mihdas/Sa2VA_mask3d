import os
import json
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import torch
import argparse
import cv2
from tqdm import tqdm
from pycocotools import mask as cocomask
import numpy as np
try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")

def mask_to_rle(mask):
    rle = []
    #print(mask.shape)
    #TODO Uncomment for pi3 models
    mask=mask.reshape(mask.shape[0],mask.shape[2],mask.shape[3])
    print(mask.shape)
    for m in mask:
        #print(m.shape)
        encoded = cocomask.encode(np.asfortranarray(m.astype(np.uint8)))
        #print(encoded)
        counts = encoded["counts"]
        if isinstance(counts, bytes):
            encoded["counts"] = counts.decode()
        elif hasattr(counts, "tolist"):
            encoded["counts"] = counts.tolist()
        # else leave as-is if already str or list
        rle.append(encoded)
    return rle


def parse_args():
    parser = argparse.ArgumentParser(description='Video Reasoning Segmentation')
    parser.add_argument('--start', default=0)
    parser.add_argument('--end', default=9508)
    parser.add_argument('--model', default="8B")
    parser.add_argument('--flash',default=False)
    args = parser.parse_args()
    return args


def visualize(pred_mask, image_path, work_dir):
    print(work_dir)
    print(image_path)
    visualizer = Visualizer()
    img = cv2.imread(image_path+".JPG")
    visualizer.set_image(img)
    visualizer.draw_binary_masks(pred_mask, colors='g', alphas=0.4)
    visual_result = visualizer.get_image()

    output_path = os.path.join(work_dir+"/images", os.path.basename(image_path)+".JPG")
    cv2.imwrite(output_path,visual_result)
    npy_path=os.path.join(work_dir+"/masks",os.path.splitext(os.path.basename(image_path))[0]+".npy")
    np.save(npy_path,pred_mask)

def validate(start_index,end_index,model_size,flash):

    print("Pi3Scanrefer - Nr3D - Uniform") #TODO Change confirmation print
    
    #model_path = "/nodes/cristal/work/nekrasov/saved/sa2va-himtok/sa2va_4b_ufo4t_pi3_hf" #TODO Change model path
    model_path = "/home/vipradas/Thesis/Sa2VA_p/pi3_scanrefer"
    model = AutoModelForCausalLM.from_pretrained(
         model_path,
         use_flash_attn=flash,
         torch_dtype="auto",
         device_map="cuda",
         trust_remote_code=True
     )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
         trust_remote_code=True
     )

    scenes=[]
    descriptions=[]
    objs=[]

    with open("/home/vipradas/Thesis/Sa2VA_p/raw_jsons/nr3d_test.json") as file:    #TODO Change benchmark json
        json_data=json.load(file)
        for prompt in json_data:
            sceneid=prompt['scene_id']
            scenes.append(sceneid)
          #  scenes.append(f"/globalwork/datasets/scannetpp/data/{sceneid}/dslr/resized_images/")
            descriptions.append(prompt['description'])
            objs.append(prompt['object_id'])
        file.close()

    with open("/home/vipradas/Thesis/Sa2VA_p/raw_jsons/uniform_samples.json") as file:    #TODO Change frame paths json
        video_folder=json.load(file)
        file.close()
    
    maskdict={}
    for idx,(scene_id,description,obj) in tqdm(enumerate(zip(scenes,descriptions,objs))):
        if idx<int(start_index) or idx>int(end_index):
           continue
        print(f"Processing {idx}th scene")
        vid_frames = []
    
        #video_folder=f"/globalwork/vipradas/scannet_images/{scene_id}/color/"

        #images_paths=os.listdir(video_folder)
        #images_paths=images_paths[::len(images_paths)//100]
        images_paths=video_folder[scene_id]
        images_paths=images_paths[:12]
        images_paths.sort(key= lambda x: int(x.split("/")[-1][:-4]))

        for img_path in images_paths:
            #img_path=video_folder+img_path
            img = Image.open(img_path).convert('RGB')
            vid_frames.append(img)
                    
        
        with torch.no_grad():
            result = model.predict_forward(video=vid_frames,text="<image>"+description+". Please provide a segmentation mask.",
                                                tokenizer=tokenizer, sample_num_frames=12) #TODO Remove sample num frames for non pi3 models
            prediction = result['prediction']
            print(description)
            print(prediction)

        if '[SEG]' in prediction:
            _seg_idx = 0
            pred_masks = result['prediction_masks']#[_seg_idx]  #TODO Change for non pi3
            #print(f"Pred masks len: {len(pred_masks), type(pred_masks)}")
            pred_masks = mask_to_rle(pred_masks)
            # for frame_idx in range(len(vid_frames)):
            #     pred_mask = pred_masks[frame_idx]
            #     os.makedirs(f"/nodes/astra/work/vipradas/masksvggt/{idx}/masks", exist_ok=True)
            #     os.makedirs(f"/nodes/astra/work/vipradas/masksvggt/{idx}/images", exist_ok=True)
            #     visualize(pred_mask, video_folder+images_paths[frame_idx][:-4],f"/nodes/astra/work/vipradas/masksvggt/{idx}")
            maskdict[idx]=pred_masks
        del result
        del vid_frames
        gc.collect()
        torch.cuda.empty_cache()
    with open(f"/home/vipradas/Thesis/Sa2VA_p/results_nr3d_pi3scanrefer_uniform_{start_index}_{int(end_index)}.json", "w") as f: #TODO Change output json path
        json.dump(maskdict, f, indent=4)


if __name__=='__main__':
    cfg = parse_args()
    validate(cfg.start, cfg.end, cfg.model, cfg.flash)

