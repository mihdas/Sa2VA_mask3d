import numpy as np
from mmengine.visualization import Visualizer
import os
import json
import cv2

def visualize(idx,scene):

    visualizer = Visualizer()
    images=os.listdir(f"/nodes/hoppiness/work/vipradas/masks4B/{idx}/masks")
    for image in images:
        img = cv2.imread(f"/globalwork/datasets/scannetpp/data/{scene}/dslr/resized_images/"+ image[:-4]+".JPG")
        pred_mask=np.load(f"/nodes/hoppiness/work/vipradas/masks4B/{idx}/masks/"+image)
        visualizer.set_image(img)
        visualizer.draw_binary_masks(pred_mask, colors='g', alphas=0.4)
        visual_result = visualizer.get_image()

        #output_path = os.path.join(work_dir+"/images", image)
        cv2.imwrite(f"/home/vipradas/Thesis/viz/"+image[:-4]+".JPG" ,visual_result)
        #npy_path=os.path.join(work_dir+"/masks",os.path.splitext(os.path.basename(image_path))[0]+".npy")
        #np.save(npy_path,pred_mask)


scenes=[]
with open("/home/vipradas/Thesis/Sa2VA_p/instruct3D_val.json") as file:
    json_data=json.load(file)
    for prompt in json_data:
        sceneid=prompt['scene_id']
        scenes.append(sceneid)
     #  scenes.append(f"/globalwork/datasets/scannetpp/data/{sceneid}/dslr/resized_images/")
        #descriptions.append(prompt['description'])
    file.close()

idx=4
visualize(idx,scenes[idx])


