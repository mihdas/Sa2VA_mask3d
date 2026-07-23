import os
from tqdm import tqdm
import collections
import torch
import imageio
import numpy as np
from PIL import Image
import json
from pycocotools import mask as cocomask
from scipy.ndimage import zoom
import cv2
from sklearn.cluster import DBSCAN

##################################################################################################################################################
# FOR SCANNET++
##################################################################################################################################################
# CameraModel = collections.namedtuple(
#     "CameraModel", ["model_id", "model_name", "num_params"])
# BaseCamera = collections.namedtuple(
#     "Camera", ["id", "model", "width", "height", "params"])
# BaseImage = collections.namedtuple(
#     "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
# Point3D = collections.namedtuple(
#     "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])



# class Image(BaseImage):
#     def qvec2rotmat(self):
#         return qvec2rotmat(self.qvec)

#     def to_transform_mat(self):
#         '''
#         R, t matrix
#         '''
#         R = self.qvec2rotmat()
#         t = self.tvec 
#         T = np.eye(4)
#         T[:3, :3] = R
#         T[:3, 3] = t
#         return T


#     @property
#     def world_to_camera(self) -> np.ndarray:
#         R = qvec2rotmat(self.qvec)
#         t = self.tvec
#         world2cam = np.eye(4)
#         world2cam[:3, :3] = R
#         world2cam[:3, 3] = t
#         return world2cam


# class Camera(BaseCamera):
#     @property
#     def K(self):
#         K = np.eye(3)
#         if self.model == "SIMPLE_PINHOLE" or self.model == "SIMPLE_RADIAL" or self.model == "RADIAL" or self.model == "SIMPLE_RADIAL_FISHEYE" or self.model == "RADIAL_FISHEYE":
#             K[0, 0] = self.params[0]
#             K[1, 1] = self.params[0]
#             K[0, 2] = self.params[1]
#             K[1, 2] = self.params[2]
#         elif self.model == "PINHOLE" or self.model == "OPENCV" or self.model == "OPENCV_FISHEYE" or self.model == "FULL_OPENCV" or self.model == "FOV" or self.model == "THIN_PRISM_FISHEYE":
#             K[0, 0] = self.params[0]
#             K[1, 1] = self.params[1]
#             K[0, 2] = self.params[2]
#             K[1, 2] = self.params[3]
#         else:
#             raise NotImplementedError
#         return K

# def qvec2rotmat(qvec):
#     return np.array([
#         [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
#          2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
#          2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
#         [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
#          1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
#          2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
#         [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
#          2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
#          1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

# CAMERA_MODELS = {
#     CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
#     CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
#     CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
#     CameraModel(model_id=3, model_name="RADIAL", num_params=5),
#     CameraModel(model_id=4, model_name="OPENCV", num_params=8),
#     CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
#     CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
#     CameraModel(model_id=7, model_name="FOV", num_params=5),
#     CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
#     CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
#     CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
# }
# CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
#                          for camera_model in CAMERA_MODELS])
# CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
#                            for camera_model in CAMERA_MODELS])

# def read_cameras_text(path):
#     """
#     see: src/base/reconstruction.cc
#         void Reconstruction::WriteCamerasText(const std::string& path)
#         void Reconstruction::ReadCamerasText(const std::string& path)
#     """
#     cameras = {}
#     with open(path, "r") as fid:
#         while True:
#             line = fid.readline()
#             if not line:
#                 break
#             line = line.strip()
#             if len(line) > 0 and line[0] != "#":
#                 elems = line.split()
#                 camera_id = int(elems[0])
#                 model = elems[1]
#                 width = int(elems[2])
#                 height = int(elems[3])
#                 params = np.array(tuple(map(float, elems[4:])))
#                 cameras[camera_id] = Camera(id=camera_id, model=model,
#                                             width=width, height=height,
#                                             params=params)
#     fid.close()
#     return cameras

# def read_images_text(path):
#     """
#     see: src/base/reconstruction.cc
#         void Reconstruction::ReadImagesText(const std::string& path)
#         void Reconstruction::WriteImagesText(const std::string& path)
#     """
#     images = {}
#     with open(path, "r") as fid:
#         while True:
#             line = fid.readline()
#             if not line:
#                 break
#             line = line.strip()
#             if len(line) > 0 and line[0] != "#":
#                 elems = line.split()
#                 image_id = int(elems[0])
#                 qvec = np.array(tuple(map(float, elems[1:5])))
#                 tvec = np.array(tuple(map(float, elems[5:8])))
#                 camera_id = int(elems[8])
#                 image_name = elems[9]
#                 elems = fid.readline().split()
#                 xys = np.column_stack([tuple(map(float, elems[0::3])),
#                                        tuple(map(float, elems[1::3]))])
#                 point3D_ids = np.array(tuple(map(int, elems[2::3])))
#                 images[image_name] = Image(
#                     id=image_id, qvec=qvec, tvec=tvec,
#                     camera_id=camera_id, name=image_name,
#                     xys=xys, point3D_ids=point3D_ids)
#     fid.close()
#     return images
##################################################################################################################################################
# FOR SCANNET++
##################################################################################################################################################


class PointCloudToImageMapper(object):
    def __init__(self, image_dim,
            visibility_threshold=0.25, cut_bound=0, intrinsics=None):
        
        self.image_dim = image_dim
        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound
        self.intrinsics = intrinsics

    def compute_mapping(self, world_to_camera, coords, depth=None, intrinsic=None):
        """
        :param world_to_camera: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        if self.intrinsics is not None: # global intrinsics
            intrinsic = self.intrinsics

        mapping = np.zeros((3, coords.shape[0]), dtype=int)
        coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coords_new.shape[0] == 4, "[!] Shape error"

        #world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coords_new)
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = np.round(p).astype(int) # simply round the projected coordinates
        inside_mask = (pi[0] >= self.cut_bound) * (pi[1] >= self.cut_bound) \
                    * (pi[0] < self.image_dim[0]-self.cut_bound) \
                    * (pi[1] < self.image_dim[1]-self.cut_bound)
        if depth is not None:
            depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
            occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]]
                                    - p[2][inside_mask]) <= \
                                    self.vis_thres * depth_cur

            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2]>0 # make sure the depth is in front
            inside_mask = front_mask*inside_mask
        mapping[0][inside_mask] = pi[1][inside_mask]
        mapping[1][inside_mask] = pi[0][inside_mask]
        mapping[2][inside_mask] = 1

        return mapping.T


def read_matrix_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        matrix = np.array([[float(num) for num in line.split()] for line in lines])
    return matrix

def read_axis_align_matrix(file_path):
    axis_align_matrix = None
    with open(file_path, "r") as f:
        for line in f:
            line_content = line.strip()
            if 'axisAlignment' in line_content:
                axis_align_matrix = [float(x) for x in line_content.strip('axisAlignment = ').split(' ')]
                axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
                break
    return axis_align_matrix

def get_8_corners(min_xyz, max_xyz, M):
    points = np.array([
        [min_xyz[0], min_xyz[1], min_xyz[2]],
        [min_xyz[0], min_xyz[1], max_xyz[2]],
        [min_xyz[0], max_xyz[1], min_xyz[2]],
        [min_xyz[0], max_xyz[1], max_xyz[2]],
        [max_xyz[0], min_xyz[1], min_xyz[2]],
        [max_xyz[0], min_xyz[1], max_xyz[2]],
        [max_xyz[0], max_xyz[1], min_xyz[2]],
        [max_xyz[0], max_xyz[1], max_xyz[2]],
    ])
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points_aligned_h = (M @ points_h.T).T
    points_aligned = points_aligned_h[:, :3]

    return points_aligned


target_height = 480
target_width = 640


###############----------------------------------------------------------------------------------------###############    
scenes={} 
with open("/home/vipradas/Thesis/Sa2VA_p/raw_jsons/ScanRefer_filtered_val.json") as file: #TODO Change dataset
    json_data=json.load(file)
    for idx,prompt in enumerate(json_data):
        if idx< 0:
            continue
        if prompt['scene_id'] in scenes:
            scenes[prompt['scene_id']]["idxs"].append(idx)
            scenes[prompt['scene_id']]["targets"].append(prompt['object_id'])
            scenes[prompt['scene_id']]["ann_ids"].append(prompt['ann_id'])
        else:
            sceneid = prompt['scene_id']
            scenes[sceneid] = {"idxs": [idx], "targets": [prompt['object_id']], "ann_ids": [prompt['ann_id']]}
    file.close()

visualize_scenes=["scene0699_00"]



with open(f"/home/vipradas/Thesis/Sa2VA_p/raw_jsons/scannet_select_frames_16.json", 'r') as file: #TODO Change image path here
    input_images=json.load(file)
    file.close()

with open(f"/home/vipradas/Thesis/Sa2VA_p/results_scanrefer_sa2va-i_volume_0_9508.json", 'r') as file: #TODO Change mask path here
    masks_dict=json.load(file)
    file.close()
iouDict=[]
split=1
for scene_index,scene_id in tqdm(enumerate(scenes)):

    K = read_matrix_txt(f"/globalwork/vipradas/scannet_images/{scene_id}/intrinsics/intrinsic_depth.txt")
    data=torch.load(f"/globalwork/vipradas/scannet/{scene_id}.pth")
    locs_in=data["vertices"]
    ground_truth=data["object_id"]
    #print(ground_truth)

    n_points = locs_in.shape[0]
    depth_scale = 1000



    point2img_mapper = PointCloudToImageMapper(image_dim=(640,480), intrinsics=K)

    depth_images = input_images[scene_id][:12]
    depth_images.sort(key= lambda x: int(x.split("/")[-1][:-4]))
    #print(depth_images)



    
    votes ={}
    preds=[]
    for depth_index,depth_image in enumerate(depth_images):
        #depth = imageio.v2.imread(f"/globalwork/vipradas/scannet_images/{scene_id}/depth/{depth_image}")/depth_scale
        depth_image=depth_image.replace('.jpg','.png').replace('color','depth')
        depth = imageio.v2.imread(depth_image)/depth_scale

        pose= np.linalg.inv(read_matrix_txt(depth_image.replace('depth','pose').replace('png','txt')))
        alignment_matrix=read_axis_align_matrix(f"/globalwork/datasets/scannet/scannet/scans/{scene_id}/{scene_id}.txt")

        
        mapping = np.ones([n_points, 3], dtype=int)
        mapping[:, 0:3] = point2img_mapper.compute_mapping(pose, locs_in, depth)

        for test,idx in enumerate(scenes[scene_id]["idxs"]):

            if f"votes{idx}" not in votes:
                votes[f"votes{idx}"] = np.zeros([n_points,], dtype=int)

            prediction2d = masks_dict[str(idx)][depth_index]
            prediction2d = cocomask.decode(prediction2d)
            #print(prediction2d.shape)
            

            zoom_height = target_height / prediction2d.shape[0]
            zoom_width = target_width / prediction2d.shape[1]

            resized_mask = zoom(prediction2d, (zoom_height, zoom_width), order=1)

            
            prediction2d = (resized_mask > 0.7)

            flag = mapping[:, 2].astype(int)
            x_img = mapping[:, 0].astype(int)
            y_img = mapping[:, 1].astype(int)

            mask_hits = prediction2d[ x_img, y_img] & flag==1
            #in_mapping= np.count_nonzero(mask_hits)            
            votes[f"votes{idx}"][mask_hits] += 1
    
    for idx,target,ann_id in zip(scenes[scene_id]["idxs"],scenes[scene_id]["targets"],scenes[scene_id]["ann_ids"]):
            
        pred = votes[f"votes{idx}"]

        
        pred=pred>0
        masked_points = locs_in[pred]
        
        gt = np.isin(ground_truth,int(target))

        intersection= (gt & pred).sum()
        union =( gt | pred).sum()
        iou = intersection / (union + 1e-6)
        #print(np.count_nonzero(pred), np.count_nonzero(gt), intersection, union)
        iouDict.append(iou)
        
        if np.count_nonzero(pred) ==0:
            pred_min_xyz = np.array([0,0,0])
            pred_max_xyz = np.array([0,0,0])
        # else:
        #     pred_min_xyz = masked_points.min(axis=0)
        #     pred_max_xyz = masked_points.max(axis=0)
        else:
            clustering = DBSCAN(eps=0.05, min_samples=10).fit(masked_points)
            labels = clustering.labels_
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            cluster_labels = labels[labels != -1]
            if cluster_labels.size == 0:
                largest_label = None   # or handle accordingly
                pred_min_xyz = np.array([0,0,0])
                pred_max_xyz = np.array([0,0,0])
            else:
                #unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                largest_label = unique_labels[np.argmax(counts)]
                largest_label = unique_labels[np.argmax(counts)]
                largest_cluster_mask = labels == largest_label
                largest_cluster_points = masked_points[largest_cluster_mask]
                pred_min_xyz = largest_cluster_points.min(axis=0)
                pred_max_xyz = largest_cluster_points.max(axis=0)

        preds.append({
            "object_id": int(target),
            "ann_id": int(ann_id),
             "aabb": [get_8_corners(pred_min_xyz, pred_max_xyz, alignment_matrix).tolist()] # Eval script needs extra batch dimension
        })
        print(f"Scene: {scene_id}, ScanRefer idx: {idx}, Target obj id: {target}, IoU: {iou}")        

    #print(votes)
    #torch.save(votes, f"/home/vipradas/Thesis/multi3drefer/{scene_id}_votes.pth")
    with open(f"/home/vipradas/Thesis/bboxpreds/scanrefer/sa2vai/volume1/{scene_id}.json", 'w') as f: # TODO CHANGE SAVE PATH HERE
        json.dump(preds, f, indent=2)
        
torch.save(iouDict, f"/home/vipradas/Thesis/iouDictpi3volume.pth")