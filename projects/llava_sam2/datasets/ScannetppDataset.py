import logging
import os
from typing import Literal
import collections
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from mmengine import print_log
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import build_origin_dataset
import copy
import barecat
#from .encode_fn import video_lisa_encode_fn
from mask3d import load_mesh
import open3d as o3d    
import copy
from xtuner.dataset.utils import get_bos_eos_token_ids
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX

def bbox_8points_to_6dof(points_8x3):
    """
    Convert 3D bounding box from 8 corner points (Nx3 numpy array) 
    to 6DoF format: center (3,), extent (3,).
    
    Args:
        points_8x3: numpy array of shape (8, 3) with 3D corner points
    
    Returns:
        tuple: (center, R, extent)
        - center: np.array(3,) - translation
        - R: np.array(3x3) - rotation matrix (box-to-world)
        - extent: np.array(3,) - half-lengths along box axes
    """
    assert points_8x3.shape == (8, 3), "Input must be (8,3)"
    points_8x3=torch.tensor(points_8x3, requires_grad=False)
    mins=points_8x3.min( dim=0).values
    maxs=points_8x3.max( dim=0).values
    bbox=torch.cat([mins,maxs],dim=0)
    return bbox

def video_lisa_encode_fn(
        example,
        tokenizer,
        max_length,
        input_ids_with_output=True,
        **kwargs
):
    """We only support the following three scenarios:

    1. Incremental pretraining dataset.
        example['conversation'] = [
                {
                    'input': '',
                    'output': '### Human: Can you write xxx'
                }
            ]

    2. Single-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                }
            ]

    3. Multi-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                },
                {
                    'input': 'Please expand on the second point.',
                    'output': 'Here is an expanded explanation of the xxx'
                }
            ]
    """
    bos_token_id, eos_token_id = get_bos_eos_token_ids(tokenizer)
    is_multi_turn_conversation = len(example['conversation']) > 1
    if is_multi_turn_conversation:
        assert input_ids_with_output

    input_ids, labels = [], []
    next_needs_bos_token = True
    for single_turn_conversation in example['conversation']:
        input = single_turn_conversation['input']
        input_encode = tokenizer.encode(input, add_special_tokens=False)
        if next_needs_bos_token:
            input_ids += bos_token_id
            labels += [IGNORE_INDEX] * len(bos_token_id)
        input_ids += input_encode
        labels += [IGNORE_INDEX] * len(input_encode)
        if input_ids_with_output:
            # Add output
            output_with_loss = single_turn_conversation.get(
                'output_with_loss', True)
            output = single_turn_conversation['output']
            output_encode = tokenizer.encode(output, add_special_tokens=False)
            input_ids += output_encode
            if output_with_loss:
                labels += copy.deepcopy(output_encode)
            else:
                labels += [IGNORE_INDEX] * len(output_encode)
            # Add EOS_TOKEN (with loss)
            if single_turn_conversation.get('need_eos_token', True):
                next_needs_bos_token = True
                input_ids += eos_token_id
                if output_with_loss:
                    labels += copy.deepcopy(eos_token_id)
                else:
                    labels += [IGNORE_INDEX] * len(eos_token_id)
            else:
                next_needs_bos_token = False
            # Add SEP (without loss)
            sep = single_turn_conversation.get('sep', '')
            if sep != '':
                sep_encode = tokenizer.encode(sep, add_special_tokens=False)
                input_ids += sep_encode
                labels += [IGNORE_INDEX] * len(sep_encode)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return {'input_ids': input_ids, 'labels': labels}

import json
import random
import pycocotools.mask as maskUtils
import cv2
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
# import sonata
# #import flash_attn
# #flash_attn=None

# CameraModel = collections.namedtuple(
#     "CameraModel", ["model_id", "model_name", "num_params"])
# BaseCamera = collections.namedtuple(
#     "Camera", ["id", "model", "width", "height", "params"])
# BaseImage = collections.namedtuple(
#     "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
# Point3D = collections.namedtuple(
#     "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

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



# def rename_keys(pc):
#     pc.pop("sampled_labels")
#     pc.pop("sampled_instance_labels")
#     pc.pop("sampled_instance_anno_id")
#     pc["coord"]=pc.pop("sampled_coords")
#     pc["color"]=pc.pop("sampled_colors")
#     #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pc["coord"])
#     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#     normals=np.array(pcd.normals)
#     pc["normal"]=normals
#     return pc





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

def convert_to_rgb(img):
    if img.mode != 'RGB':
        return img.convert('RGB')
    return img

# def accumulate_projected_features(
#     point_cloud, 
#     point_features,     
#     intrinsics,       
#     world2cam,
#     k1,
#     k2,
#     k3,
#     k4,
#     original_width,
#     original_height


    
# ) -> torch.Tensor:
   
#     device = torch.device("cuda")
    

#     N, C = point_features.shape
#     H, W = (1024,1024)
   
#     with torch.no_grad():

#         intrinsics=intrinsics.to(device=device, dtype=torch.bfloat16)
#         point_cloud=point_cloud.to(device=device,dtype=torch.bfloat16 )

#         world2cam=world2cam.to(device, dtype=torch.bfloat16)
#         point_features= point_features.to(device=device,dtype=torch.bfloat16)

#         x_min, y_min, z_min = torch.amin(point_cloud, dim=0)
#         x_max, y_max, _ = torch.amax(point_cloud, dim=0)
#         shift = torch.tensor([(-x_min + x_max) / 2, (-y_min +y_max) / 2, z_min], device = device)
#         point_cloud=point_cloud+shift
  
#         ones = torch.ones((N, 1),device=device,dtype=torch.bfloat16)
#         homog_points = torch.cat([point_cloud, ones], dim=1)
#         homog_points=homog_points.to(device, dtype=torch.bfloat16)  # (N, 4)
#         cam_points = (world2cam @ homog_points.T).T[:, :3]     # (N, 3)

#         scale_x = W / original_width
#         scale_y = H / original_height

#         K_new=intrinsics.clone()

#         K_new[0, 0]=intrinsics[0,0]*scale_x
#         K_new[1, 1]=intrinsics[1,1]*scale_y
#         K_new[0, 2]=intrinsics[0,2]*scale_x
#         K_new[1, 2]=intrinsics[1,2]*scale_y

       
#         X, Y, Z = cam_points[:, 0], cam_points[:, 1], cam_points[:, 2]
#         x = X/Z
#         y = Y/Z

#         #radial distortion
#         r2 = x ** 2 + y ** 2
#         r = torch.sqrt(r2)
#         theta = torch.atan(r)
#         theta_d = theta * (1 + k1 * theta ** 2 + k2 * theta ** 4 + k3 * theta ** 6 + k4 * theta ** 8)
#         scale = torch.where(r > 1e-8, theta_d / r, torch.ones_like(r))
#         x_d = x * scale
#         y_d = y * scale

#         # Apply intrinsics
#         fx, fy = K_new[0, 0], K_new[1, 1]
#         cx, cy = K_new[0, 2], K_new[1, 2]

#         x = fx * x_d + cx
#         y = fy * y_d + cy



#         valid = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (Z > 0)
#         x = x[valid]
#         y = y[valid]
#         feats = point_features[valid]
#         feats=feats.to(dtype=torch.bfloat16, device=device)


#         feat_map = torch.zeros((C, H, W), device=device, dtype=torch.bfloat16)
#         counts = torch.zeros((1, H, W), device=device, dtype=torch.bfloat16) + 1e-5

#         idx_flat = y * W + x
#         idx_flat = idx_flat.long() 
#         feat_map = feat_map.view(C, -1)
#         counts = counts.view(1, -1)

#         feat_map = feat_map.index_add(1, idx_flat, feats.T) 
#         counts = counts.index_add(1, idx_flat, torch.ones((1, feats.shape[0]), device=device, dtype=torch.bfloat16))

#         feat_map = (feat_map / counts).view(C, H, W)

# #        print(feat_map.shape)
        
        
#     return feat_map  



SEG_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",

    "Can you segment the {class_name} in this image",
    "Please segment {class_name} in this image",
    "What is {class_name} in this image? Please respond with segmentation mask",
    "What is {class_name} in this image? Please output segmentation mask",

    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",

    "Could you provide a segmentation mask for the {class_name} in this image",
    "Please identify and segment the {class_name} in this image",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask",
    "Can you highlight the {class_name} in this image with a segmentation mask",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

class ScannetppDataset(Dataset):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    FAST_IMG_CONTEXT_TOKEN = '<FAST_IMG_CONTEXT>'
    FAST_IMG_START_TOKEN = '<fast_img>'
    FAST_IMG_END_TOKEN = '</fast_img>'

    def __init__(self,
                 image_folder,
                 expression_file,
                 mask_file=None,
                 extra_image_processor=None,
                 tokenizer=None,
                 select_number=5,
                 sampled_frames=10,
                 offline_processed_text_folder=None,
                 template_map_fn=None,
                 max_length=2048,
                 lazy=True,
                 repeats=1,
                 special_tokens=None,
                 frame_contiguous_sample=False,
                 use_fast=False,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 # only work if use_fast = True
                 n_fast_images=50,
                 fast_pool_size=4,
                 fast_token_after_question=False,
    ):
        assert lazy is True
        self.tokenizer = BUILDER.build(tokenizer)
        self.select_number = select_number
        self.sampled_frames = sampled_frames
        assert offline_processed_text_folder or (expression_file and tokenizer)
        self.lazy = lazy

        self.max_length = max_length

        self.template_map_fn = template_map_fn
        if isinstance(self.template_map_fn, dict) and self.lazy:
            _type = self.template_map_fn['type']
            del self.template_map_fn['type']
            self.template_map_fn = _type(**self.template_map_fn)

        if offline_processed_text_folder and expression_file:
            print_log(
                'Both `offline_processed_text_folder` and '
                '`data_path` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)

        self.arch_type = arch_type
        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = '<|vision_start|>'
            self.IMG_END_TOKEN = '<|vision_end|>'
        elif self.arch_type == 'llava':
            self.IMG_CONTEXT_TOKEN = '<image>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''


        if offline_processed_text_folder is not None:
            raise NotImplementedError
        else:
            vid2metaid, metas, mask_dict = self.json_file_preprocess(expression_file, mask_file)
            self.vid2metaid = vid2metaid
            self.videos = list(self.vid2metaid.keys())
            self.mask_dict = mask_dict
            self.json_datas = metas
            json_datas = metas
            json_data = DatasetDict({'train': HFDataset.from_list(json_datas)})
            if self.lazy:
                self.text_data = build_origin_dataset(json_data, 'train')
            else:
                raise NotImplementedError

        self.image_folder = image_folder
        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)
        self.down_ratio = 1
        self.repeats = repeats

        self._system = ''

        self.downsample_ratio = 0.5
        if self.arch_type == 'llava':
            self.downsample_ratio = 1
        self.image_size = 448
        if self.arch_type == 'llava':
            self.image_size = 336
        patch_size = 14
        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))
        if self.arch_type == 'qwen':
            self.patch_token = 1

        if preprocessor is None:
            self.transformer = T.Compose([
    T.Lambda(convert_to_rgb),
    T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
])
            self.preprocessor = None
        else:
            self.transformer = None
            self.preprocessor = BUILDER.build(preprocessor)

        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.use_fast = use_fast
        self.n_fast_images = n_fast_images
        self.fast_pool_size = fast_pool_size

        self.frame_contiguous_sample = frame_contiguous_sample

        # for visualization debug
        self.save_folder = './work_dirs/video_debug/'
        self.cur_number = 0

        # exist_thr
        self.exist_thr = 8
        self.fast_token_after_question = fast_token_after_question
        if self.fast_token_after_question:
            assert self.use_fast

        print("Video res dataset, include {} items.".format(len(self.vid2metaid)))

    def __len__(self):
        return len(self.vid2metaid) * self.repeats

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.vid2metaid:
            cur_len = 10000
            length_list.append(cur_len)
        return length_list

    def real_len(self):
        return len(self.vid2metaid)

    def json_file_preprocess(self, expression_file, mask_file):
        # prepare expression annotation files
        with open(expression_file, 'r') as f:
            expression_datas = json.load(f)['videos']

        metas = []
        anno_count = 0  # serve as anno_id
        vid2metaid = {}
        for vid_name in expression_datas:
            vid_express_data = expression_datas[vid_name]

            vid_frames = vid_express_data['frames']
            vid_len = len(vid_frames)

            exp_id_list = sorted(list(vid_express_data['expressions'].keys()))
            for exp_id in exp_id_list:
                exp_dict = vid_express_data['expressions'][exp_id]
                meta = {}
                meta['video'] = vid_name
                meta['exp'] = exp_dict['exp']  # str
                meta['mask_anno_id'] = exp_dict['anno_id']

                if 'obj_id' in exp_dict.keys():
                    meta['obj_id'] = exp_dict['obj_id']
                else:
                    meta['obj_id'] = [0, ]  # Ref-Youtube-VOS only has one object per expression
                meta['anno_id'] = [str(anno_count), ]
                anno_count += 1
                meta['frames'] = vid_frames
                meta['exp_id'] = exp_id

                meta['length'] = vid_len
                metas.append(meta)
                if vid_name not in vid2metaid.keys():
                    vid2metaid[vid_name] = []
                vid2metaid[vid_name].append(len(metas) - 1)

        # process mask annotation files
        with open(mask_file, 'rb') as f:
            mask_dict = json.load(f)

        return vid2metaid, metas, mask_dict

    def create_img_to_refs_mapping(self, refs_train):
        img2refs = {}
        for ref in refs_train:
            img2refs[ref["image_id"]] = img2refs.get(ref["image_id"], []) + [ref, ]
        return img2refs

    def decode_mask(self, video_masks, image_size):
        ret_masks = []
        for object_masks in video_masks:
            # None object
            if len(object_masks) == 0:
                if len(ret_masks) != 0:
                    _object_masks = ret_masks[0] * 0
                else:
                    _object_masks = np.zeros(
                        (self.sampled_frames, image_size[0], image_size[1]), dtype=np.uint8)
            else:
                _object_masks = []
                for i_frame in range(len(object_masks[0])):
                    _mask = np.zeros(image_size, dtype=np.uint8)
                    for i_anno in range(len(object_masks)):
                        if object_masks[i_anno][i_frame] is None:
                            continue
                        m = maskUtils.decode(object_masks[i_anno][i_frame])
                        if m.ndim == 3:
                            m = m.sum(axis=2).astype(np.uint8)
                        else:
                            m = m.astype(np.uint8)
                        _mask = _mask | m
                    _object_masks.append(_mask)
                _object_masks = np.stack(_object_masks, axis=0)
            # if self.pad_image_to_square:
            #     _object_masks = expand2square_mask(_object_masks)
            ret_masks.append(_object_masks)
        _shape = ret_masks[0].shape
        for item in ret_masks:
            if item.shape != _shape:
                print([_ret_mask.shape for _ret_mask in ret_masks])
                return None
        ret_masks = np.stack(ret_masks, axis=0)  # (n_obj, n_frames, h, w)

        ret_masks = torch.from_numpy(ret_masks)
        # ret_masks = F.interpolate(ret_masks, size=(self.image_size // self.down_ratio,
        #                           self.image_size // self.down_ratio), mode='nearest')
        ret_masks = ret_masks.flatten(0, 1)
        return ret_masks

    def dataset_map_fn(self, data_dict, select_k=5):
        images = []
        # print(data_dict)
        obj_ids=[]

        len_frames = len(data_dict[0]['frames'])
        for objet_info in data_dict:
            assert len_frames == len(objet_info['frames'])

        # prepare images, random select k frames
#        if len_frames > select_k + 1:
#            if self.frame_contiguous_sample and random.random() < 0.5:
                # do contiguous sample
#                selected_start_frame = np.random.choice(len_frames - select_k, 1, replace=False)
#                selected_frame_indexes = [selected_start_frame[0] + _i for _i in range(select_k)]
#            else:
#                selected_frame_indexes = np.random.choice(len_frames, select_k, replace=False)
#        else:
#            selected_frame_indexes = np.random.choice(len_frames, select_k, replace=True)

        selected_frame_indexes=[i for i in range(len_frames)]

        # selected_frame_indexes=selected_frame_indexes[:24]
        #print(selected_frame_indexes)
#        selected_frame_indexes.sort()
        if self.use_fast:
            # sample fast branch
            fast_interval = len_frames / (self.n_fast_images + 1e-4)
            sampled_fast_frame_idxs = [min(int(i * fast_interval), len_frames - 1) for i in range(self.n_fast_images)]
            fast_video_frames = []
            for selected_frame_index in sampled_fast_frame_idxs:
                frame_id = data_dict[0]['frames'][selected_frame_index]
                fast_video_frames.append(os.path.join(data_dict[0]['video'], frame_id + ".jpg"))
        else:
            fast_video_frames = None
            sampled_fast_frame_idxs = None
        count=1
        for selected_frame_index in selected_frame_indexes:
            
            frame_id = data_dict[0]['frames'][selected_frame_index]
            images.append(os.path.join(data_dict[0]['video'] ,frame_id + ".jpg"))
            count+=1
 # prepare text
        expressions = [object_info['exp'] for object_info in data_dict]
        if self.use_fast:
            text_dict = self.prepare_text(select_k, expressions, num_image_tokens=self.patch_token,
                                          n_fast_images=len(fast_video_frames),)
        else:
            text_dict = self.prepare_text(select_k, expressions, num_image_tokens=self.patch_token)


        # prepare masks
        video_masks = []
        for object_info in data_dict:
            obj_id = object_info['obj_id']
            obj_ids.append(obj_id)
            anno_ids = object_info['mask_anno_id']
            # print('anno_ids: ', anno_ids)
            obj_masks = []
            for anno_id in anno_ids:
                anno_id = str(anno_id)
                #frames_masks = self.mask_dict[anno_id]
                frames_masks=[]
                frames_masks_ = []
                # for frame_idx in selected_frame_indexes:
                #     frames_masks_.append(copy.deepcopy(frames_masks[frame_idx]))
                obj_masks.append(frames_masks_)
            video_masks.append(obj_masks)

        if self.use_fast:
            fast_video_masks = []
            assert sampled_fast_frame_idxs is not None
            for object_info in data_dict:
                anno_ids = object_info['mask_anno_id']
                obj_masks = []
                for anno_id in anno_ids:
                    anno_id = str(anno_id)
                    frames_masks = self.mask_dict[anno_id]
                    frames_masks_ = []
                    for frame_idx in sampled_fast_frame_idxs:
                        frames_masks_.append(copy.deepcopy(frames_masks[frame_idx]))
                    obj_masks.append(frames_masks_)
                fast_video_masks.append(obj_masks)
        else:
            fast_video_masks = None
#        print(obj_ids)
        ret = {'images': images, 'obj_ids':obj_ids,
               'video_masks': video_masks, 'conversation': text_dict['conversation'],
               'fast_images': fast_video_frames, 'fast_video_masks': fast_video_masks}

 #       print(ret)
        return ret

    def prepare_text(self, n_frames, expressions, num_image_tokens=256, n_fast_images=50):

        if self.use_fast and not self.fast_token_after_question:
            fast_frame_token_str = f'{self.FAST_IMG_START_TOKEN}' \
                          f'{self.FAST_IMG_CONTEXT_TOKEN * n_fast_images * self.fast_pool_size * self.fast_pool_size}' \
                          f'{self.FAST_IMG_END_TOKEN}' + '\n'
        else:
            fast_frame_token_str = ''

        frame_token_str = f'{self.IMG_START_TOKEN}' \
                          f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                          f'{self.IMG_END_TOKEN}'
        if self.fast_token_after_question:
            assert self.use_fast
            after_question_str = f'{self.FAST_IMG_START_TOKEN}' \
                          f'{self.FAST_IMG_CONTEXT_TOKEN * n_fast_images * self.fast_pool_size * self.fast_pool_size}' \
                          f'{self.FAST_IMG_END_TOKEN}'
        else:
            after_question_str = ''

        questions = []
        answers = []
        for i, exp in enumerate(expressions):
            # the exp is a question
            if '?' in exp:
                questions.append(exp)
            else:
                exp = exp.replace('.', '').strip()
                question_template = random.choice(SEG_QUESTIONS)
                questions.append(question_template.format(class_name=exp.lower()))

            answers.append(random.choice(ANSWER_LIST))
        qa_list = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                frame_tokens = frame_token_str + '\n'
                # frame_tokens = '=' + ' '
                frame_tokens = frame_tokens * n_frames
                frame_tokens = frame_tokens.strip()
                frame_tokens = fast_frame_token_str + frame_tokens
                qa_list.append(
                    {'from': 'human', 'value': frame_tokens + question + after_question_str}
                )
            else:
                qa_list.append(
                    {'from': 'human', 'value': question + after_question_str}
                )
            qa_list.append(
                {'from': 'gpt', 'value': answer}
            )

        input = ''
        conversation = []
        for msg in qa_list:
            if msg['from'] == 'human':
                input += msg['value']
            elif msg['from'] == 'gpt':
                conversation.append({'input': input, 'output': msg['value']})
                input = ''
            else:
                raise NotImplementedError

        # add system information
        conversation[0].update({'system': self._system})
        return {'conversation': conversation}

    def __getitem__(self, index):
 #       print(self.vid2metaid, self.videos, self.json_datas)
      #assert isinstance(index, int), f"Expected int, got {type(index)}: {index}"
        index = index % self.real_len()
        selected_video_objects = self.vid2metaid[self.videos[index]]
        video_objects_infos = [copy.deepcopy(self.text_data[idx]) for idx in selected_video_objects]

        if len(video_objects_infos) > self.select_number:
            selected_indexes = np.random.choice(len(video_objects_infos), self.select_number)
            video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]
        else:
            selected_indexes = np.random.choice(len(video_objects_infos), self.select_number, replace=True)
            video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]

        data_dict = self.dataset_map_fn(video_objects_infos, select_k=self.sampled_frames)
    
        assert 'images' in data_dict.keys()
        pixel_values = []
        features_3d=[]
        extra_pixel_values = []
        num_video_tokens = None
        num_frame_tokens = None
        if data_dict.get('images', None) is not None:

            frames_files = data_dict['images']
            scene_id=frames_files[0].split("/")[0]
            #print( "scene id:", scene_id)
            frames_files = [os.path.join(self.image_folder, scene_id, "color", frame_file.split("/")[-1]) for frame_file in frames_files]
            # print(frames_files)
        
            #sonata_features = torch.load(os.path.join("/globalwork/vipradas/mask3dfeats",scene_id)+".pth", map_location='cpu').detach()

            mesh = load_mesh(f'/globalwork/datasets/scannet/scannet/scans/{scene_id}/{scene_id}_vh_clean_2.ply')
            sonata_features = {
                'vertices': np.asarray(mesh.vertices),
                'colors': np.asarray(mesh.vertex_colors),
            }

            

            projected= torch.load(f"/globalwork/vipradas/mask3dshuffle/{scene_id}.pth",  map_location='cpu').detach()
            projected=projected[:,:,:]
        
            data_dict['projected']=projected.to(dtype=torch.bfloat16)



            # with barecat.Barecat("/p/project1/llmvidseg/datasets/scanrefer/scanrefer_barecat/scanrefer.barecat", readonly=True) as bc:
            #     feat_path=f"sonata_feats/{scene_id}.pth"
            #     with bc.open(feat_path, 'rb') as f:
            #         sonata_features = torch.load(f, map_location='cpu')
            
            gt = torch.load(os.path.join("/globalwork/vipradas/scannet_m3drefer/val/",scene_id)+".pth")
            gt_bboxes=gt['aabb_corner_xyz']
            gt = gt ['instance_ids']
            
            gt_masks=[]
            bboxes=[]
            for obj in data_dict['obj_ids']:
                
                bbox=bbox_8points_to_6dof(gt_bboxes[int(obj[0])])

                mask=np.isin(gt,int(obj[0]))
                gt_masks.append(mask)
                bboxes.append(bbox)

            gt_masks= np.stack(gt_masks)
            bboxes=torch.stack(bboxes)
   #         cameras= read_cameras_text(os.path.join(self.image_folder,scene_id,"dslr","colmap","cameras.txt"))
   #         scannetppimages= read_images_text((os.path.join(self.image_folder,scene_id,"dslr","colmap","images.txt")))
            #if flash_attn is not None:
            #    model = sonata.load("sonata", repo_id="facebook/sonata").cuda()
            #else:
            #    custom_config = dict(
            #        enc_patch_size=[1024 for _ in range(5)],  # reduce patch size if necessary
            #        enable_flash=False,
            #        )
            #    model = sonata.load("sonata", repo_id="facebook/sonata", custom_config=custom_config).cuda()
            #transform = sonata.transform.default()
            #point = sonata.data.load(pc)
            ##original_coord = pc["coord"].copy()
            #pc = transform(pc)
            #self.model.eval()
            #with torch.inference_mode():
            #    for key in pc.keys():
            #        if isinstance(pc[key], torch.Tensor):
            #            pc[key] = pc[key].cuda(non_blocking=True)
        
             #   pc = self.model(pc)
            

    #        fx, fy, cx, cy, k1, k2, k3, k4 = cameras[1].params
    #        K = np.array([[fx,  0, cx],[ 0, fy, cy],[ 0,  0,  1]])
            


            for frame_path in frames_files:
                if "barecat" in frame_path:
                    frame_path = Path(frame_path)
                    barecat_dir = frame_path.parent.parent
                    with barecat.Barecat("/p/project1/llmvidseg/datasets/scanrefer/scanrefer_barecat/scanrefer.barecat", readonly=True) as bc:
                        frame_path = scene_id+ "color" + str(frame_path.name)
                        if bc.exists(frame_path):
                            with bc.open(frame_path, 'rb') as f:
                                frame_image = Image.open(f).convert('RGB')
                else:
                    frame_image = Image.open(frame_path).convert('RGB')
                ori_width, ori_height = frame_image.size
                if self.extra_image_processor is not None:
                    g_image = np.array(frame_image)  # for grounding
                    g_image = self.extra_image_processor.apply_image(g_image)
                    g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                    extra_pixel_values.append(g_pixel_values)
                    img=frame_path.split("/")[-1]
#                    projected_features=accumulate_projected_features(pc["coord"],pc["feat"],torch.tensor(K),torch.tensor(scannetppimages[img].world_to_camera),k1,k2,k3,k4,cameras[1].width, cameras[1].height)
                    #features_3d.append(pc['feat'])
                if self.preprocessor is not None:
                    pass
                else:
                    frame_image = self.transformer(frame_image)
                pixel_values.append(frame_image)

            if self.preprocessor is not None:
                if self.arch_type == 'qwen':
                    _data_dict = self.preprocessor(pixel_values, do_resize=True, size=(self.image_size, self.image_size))
                    _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                    _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int)
                    num_frame_tokens = int(_data_dict['image_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
                    num_frames = _data_dict['image_grid_thw'].shape[0]
                    num_video_tokens = num_frame_tokens * num_frames
                elif self.arch_type == 'llava':
                    _data_dict = self.preprocessor(pixel_values, do_resize=True, size=(self.image_size, self.image_size))
                    _data_dict['pixel_values'] = np.stack(_data_dict['pixel_values'], axis=0)
                    _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                else:
                    raise NotImplementedError
                data_dict.update(_data_dict)
            else:
                pixel_values = torch.stack(pixel_values, dim=0) # (n_f, 3, h, w)
                data_dict['pixel_values'] = pixel_values
            if self.extra_image_processor is not None:
                data_dict['g_pixel_values'] = extra_pixel_values
            
            # data_dict['scene_id']=scene_id
            data_dict['features_3d']= sonata_features
            data_dict['gt']=gt_masks
            data_dict['bboxes']=bboxes
            # process and get masks
            #masks = self.decode_mask(data_dict['video_masks'], image_size=(ori_height, ori_width))
            masks=torch.tensor([[[]]])
            if masks is None:
                return self.__getitem__(random.randint(0, self.real_len()))
            data_dict['masks'] = masks
        else:
            data_dict['pixel_values'] = torch.zeros(0, 3, self.image_size, self.image_size)
            data_dict['masks'] = None

        if num_video_tokens is not None:
            assert self.patch_token == 1
            input_str = data_dict['conversation'][0]['input']
            input_str = input_str.replace(self.IMG_CONTEXT_TOKEN, self.IMG_CONTEXT_TOKEN * num_frame_tokens)
            assert input_str.count(self.IMG_CONTEXT_TOKEN) == num_video_tokens
            data_dict['conversation'][0]['input'] = input_str

        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = video_lisa_encode_fn(data_dict, tokenizer=self.tokenizer, max_length=self.max_length)
        data_dict.update(result)

        # for fast branch
        if self.use_fast:
            fast_pixel_values = []
            frames_files = data_dict['fast_images']
            frames_files = [os.path.join(self.image_folder, frame_file) for frame_file in frames_files]
            for frame_path in frames_files:
                frame_image = Image.open(frame_path).convert('RGB')
                ori_width, ori_height = frame_image.size

                frame_image = self.transformer(frame_image)
                fast_pixel_values.append(frame_image)

            fast_pixel_values = torch.stack(fast_pixel_values, dim=0)  # (n_f, 3, h, w)
            data_dict['fast_pixel_values'] = fast_pixel_values

            # process and get masks
            masks = self.decode_mask(data_dict['fast_video_masks'], image_size=(ori_height, ori_width))

            if masks is None:
                return self.__getitem__(random.randint(0, self.real_len()))

            data_dict['fast_exists'] = masks.to(dtype=torch.int).sum(dim=(-2, -1)).ge(self.exist_thr).unsqueeze(-1)


            del data_dict['fast_video_masks']
        data_dict['type'] = 'video'
        return data_dict

    def visualization_debug(self, data_dict):
        save_folder = os.path.join(self.save_folder, 'sample_{}'.format(self.cur_number))
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        self.cur_number += 1

        # images

        show_images = []

        pixel_values = data_dict['pixel_values']
        save_folder_image = os.path.join(save_folder, 'image')
        if not os.path.exists(save_folder_image):
            os.mkdir(save_folder_image)
        for i_image, image_pixel_value in enumerate(pixel_values):
            # print(image_pixel_value.shape)
            image_pixel_value[0] = image_pixel_value[0] * 0.2686
            image_pixel_value[1] = image_pixel_value[1] * 0.2613
            image_pixel_value[2] = image_pixel_value[2] * 0.2757
            image_pixel_value[0] = image_pixel_value[0] + 0.4814
            image_pixel_value[1] = image_pixel_value[1] + 0.4578
            image_pixel_value[2] = image_pixel_value[2] + 0.4082
            image_pixel_value = image_pixel_value * 255
            image_pixel_value = image_pixel_value.permute(1, 2, 0)
            image_pixel_value = image_pixel_value.to(torch.uint8).numpy()
            # print(os.path.join(save_folder_image, '{}.jpg'.format(i_image)))
            # print(image_pixel_value.shape)
            show_images.append(image_pixel_value)
            cv2.imwrite(os.path.join(save_folder_image, '{}.jpg'.format(i_image)), image_pixel_value)

        # text
        input_text = self.tokenizer.decode(data_dict['input_ids'], skip_special_tokens=False)
        with open(os.path.join(save_folder, 'text.json'), 'w') as f:
            json.dump([input_text], f)

        # masks
        save_folder_mask = os.path.join(save_folder, 'mask')
        if not os.path.exists(save_folder_mask):
            os.mkdir(save_folder_mask)
        n_frames = len(pixel_values)
        masks = data_dict['masks']
        _, h, w = masks.shape
        masks = masks.reshape(-1, n_frames, h, w)
        for i_obj, obj_masks in enumerate(masks):
            save_folder_mask_obj_folder = os.path.join(save_folder_mask, 'obj_{}'.format(i_obj))
            if not os.path.exists(save_folder_mask_obj_folder):
                os.mkdir(save_folder_mask_obj_folder)
            for i_frame, f_mask in enumerate(obj_masks):
                f_mask = f_mask.numpy()
                f_mask = f_mask * 255
                f_mask = np.stack([f_mask * 1, f_mask * 0, f_mask * 0], axis=2)
                f_mask = show_images[i_frame] * 0.3 + 0.7 * f_mask
                f_mask = f_mask.astype(np.uint8)
                cv2.imwrite(os.path.join(save_folder_mask_obj_folder, '{}.png'.format(i_frame)), f_mask)
        return



if __name__ == '__main__':
    from transformers import AutoTokenizer
    from projects.llava_sam2.models.preprocess.image_resize import DirectResize
    from xtuner.dataset.map_fns import template_map_fn_factory
    from xtuner.utils import PROMPT_TEMPLATE
    prompt_template = PROMPT_TEMPLATE.internlm2_chat
    special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>']
    path = '/nodes/hoppiness/work/vipradas/InternVL2_5-4B'
    tokenizer = dict(type=AutoTokenizer.from_pretrained, pretrained_model_name_or_path=path, trust_remote_code=True,  padding_side='right')

    extra_image_processor = dict(type=DirectResize, target_length=1024,)
    
    video_scannetpp_dataset= ScannetppDataset(image_folder="/globalwork/vipradas/scannet_images",
                                                expression_file="/home/vipradas/Thesis/Sa2VA_p/expressions/scanrefer_val.json",
                                                mask_file="/home/vipradas/Thesis/Sa2VA_p/mask_dicts/mask_dict.json",
                                                tokenizer=tokenizer,
                                                extra_image_processor=extra_image_processor,
                                                template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),)

    print("dataset length:", len(video_scannetpp_dataset))
    item=video_scannetpp_dataset[0]
