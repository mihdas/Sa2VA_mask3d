import argparse
import concurrent.futures
import json
import os

import cv2
import mmengine
import numpy as np
import torch
import torch.distributed
import torch.utils.data
import tqdm
from PIL import Image as PILImage
from pycocotools import mask as cocomask
from transformers import AutoModel, AutoTokenizer

from projects.llava_sam2.evaluation.dataset import RefVOSDataset
from projects.llava_sam2.evaluation.utils import (
    _init_dist_pytorch,
    _init_dist_slurm,
    collect_results_cpu,
    get_dist_info,
    get_rank,
)
from sklearn.cluster import DBSCAN
import imageio

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



def async_func(executor, func, **kwargs):
    future = executor.submit(func, **kwargs)
    return future


def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(cocomask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]["counts"] = rle[-1]["counts"].decode()
    return rle


def mask_save(item, mask_prediction, save_dir, save_using_cv2=False):
    vid_id = item["video_id"]
    exp_id = item["exp_id"]
    save_path = os.path.join(save_dir, vid_id, exp_id)
    mmengine.mkdir_or_exist(save_path)
    for id_m, mask in enumerate(mask_prediction):
        if save_using_cv2:
            mask = mask.astype(bool) * 255
            cv2.imwrite("example_binary_mask.png", mask)
        else:
            mask = PILImage.fromarray(mask.astype(np.float32) * 255).convert("L")
            file_name = item["frames"][id_m]
            save_file = os.path.join(save_path, file_name + ".png")
            mask.save(save_file)


DATASETS_INFO = {
    "DAVIS": {
        "data_root": "/davis17/",
        "image_folder": "/davis17/valid/JPEGImages/",
        "expression_file": "/davis17/meta_expressions/valid/meta_expressions.json",
        "mask_file": "/davis17/valid/mask_dict.pkl",
    },
    "MEVIS": {
        "data_root": "/mevis/valid/",
        "image_folder": "/mevis/valid/JPEGImages",
        "expression_file": "/mevis/valid/meta_expressions.json",
        "mask_file": None,
    },
    "MEVIS_U": {
        "data_root": "/mevis/valid_u/",
        "image_folder": "/mevis/valid_u/JPEGImages/",
        "expression_file": "/mevis/valid_u/meta_expressions.json",
        "mask_file": "/mevis/valid_u/mask_dict.json",
    },
    "REFYTVOS": {
        "data_root": "/youtube-vos/",
        "image_folder": "/youtube-vos/valid/JPEGImages/",
        "expression_file": "/youtube-vos/meta_expressions/valid/meta_expressions.json",
        "mask_file": None,
    },
    "REVOS": {
        "data_root": "/revos/REVOS/",
        "image_folder": "/revos/REVOS/JPEGImages/",
        "expression_file": "/revos/REVOS/meta_expressions_valid_.json",
        "mask_file": None,
    },
    "SCANREFER": {
        "data_root": "/globalwork/datasets/scannet/",  #TODO
        "image_folder": "/globalwork/vipradas/scannet_images/", #TODO
        "expression_file": "/globaldata/scanrefer/ScanRefer_filtered_val.json", #TODO
        "mask_file": None,
    },


}


def read_matrix_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        matrix = np.array([[float(num) for num in line.split()] for line in lines])
    return matrix


def parse_args():
    parser = argparse.ArgumentParser(description="RefVOS")
    parser.add_argument("model_path", help="hf model path.")
    parser.add_argument(
        "--dataset",
        choices=DATASETS_INFO.keys(),
        default="MEVIS",
        help="Specify a dataset",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--deepspeed", type=str, default=None)  # dummy
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


if __name__ == "__main__":
    args = parse_args()

    print(args)
    save_path = args.save_path
    # language_data_path = "/nodes/cristal/work/nekrasov/data/language-data"
    language_data_path = args.data_path

    if args.launcher == "none":
        rank = 0
        world_size = 1
    elif args.launcher == "pytorch":
        _init_dist_pytorch("nccl")
        rank, world_size = get_dist_info()
    elif args.launcher == "slurm":
        _init_dist_slurm("nccl")
        rank, world_size = get_dist_info()

    model = (
        AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        )
        .eval()
        .to_empty(device='cuda')
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    dataset_info = DATASETS_INFO[args.dataset]

    image_folder = language_data_path + dataset_info["image_folder"]
    expression_file = language_data_path + dataset_info["expression_file"]
    mask_file = dataset_info["mask_file"]
    if mask_file is not None:
        mask_file = language_data_path + mask_file

    dataset = RefVOSDataset(
        image_folder=image_folder,
        expression_file=expression_file,
        mask_file=mask_file,
    )

    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=2,
        pin_memory=False,
        collate_fn=lambda x: x[0],
    )
    results = []
    executor = concurrent.futures.ThreadPoolExecutor()
    for item in tqdm.tqdm(dataloader):
        with torch.no_grad():
            result = model.predict_forward(
                video=item["images"],
                text=item["text_prompt"],
                tokenizer=tokenizer,
            )

        text_idx = 0
        text_prediction = result["prediction"]
        if len(result["prediction_masks"]) > 0:
            mask_prediction = result["prediction_masks"][text_idx]
        else:
            print(text_prediction)
            mask_prediction = np.zeros(
                (item["length"], item["ori_height"], item["ori_width"]), dtype=np.uint8
            )

        # only saving the frames that were sampled for prediction
        if result.get("sample_idx") is not None:
            item["frames"] = np.array(item["frames"])[result["sample_idx"]].tolist()

        if args.submit:
            async_func(
                executor,
                mask_save,
                item=item,
                mask_prediction=mask_prediction,
                save_dir=save_path,
                save_using_cv2=args.dataset == "REFYTVOS",
            )
            # mask_save(item=item, mask_prediction=mask_prediction, save_dir=save_path)
            encoded_mask = None
        else:
            encoded_mask = mask_to_rle(mask_prediction)
        
        scene_id=item["video_id"].split("/")[0]
        


        K = read_matrix_txt(f"/globalwork/vipradas/scannet_images/{scene_id}/intrinsics/intrinsic_depth.txt")#TODO
        data=torch.load(f"/globalwork/vipradas/scannet/{scene_id}.pth") #TODO
        locs_in=data["vertices"]    #TODO
        ground_truth=data["object_id"] #TODO
        n_points = locs_in.shape[0]
        depth_scale = 1000


        point2img_mapper = PointCloudToImageMapper(image_dim=(640,480), intrinsics=K)
        
        votes = np.zeros([n_points,], dtype=int)
        imgs=#read the uniform json list 
        target_height = 480
        target_width = 640
        for idx,mask in enumerate(mask_prediction):
            #TODO save uniformly sampled frames list to a json file and read it
            depth_image=depth_image.replace('.jpg','.png').replace('color','depth') #TODO use model predicted depth
            depth = imageio.v2.imread(depth_image)/depth_scale
            pose= np.linalg.inv(read_matrix_txt(depth_image.replace('depth','pose').replace('png','txt'))) #TODO
            #TODO read axis align matrix and multiply with pose
            mapping = np.ones([n_points, 3], dtype=int)
            mapping[:, 0:3] = point2img_mapper.compute_mapping(pose, locs_in, depth)

            zoom_height = target_height / mask.shape[0]
            zoom_width = target_width / mask.shape[1]

            resized_mask = zoom(mask, (zoom_height, zoom_width), order=1)
          
            prediction2d = (resized_mask > 0.7)

            flag = mapping[:, 2].astype(int)
            x_img = mapping[:, 0].astype(int)
            y_img = mapping[:, 1].astype(int)

            mask_hits = prediction2d[ x_img, y_img] & flag==1
            votes[mask_hits] += 1

        pred=votes>0
        ground_truth=np.isin(ground_truth, item['object_id'])
        iou = (pred & ground_truth).sum() / ((pred | ground_truth).sum()+1e-6)
        print("IoU", iou)
        if np.count_nonzero(votes)>0:
            masked_points = locs_in[pred]
            clustering = DBSCAN(eps=0.05, min_samples=10).fit(masked_points)
            labels = clustering.labels_
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            largest_label = unique_labels[np.argmax(counts)]
            largest_cluster_mask = labels == largest_label
            largest_cluster_points = masked_points[largest_cluster_mask]
            pred_min_xyz = masked_points.min(axis=0)
            pred_max_xyz = masked_points.max(axis=0)
            bbox=np.concatenate([pred_min_xyz, pred_max_xyz], axis=0)
        else:
            bbox = np.zeros((6,)) # empty bbox

        result = {
            "index": item["index"],
            "video_id": item["video_id"],
            "exp_id": item["exp_id"],
            "text_prediction": text_prediction,
            "frames": item["frames"],
            "exp": item["text_prompt"],
            "prediction_masks": encoded_mask,
            "iou": iou,
            "aabb_bound": bbox,

        }
        print(result)
        results.append(result)

    executor.shutdown(wait=True)
    print(f"[Rank {rank}] : Finished.")

    if not args.submit:
        results = collect_results_cpu(results, len(dataset))
        if get_rank() == 0:
            final_results = {}
            for item in results:
                vid_id = item["video_id"]
                exp_id = item["exp_id"]
                if vid_id not in final_results:
                    final_results[vid_id] = {}
                assert exp_id not in final_results[vid_id]
                final_results[vid_id][exp_id] = item
            os.makedirs(save_path, exist_ok=True)
            json.dump(final_results, open(f"{save_path}/results.json", "w"))

    if rank == 0:
        print("Done")
