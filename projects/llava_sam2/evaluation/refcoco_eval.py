import argparse
import copy
import math
import os
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset import RESDataset
from pycocotools import mask as _mask
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from utils import _init_dist_pytorch, collect_results_cpu, get_dist_info, get_rank
# from projects.llava_sam2.himtok_decoder.himt_dec import MaskDecoder


def parse_args():
    parser = argparse.ArgumentParser(description="RefCocoSeg")
    parser.add_argument("--model_path", help="hf model path.")
    parser.add_argument(
        "--data_path",
        help="Language data path",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS_ATTRIBUTES.keys(),
        default="refcoco",
        help="Specify a ref dataset",
    )
    parser.add_argument("--split", default="val", help="Specify a split")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


DATASETS_ATTRIBUTES = {
    "refcoco": {"splitBy": "unc", "dataset_name": "refcoco"},
    "refcoco_plus": {"splitBy": "unc", "dataset_name": "refcoco_plus"},
    "refcocog": {"splitBy": "umd", "dataset_name": "refcocog"},
}


def collate_fn(batch):
    """Custom collate function to handle variable-length text sequences"""
    return {
        "image": [item["image"] for item in batch],
        "text": [item["text"] for item in batch],
        "gt_masks": [item["gt_masks"] for item in batch],
        "img_id": [item["img_id"] for item in batch],
    }


def main():
    args = parse_args()

    if args.launcher != "none":
        _init_dist_pytorch("nccl")
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # Build model
    print(args)
    # language_data_path = "/nodes/cristal/work/nekrasov/data/language-data"
    language_data_path = args.data_path
    IMAGE_FOLDER = f"{language_data_path}/coco_barecat/train2014"
    DATA_PATH = f"{language_data_path}/refer_seg"
    
    model = (
        AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            use_flash_attn=True,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )

    # model.mask_decoder = MaskDecoder.init_model_from_config(
    #     model_path="/p/scratch/llmvidseg/alexey/saved/himtok.pth",
    #     config_path="/p/project1/llmvidseg/alexey/code/Sa2VA/projects/llava_sam2/configs/himt.yaml",
    #     need_encoder=True,
    #     need_decoder=True,
    #     device="cuda",
    #     dtype=torch.bfloat16
    # )
    #
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    dataset_info = DATASETS_ATTRIBUTES[args.dataset]

    dataset = RESDataset(
        image_folder=IMAGE_FOLDER,
        dataset_name=dataset_info["dataset_name"],
        data_path=DATA_PATH,
        split=args.split,
    )
    print(f"dataset len: {len(dataset)}")

    # Create distributed sampler for multi-GPU training
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
    else:
        sampler = None

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one sample at a time due to varying text lengths
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    results = []

    with torch.inference_mode():
        with tqdm(
            total=len(dataloader),
            desc=f"Rank {rank}: Processing videos",
            position=rank,
            leave=False,
        ) as pbar:
            for batch in dataloader:
                # Unpack batch (batch_size=1)
                image = batch["image"][0]
                texts = batch["text"][0]  # List of texts for this image
                gt_masks = batch["gt_masks"][0]
                img_id = batch["img_id"][0]

                prediction = {
                    "img_id": img_id,
                    "gt_masks": mask_to_rle(gt_masks.cpu().numpy()),
                }

                pred_masks = []
                for i, text in enumerate(texts):
                    data_batch = {
                        "image": image,
                        "text": text,
                    }
                    #from pudb import set_trace; set_trace()
                    pred_mask = model.predict_forward(
                        **data_batch, tokenizer=tokenizer
                    )["prediction_masks"]
                    if len(pred_mask) == 0:
                        print("No seg pred !!!")
                        pred_masks.append(None)
                    else:
                        _ret_mask = pred_mask[0]
                        _ret_mask = mask_to_rle(_ret_mask)
                        pred_masks.append(_ret_mask)

                prediction.update({"prediction_masks": pred_masks})
                results.append(prediction)
                pbar.update(1)

        # Create a shared temporary directory
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
        else:
            tmpdir = None

        # Broadcast the temporary directory path to all processes
        if world_size > 1:
            tmpdir = [tmpdir]
            torch.distributed.broadcast_object_list(tmpdir, src=0)
            tmpdir = tmpdir[0]

        # Collect results using the temporary directory
        results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)

        # Synchronize all processes before cleanup
        if world_size > 1:
            torch.distributed.barrier()

        # Only rank 0 evaluates the results
        if rank == 0:
            metric = dataset.evaluate(results, None)
            print(metric)

        # Clean up the temporary directory
        if rank == 0:
            import shutil

            shutil.rmtree(tmpdir)

        if world_size > 1:
            torch.distributed.barrier()

        if rank == 0:
            print("Done")


def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]["counts"] = rle[-1]["counts"].decode()
    return rle


if __name__ == "__main__":
    main()
