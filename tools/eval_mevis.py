###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

import cv2
from metrics import db_eval_boundary, db_eval_iou
from pycocotools import mask as cocomask

NUM_WORKERS = 128


def eval_queue(q, rank, out_dict, mevis_pred_path):
    while not q.empty():
        # print(q.qsize())
        vid_name, exp = q.get()

        vid = exp_dict[vid_name]

        exp_name = f"{vid_name}_{exp}"

        if not os.path.exists(f"{mevis_pred_path}/{vid_name}/{exp}"):
            print(f"{exp_name} not found")
            out_dict[exp_name] = [0, 0]
            continue

        pred_0_path = f"{mevis_pred_path}/{vid_name}/{exp}/00000.png"
        pred_0 = cv2.imread(pred_0_path, cv2.IMREAD_GRAYSCALE)
        h, w = pred_0.shape
        vid_len = len(vid["frames"])
        gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

        anno_ids = vid["expressions"][exp]["anno_id"]

        for frame_idx, frame_name in enumerate(vid["frames"]):
            for anno_id in anno_ids:
                mask_rle = mask_dict[str(anno_id)][frame_idx]
                if mask_rle:
                    gt_masks[frame_idx] += cocomask.decode(mask_rle)

            pred_masks[frame_idx] = cv2.imread(
                f"{mevis_pred_path}/{vid_name}/{exp}/{frame_name}.png",
                cv2.IMREAD_GRAYSCALE,
            )

        j = db_eval_iou(gt_masks, pred_masks).mean()
        f = db_eval_boundary(gt_masks, pred_masks).mean()
        out_dict[exp_name] = [j, f]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mevis_path",
        type=str,
        default="/nodes/cristal/work/nekrasov/data/language-data/mevis/valid_u",
    )
    parser.add_argument(
        "--prediction_path",
        type=str,
        default="runs/experiment/xmem_prediction",
    )
    args = parser.parse_args()
    queue = mp.Queue()
    exp_dict = json.load(open(str(Path(args.mevis_path) / "meta_expressions.json")))[
        "videos"
    ]
    mask_dict = json.load(open(str(Path(args.mevis_path) / "mask_dict.json")))

    shared_exp_dict = mp.Manager().dict(exp_dict)
    shared_mask_dict = mp.Manager().dict(mask_dict)
    output_dict = mp.Manager().dict()

    for vid_name in exp_dict:
        vid = exp_dict[vid_name]
        for exp in vid["expressions"]:
            queue.put([vid_name, exp])

    start_time = time.time()
    processes = []
    for rank in range(NUM_WORKERS):
        p = mp.Process(
            target=eval_queue, args=(queue, rank, output_dict, args.prediction_path)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    with open(Path(args.prediction_path).parent / "test_dict.json", "w") as f:
        json.dump(dict(output_dict), f)

    j = [output_dict[x][0] for x in output_dict]
    f = [output_dict[x][1] for x in output_dict]

    print(f"J: {np.mean(j)}")
    print(f"F: {np.mean(f)}")
    print(f"J&F: {(np.mean(j) + np.mean(f)) / 2}")

    end_time = time.time()
    total_time = end_time - start_time
    print("time: %.4f s" % (total_time))
