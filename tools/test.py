# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import logging
import os
import os.path as osp
from pathlib import Path
from types import FunctionType

import numpy as np
import torch
from mmengine import print_log
from mmengine.config import Config, DictAction
from mmengine.model import is_model_wrapper
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from PIL import Image
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import MAP_FUNC


def parse_args():
    parser = argparse.ArgumentParser(description="Test model")
    parser.add_argument("config", help="config file name or path.")
    parser.add_argument(
        "--val_folder", default=None, help="Path to the Mevis val dataset"
    )
    parser.add_argument("--checkpoint", default=None, help="checkpoint file")
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument("--deepspeed", default=None, help="Dummy option")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    #     args = parser.parse_args()
    #     if 'LOCAL_RANK' not in os.environ:
    #         os.environ['LOCAL_RANK'] = str(args.local_rank)
    #     return args
    #
    # def parse_args():
    #     parser = argparse.ArgumentParser(
    #         description="Video Reasoning Segmentation for val set"
    #     )
    parser.add_argument(
        "--save-dir",
        default="./val_results",
        help="Directory to save segmented results",
    )
    args = parser.parse_args()
    return args


def register_function(cfg_dict):
    if isinstance(cfg_dict, dict):
        for key, value in dict.items(cfg_dict):
            if isinstance(value, FunctionType):
                value_str = str(value)
                if value_str not in MAP_FUNC:
                    MAP_FUNC.register_module(module=value, name=value_str)
                cfg_dict[key] = value_str
            else:
                register_function(value)
    elif isinstance(cfg_dict, (list, tuple)):
        for value in cfg_dict:
            register_function(value)


def main():
    args = parse_args()

    if args.deepspeed is not None:
        print_log(
            "Deepspeed is not adopted during inference, Skipped.", level=logging.WARN
        )

    # parse config
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f"Cannot find {args.config}")

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register FunctionType object in cfg to `MAP_FUNC` Registry and
    # change these FunctionType object to str
    register_function(cfg._cfg_dict)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    # build the runner from config
    if "runner_type" not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)

        if is_model_wrapper(runner.model):
            runner.model.module.load_state_dict(state_dict, strict=False)
        else:
            runner.model.load_state_dict(state_dict, strict=False)
        runner.logger.info(f"Load checkpoint from {args.checkpoint}")
    else:
        Warning("No checkpoint !!!")

    model = runner.model.module
    tokenizer = model.tokenizer
    model.eval()
    # 2. Read the "expressions.json" from val folder
    val_path = Path(args.val_folder)
    ann_file = val_path / "meta_expressions.json"
    if not ann_file.exists():
        raise FileNotFoundError(
            f"Could not find {ann_file}. Is the val folder correct?"
        )

    with open(ann_file, "r") as f:
        annotation_data = json.load(f)

    # The JSON structure is typically: {"videos": {video_id: {frames: [...], expressions: {...}}}}
    videos_info = annotation_data["videos"]
    print(f"Found {len(videos_info)} videos in {ann_file}")

    # 3. Prepare output directory
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Results will be saved to: {args.save_dir}\n")

    # 4. Loop over each video, then each expression
    with torch.inference_mode():
        for video_id, vid_data in videos_info.items():
            # Frames for this video
            vid_frames = sorted(vid_data["frames"])  # e.g., ["0001", "0002", ...]

            # Load all frames as PIL
            # JPEGImages/{video_id}/{frame_name}.jpg
            frame_paths = [
                os.path.join(val_path, "JPEGImages", video_id, f"{fn}.jpg")
                for fn in vid_frames
            ]
            frames_pil = []
            for fp in frame_paths:
                if not os.path.exists(fp):
                    raise FileNotFoundError(f"Frame not found: {fp}")
                frames_pil.append(Image.open(fp).convert("RGB"))

            # Go through each expression in this video
            expressions_dict = vid_data["expressions"]
            for exp_id, exp_info in expressions_dict.items():
                out_dir = os.path.join(args.save_dir, video_id, exp_id)
                if os.path.exists(out_dir):
                    continue
                # We'll form a text prompt for the model
                # The default text is cfg.text, e.g., "<image>Please describe the video content."
                # Some code might append the expression, but you can adapt as needed.
                # For example:
                # text_prompt = f"<image> Describe: {exp_info['exp']}"
                # Or simply use cfg.text.
                text_prompt = f"<image>Please segment {exp_info['exp']}."

                print(
                    f"Processing video={video_id}, exp_id={exp_id} with {len(frames_pil)} frames."
                )

                # 5. Run the model for the entire video
                result = model.predict_forward(
                    video=frames_pil, text=text_prompt, tokenizer=tokenizer
                )

                # 6. Print or log the raw language model output
                prediction_text = result["prediction"]
                print(f"  Model output: {prediction_text}")

                # 7. Check for segmentations, overlay, and save
                if "[SEG]" in prediction_text:
                    # The model might return multiple sets of masks,
                    # typically `result['prediction_masks']`
                    # Here we assume there's at least one set
                    seg_idx = 0
                    if "prediction_masks" not in result:
                        print(
                            "  prediction_masks missing from result; cannot visualize."
                        )
                        continue
                    pred_masks = result["prediction_masks"][seg_idx]

                    # Create output folder: {work_dir}/{video_id}/{exp_id}
                    out_dir = os.path.join(args.save_dir, video_id, exp_id)
                    os.makedirs(out_dir, exist_ok=True)

                    # For each frame, overlay mask and save
                    for frame_idx, frame_name in enumerate(vid_frames):
                        pred_mask = pred_masks[frame_idx]
                        out_path = os.path.join(out_dir, f"{frame_name}.png")
                        # visualize_and_save(pred_mask, frame_paths[frame_idx], out_path)
                        mask = pred_mask.astype(np.float32)
                        mask = Image.fromarray(mask * 255).convert("L")
                        mask.save(out_path)

                else:
                    print("No segmentation; skipping mask overlay.")


if __name__ == "__main__":
    main()
