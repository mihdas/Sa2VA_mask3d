import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor


def main(
    checkpoint,
    predictions_path,
    base_video_dir,
    prediction_dir_name,
    output_dir_name,
):
    try:
        # test distributed
        dist.init_process_group("nccl")

        # global rank
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    except Exception as e:
        print(e)
        print("Failed to initialize distributed environment")
        rank = 0
        world_size = 1

    # Define checkpoint and model configuration
    model_cfg = "configs/sam2/sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    # Define directories
    prediction_path = Path(predictions_path)
    base_video_dir = Path(base_video_dir)
    prediction_dir = prediction_path / prediction_dir_name
    output_base_dir = prediction_path / output_dir_name

    # Process each video sequence
    predicted_videos = sorted(prediction_dir.glob("*"))
    predicted_videos_slice = predicted_videos[rank::world_size]
    for video_dir in predicted_videos_slice:
        if not video_dir.is_dir():
            continue
        for seq_dir in sorted(video_dir.glob("*")):
            video_frames_dir = base_video_dir / seq_dir.parent.name
            # Create output directory mirroring the original structure
            relative_path = seq_dir.relative_to(prediction_dir)
            output_seq_dir = output_base_dir / relative_path
            output_seq_dir.mkdir(
                parents=True, exist_ok=True
            )  # Ensure output dir exists

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = predictor.init_state(str(video_frames_dir))
                obj_id = 1
                # Add initial masks
                for frame_path in seq_dir.glob("*.png"):
                    mask = torch.from_numpy(
                        np.array(cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE))
                    )
                    frame_id = int(frame_path.stem)
                    predictor.add_new_mask(state, frame_id, obj_id, mask)

                # Propagate masks through the video and save results
                for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                    # Extract mask for the tracked object (obj_id=1)
                    mask_tensor = masks[0][0]  # Shape: [H, W]
                    # Convert to binary mask and save as PNG
                    mask_np = (mask_tensor > 0.0).cpu().numpy()
                    output_path = output_seq_dir / f"{int(frame_idx):05d}.png"
                    mask = Image.fromarray(mask_np.astype(np.float32) * 255).convert(
                        "L"
                    )
                    mask.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 Video Object Tracking")
    parser.add_argument(
        "--checkpoint", default="/p/project1/llmvidseg/alexey/code/Sa2VA/pretrained/sam2_hiera_large.pt"
    )
    parser.add_argument(
        "--prediction-path",
        default="/nodes/cristal/fastwork/nekrasov/saved/sa2va_lila/lila_tar2_from_video_repro/",
    )
    parser.add_argument(
        "--base-video-dir",
        default="/p/project1/llmvidseg/alexey/data/mevis/valid/JPEGImages/",
    )
    parser.add_argument(
        "--prediction-dir-name",
        default="prediction",
    )
    parser.add_argument(
        "--output-dir-name",
        default="sam2_prediction",
    )

    args = parser.parse_args()
    main(
        args.checkpoint,
        args.prediction_path,
        args.base_video_dir,
        args.prediction_dir_name,
        args.output_dir_name,
    )
