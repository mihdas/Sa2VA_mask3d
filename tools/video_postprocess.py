import argparse
import logging
import random
import re
import shutil
import tempfile
from pathlib import Path

import torch.distributed as dist
from inference.run_on_video import run_on_video
from PIL import Image
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def are_masks_valid(mask_files):
    for mask_file in mask_files:
        with Image.open(str(mask_file)) as img:
            if any(px != 0 for px in img.getdata()):
                return True, mask_file
    return False, None


def process_single_video(video_path: Path, masks_path: Path, output_path: Path):
    """
    Processes a single video with its corresponding masks and generates segmentation masks and overlays.
    Args:
        video_path (Path): Path to the video file or directory containing video frames.
        masks_path (Path): Path to the directory containing mask images.
        output_path (Path): Path to the directory where outputs will be saved.
    """
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Using masks from: {masks_path}")
    logger.info(f"Output will be saved to: {output_path}")

    # Collect frames with masks
    #I need indices, not ids
    frames_with_masks = sorted([int(re.search(r"\d+", p.stem).group()) for p in masks_path.glob("*.png")])
    frames = sorted([int(re.search(r"\d+", p.stem).group()) for p in video_path.glob("*")])
    mask_indices = []
    for mask_frame_id in frames_with_masks:
        try:
            mask_indices.append(frames.index(mask_frame_id))
        except ValueError:
            print(f"Warning: Mask frame ID {mask_frame_id} not found in video frames.") # Changed to warning
            #Optionally, you could skip this frame id.
            #pass
    frames_with_masks = mask_indices
    # frames_with_masks = [int(re.search(r"\d+", p.stem).group()) for p in sorted(masks_path.iterdir()) if p.is_file() and p.suffix == ".png"]

    # frames_with_masks = []
    # for file_path in (
    #     p for p in masks_path.iterdir() if p.is_file() and p.suffix == ".png"
    # ):
    #     frame_number_match = re.search(r"\d+", file_path.stem)
    #     if frame_number_match is None:
    #         logger.error(
    #             f"ERROR: File {file_path} does not contain a frame number. Cannot load it as a mask."
    #         )
    #         raise ValueError(f"Invalid mask filename: {file_path.name}")
    #     frames_with_masks.append(int(frame_number_match.group()))

    logger.info(f"Using masks for frames: {sorted(frames_with_masks)}")

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Call the run_on_video function directly
    try:
        run_on_video(
            str(video_path), str(masks_path), str(output_path), frames_with_masks, print_progress=False
        )
        logger.info(f"run_on_video completed for {video_path}")
    except Exception as e:
        logger.error(f"Failed to process video {video_path}: {e}")
        raise


def process_video(expression_folder: Path, base_path: Path, output_path: Path):
    """
    Processes a single expression folder by processing its corresponding video.
    Args:
        expression_folder (Path): Path to the folder containing mask annotations for a specific expression.
        base_path (Path): Base path where the video frames are stored.
        output_path (Path): Base output path where results will be saved.
    """
    if "revos" in str(base_path):
        folder_id = "/".join([
            expression_folder.parent.parent.parent.name,  # Third last folder
            expression_folder.parent.parent.name,        # Second last folder
            expression_folder.parent.name                # Last folder
        ])
    else:
        folder_id = expression_folder.parent.name
    expression = expression_folder.name
    video_folder = base_path / "JPEGImages" / folder_id
    masks_folder = expression_folder

    if not video_folder.exists():
        logger.error(f"Video folder {video_folder} does not exist. Skipping.")
        return

    # Count the number of frames in the video folder (input)
    num_video_frames = len(list(video_folder.glob("*")))

    if num_video_frames == 0:
        logger.error(f"No frames found in {video_folder}. Skipping.")
        return

    final_output_folder = output_path / folder_id / expression

    # If output folder exists, check if it already has the correct number of files
    if final_output_folder.exists():
        existing_output_files = list(final_output_folder.glob("*.png"))
        num_output_files = len(existing_output_files)
        if num_output_files == num_video_frames:
            logger.info(
                f"Output already has the correct number of files for {expression_folder} ({num_output_files} files). Skipping."
            )
            return
        else:
            logger.warning(
                f"Output folder exists for {expression_folder} but does not have the correct number of files ({num_output_files}/{num_video_frames}). Will re-process."
            )

    mask_files = list(expression_folder.glob("*.png"))  # Assuming masks are PNG files
    has_valid_mask, first_valid_mask_path = are_masks_valid(mask_files)

    if len(mask_files) == num_video_frames:
        logger.info(
            f"Number of input masks ({len(mask_files)}) matches the number of video frames ({num_video_frames}). Copying directly..."
        )
        final_output_folder.mkdir(parents=True, exist_ok=True)
        for mf in mask_files:
            shutil.copy2(mf, final_output_folder / mf.name)
        logger.info(f"Successfully copied masks to {final_output_folder}")
        return

    if not has_valid_mask:
        logger.warning(
            f"No valid masks found for {expression_folder}. Generating empty or copying existing masks."
        )
        final_output_folder.mkdir(parents=True, exist_ok=True)
        image_frames = list(video_folder.glob("*"))
        image_shape = Image.open(image_frames[0]).size

        for mask_name in list(video_folder.glob("*")):
            mask_name = mask_name.stem + ".png"
            new_mask_path = final_output_folder / mask_name
            empty_mask = Image.new("L", image_shape)
            empty_mask.save(new_mask_path)
        return

    # If we have valid masks, run the processing
    with tempfile.TemporaryDirectory() as temp_output_dir:
        temp_output_folder = Path(temp_output_dir)
        try:
            # Process the video directly
            process_single_video(video_folder, masks_folder, temp_output_folder)
        except Exception as e:
            logger.error(f"Error processing {expression_folder}: {e}")
            return

        # Now check generated masks
        temp_masks_folder = temp_output_folder / "masks"
        num_mask_files = len(list(temp_masks_folder.glob("*")))
        logger.debug(f"Number of mask files generated: {num_mask_files}")

        # If counts match, move files from temp to final output folder
        if num_mask_files == num_video_frames:
            final_output_folder.mkdir(parents=True, exist_ok=True)
            for file in temp_masks_folder.iterdir():
                if file.is_file():
                    shutil.move(str(file), str(final_output_folder))
            logger.info(f"Processed and moved masks for {expression_folder}.")
        else:
            logger.error(
                f"Frame count mismatch for {expression_folder}. "
                f"Expected {num_video_frames}, got {num_mask_files}."
            )


def process_videos(
    base_path: Path,
    prediction_path: Path,
    output_path: Path,
    expression_folders: list,  # Pass the list of folders directly
):
    """
    Processes a subset of expression folders.
    Args:
        base_path (Path): Base path where the video frames are stored.
        prediction_path (Path): Path to the directory containing prediction/mask folders.
        output_path (Path): Base output path where results will be saved.
        expression_folders (list): List of expression folders to process.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Divide the folders among the processes
    num_folders = len(expression_folders)
    folders_per_process = num_folders // world_size
    remainder = num_folders % world_size

    start_idx = rank * folders_per_process + min(rank, remainder)
    end_idx = (rank + 1) * folders_per_process + min(
        rank + 1, remainder
    )  # Exclusive end index

    subset_folders = expression_folders[start_idx:end_idx]

    logger.info(
        f"Rank {rank}: Processing folders {start_idx} to {end_idx} out of {num_folders} total."
    )

    # Shuffle to distribute workload evenly within each process
    random.shuffle(subset_folders)

    with tqdm(
        total=num_folders,
        desc=f"Rank {rank}: Processing videos",
        position=rank,  # Ensure tqdm doesn't overwrite
        leave=False,
        # disable=not (rank == 0),  # Only show progress bar for rank 0
    ) as pbar:
        pbar.update(start_idx)  # start the progress bar at the correct position
        for expression_folder in subset_folders:
            try:
                process_video(expression_folder, base_path, output_path)
            except Exception as exc:
                logger.error(
                    f"Rank {rank}: Generated an exception for {expression_folder}: {exc}"
                )
            finally:
                pbar.update(1)

    logger.info(f"Rank {rank}: Finished processing videos.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Batch process videos with mask annotations."
    )
    parser.add_argument(
        "--base_video_path",
        type=str,
        default="/nodes/cristal/work/nekrasov/data/language-data/mevis/valid",
        help="Path to the base video directory containing 'JPEGImages'.",
    )
    parser.add_argument(
        "--prediction_path",
        type=str,
        default="input",
        help="Name of the input folder within experiment_path containing masks.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="xmem_prediction",
        help="Name of the output folder within experiment_path for predictions.",
    )
    # Removed folders_per_job
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank of the process (used by torchrun).",
    )
    return parser.parse_args()


def main(rank, world_size):
    args = parse_arguments()
    base_video_path = Path(args.base_video_path)
    prediction_path = Path(args.prediction_path)
    output_path = Path(args.output_path)

    # Validate paths
    if not base_video_path.exists():
        logger.error(f"Base video path does not exist: {base_video_path}")
        exit(1)
    if not prediction_path.exists():
        logger.error(f"Prediction path does not exist: {prediction_path}")
        exit(1)

    # Get the list of expression folders
    if "revos" in str(base_video_path):
        expression_folders = sorted(list(prediction_path.glob("*/*/*/*")))
    else:
        expression_folders = sorted(list(prediction_path.glob("*/*")))
    num_folders = len(expression_folders)

    if num_folders == 0:
        logger.error(f"No expression folders found in {prediction_path}.")
        exit(1)

    logger.info(f"Rank {rank}: Found {num_folders} expression folders.")

    # Process videos using the updated function
    process_videos(
        base_path=base_video_path,
        prediction_path=prediction_path,
        output_path=output_path,
        expression_folders=expression_folders,  # Pass the list of folders
    )


if __name__ == "__main__":
    # Initialize distributed environment
    dist.init_process_group(backend="nccl")  # or "gloo" if NCCL is not available
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Call the main function
    main(rank, world_size)

    # Clean up distributed environment
    dist.destroy_process_group()
