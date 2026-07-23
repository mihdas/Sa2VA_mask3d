from typing import Optional, Tuple

import numpy as np
from PIL import Image


def dynamic_preprocess_mask(
    mask, min_num=1, max_num=6, image_size=448, use_thumbnail=False, threshold=0.5
):
    """
    Dynamically generate multiple binary masks from a single binary mask.
    Uses bilinear interpolation for smoother resizing, then thresholds to maintain binary values.

    Args:
        mask: Binary mask as numpy array (H, W) or PIL Image
        min_num: Minimum number of output masks
        max_num: Maximum number of output masks
        image_size: Size of each output mask
        use_thumbnail: Whether to add a thumbnail of the original mask
        threshold: Threshold for binarizing after interpolation (0-1)

    Returns:
        List of binary masks as numpy arrays
    """
    # Convert to PIL Image if it's a numpy array
    if isinstance(mask, np.ndarray):
        # Ensure mask is binary (0 and 1 or 0 and 255)
        if mask.max() > 1:
            mask_norm = (mask > 127).astype(np.uint8)
        else:
            mask_norm = mask.astype(np.uint8)
        mask_pil = Image.fromarray(mask_norm * 255)  # Convert to 0-255 for PIL
    else:
        mask_pil = mask
        # Ensure it's in the right mode
        if mask_pil.mode != "L":
            mask_pil = mask_pil.convert("L")

    orig_width, orig_height = mask_pil.size
    aspect_ratio = orig_width / orig_height

    # Calculate target ratios (same logic as original)
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the mask using bilinear interpolation for smoother edges
    resized_mask = mask_pil.resize((target_width, target_height), Image.BILINEAR)

    processed_masks = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # Split the mask
        split_mask = resized_mask.crop(box)
        # Convert to numpy array and apply threshold to maintain binary values
        split_mask_np = np.array(split_mask, dtype=np.float32) / 255.0  # Convert to 0-1
        split_mask_binary = (split_mask_np > threshold).astype(np.uint8)
        processed_masks.append(split_mask_binary)

    assert len(processed_masks) == blocks

    if use_thumbnail and len(processed_masks) != 1:
        # Create thumbnail using bilinear interpolation
        thumbnail_mask = mask_pil.resize((image_size, image_size), Image.BILINEAR)
        thumbnail_np = np.array(thumbnail_mask, dtype=np.float32) / 255.0
        thumbnail_binary = (thumbnail_np > threshold).astype(np.uint8)
        processed_masks.append(thumbnail_binary)

    return processed_masks


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=6, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def select_frames(
    vid_len: int,
    num_frames: int,
    tarvis_num_frames: int,
    train_mode: Optional[bool] = True,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Select frames for CLIP processing and TarViS context.

    Returns:
        sample_indx: Indices for CLIP context frames.
        tarvis_sample_indx: Indices for TarViS context frames.
        context_num_frames: Number of frames for CLIP context.
        tarvis_context_num_frames: Number of frames for TarViS context.
    """
    if vid_len <= num_frames:
        # Use all frames if fewer than num_frames
        sample_indx = np.arange(vid_len)
        context_num_frames = vid_len
        tarvis_sample_indx = np.arange(vid_len)
        tarvis_context_num_frames = vid_len
    else:
        # Sample frames when video is longer
        context_num_frames = num_frames
        sample_indx = sample_frames(num_frames, vid_len)

        # TarViS also gets sampled frames
        if vid_len <= tarvis_num_frames:
            tarvis_sample_indx = np.arange(vid_len)
            tarvis_context_num_frames = vid_len
        if tarvis_num_frames == num_frames:
            tarvis_sample_indx = sample_indx
            tarvis_context_num_frames = context_num_frames
        else:
            tarvis_sample_indx = sample_frames(tarvis_num_frames, vid_len)
            tarvis_context_num_frames = tarvis_num_frames

    if not train_mode:
        # For inference: use evenly spaced frames
        sample_indx = np.linspace(0, vid_len - 1, context_num_frames).astype(int)
        tarvis_sample_indx = np.linspace(
            0, vid_len - 1, tarvis_context_num_frames
        ).astype(int)

    return (
        sample_indx,
        tarvis_sample_indx,
        context_num_frames,
        tarvis_context_num_frames,
    )


def sample_frames(context_num_frames: int, vid_len: int) -> np.ndarray:
    """
    Jittered frame sampling strategy.

    Returns:
        Numpy array of selected frame indices.
    """
    padding = (vid_len / context_num_frames) / 2
    base_indices = np.linspace(
        padding - 0.5, vid_len - 1 - (padding - 0.5), context_num_frames
    )
    interval = (vid_len - 2 * padding) / (context_num_frames - 1)
    fixed_offset = np.random.uniform(-interval / 2, interval / 2)
    sample_idx = base_indices + fixed_offset
    return np.round(sample_idx).astype(int)
