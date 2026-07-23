import math

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import (
    resize,  # type: ignore
    to_pil_image,
)


class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode="RGB")
        return np.array(img.resize((self.target_length, self.target_length)))


class AspectRatioPatchResize:
    """
    Resize images while preserving aspect ratio and ensuring patch size compatibility.
    This is designed for images that will be processed by PI3 spatial model and then
    matched to VIT features in the update_features function.
    """

    def __init__(
        self,
        max_pixels: int = 255000,
        patch_size: int = 14,
        target_size: tuple = None,
    ) -> None:
        """
        Args:
            max_pixels (int): Maximum number of pixels allowed (default: 255000)
            patch_size (int): Size of VIT patches (default: 14)
            target_size (tuple): Optional target size (width, height). If provided,
                               max_pixels constraint will still be enforced.
        """
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.target_size = target_size

    def _calculate_target_size(
        self, original_width: int, original_height: int
    ) -> tuple:
        """Calculate target dimensions that are patch-compatible."""
        # If target_size is provided, use it but ensure patch compatibility and max_pixels
        if self.target_size is not None:
            TARGET_W, TARGET_H = self.target_size
            # Ensure target size doesn't exceed max_pixels
            if TARGET_W * TARGET_H > self.max_pixels:
                scale = math.sqrt(self.max_pixels / (TARGET_W * TARGET_H))
                TARGET_W = int(TARGET_W * scale)
                TARGET_H = int(TARGET_H * scale)
        else:
            # Calculate scaling factor to fit within max_pixels while preserving aspect ratio
            if original_width * original_height > self.max_pixels:
                scale = math.sqrt(self.max_pixels / (original_width * original_height))
                W_scaled, H_scaled = original_width * scale, original_height * scale
            else:
                W_scaled, H_scaled = original_width, original_height

            # Round to nearest patch multiples
            k = max(1, round(W_scaled / self.patch_size))
            m = max(1, round(H_scaled / self.patch_size))

            # Ensure we don't exceed max_pixels
            while (k * self.patch_size) * (m * self.patch_size) > self.max_pixels:
                if k / m > W_scaled / H_scaled:
                    k = max(1, k - 1)
                else:
                    m = max(1, m - 1)

            TARGET_W, TARGET_H = k * self.patch_size, m * self.patch_size

        return TARGET_W, TARGET_H

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        Returns resized image as numpy array.
        """
        img = to_pil_image(image, mode="RGB")
        original_width, original_height = img.size

        target_width, target_height = self._calculate_target_size(
            original_width, original_height
        )

        resized_img = img.resize(
            (target_width, target_height), Image.Resampling.LANCZOS
        )
        return np.array(resized_img)


def resize_pi3_patches_to_vit(
    pi3_patches: torch.Tensor,
    vit_patch_count: int,
    pi3_height: int = None,
    pi3_width: int = None,
) -> torch.Tensor:
    """
    Resize PI3 patch features to match VIT patch dimensions.

    Args:
        pi3_patches (torch.Tensor): PI3 patch features with shape [B, N_pi3, C]
        vit_patch_count (int): Number of patches in VIT features (e.g., 256 for 16x16)
        pi3_height (int, optional): Height of PI3 patch grid. If None, assumes square.
        pi3_width (int, optional): Width of PI3 patch grid. If None, assumes square.

    Returns:
        torch.Tensor: Resized PI3 patches with shape [B, vit_patch_count, C]
    """
    B, N_pi3, C = pi3_patches.shape

    # Calculate target VIT grid dimensions (assuming square VIT)
    vit_side = int(math.sqrt(vit_patch_count))
    vit_height = vit_width = vit_side

    # Calculate PI3 grid dimensions
    if pi3_height is None or pi3_width is None:
        # Assume square PI3 if dimensions not provided
        pi3_side = int(math.sqrt(N_pi3))
        pi3_height = pi3_width = pi3_side
    else:
        # Verify provided dimensions match total patch count
        assert (
            pi3_height * pi3_width == N_pi3
        ), f"PI3 dimensions {pi3_height}x{pi3_width} don't match patch count {N_pi3}"

    # Reshape to spatial format: [B, C, H, W]
    pi3_spatial = pi3_patches.permute(0, 2, 1).reshape(B, C, pi3_height, pi3_width)

    # Interpolate to VIT dimensions
    resized_spatial = torch.nn.functional.interpolate(
        pi3_spatial, size=(vit_height, vit_width), mode="bilinear", align_corners=False
    )

    # Reshape back to patch format: [B, vit_patch_count, C]
    resized_patches = resized_spatial.reshape(B, C, vit_patch_count).permute(0, 2, 1)

    return resized_patches


def resize_vit_patches_to_pi3(
    vit_patches: torch.Tensor,
    pi3_patches_shape: tuple,
    pi3_height: int = None,
    pi3_width: int = None,
) -> torch.Tensor:
    """
    Resize VIT patch features to match PI3 patch dimensions.

    Args:
        vit_patches (torch.Tensor): VIT patch features with shape [B, N_vit, C]
        pi3_patches_shape (tuple): Shape of PI3 patches (B, N_pi3, C)
        pi3_height (int, optional): Height of PI3 patch grid. If None, assumes square.
        pi3_width (int, optional): Width of PI3 patch grid. If None, assumes square.

    Returns:
        torch.Tensor: Resized VIT patches with shape [B, N_pi3, C]
    """
    B_vit, N_vit, C_vit = vit_patches.shape
    B_pi3, N_pi3, C_pi3 = pi3_patches_shape

    # Ensure batch sizes match
    assert B_vit == B_pi3, f"Batch sizes don't match: VIT={B_vit}, PI3={B_pi3}"

    # Calculate VIT grid dimensions (assuming square VIT)
    vit_side = int(math.sqrt(N_vit))
    vit_height = vit_width = vit_side

    # Calculate PI3 grid dimensions
    if pi3_height is None or pi3_width is None:
        # Assume square PI3 if dimensions not provided
        pi3_side = int(math.sqrt(N_pi3))
        pi3_height = pi3_width = pi3_side
    else:
        # Verify provided dimensions match total patch count
        assert (
            pi3_height * pi3_width == N_pi3
        ), f"PI3 dimensions {pi3_height}x{pi3_width} don't match patch count {N_pi3}"

    # Reshape to spatial format: [B, C, H, W]
    vit_spatial = vit_patches.permute(0, 2, 1).reshape(
        B_vit, C_vit, vit_height, vit_width
    )

    # Interpolate to PI3 dimensions
    resized_spatial = torch.nn.functional.interpolate(
        vit_spatial, size=(pi3_height, pi3_width), mode="bilinear", align_corners=False
    )

    # Reshape back to patch format: [B, N_pi3, C]
    resized_patches = resized_spatial.reshape(B_vit, C_vit, N_pi3).permute(0, 2, 1)

    return resized_patches


def calculate_patch_grid_sizes(
    vit_patch_count: int,
    pi3_patch_count: int,
    pi3_height: int = None,
    pi3_width: int = None,
) -> tuple:
    """
    Calculate the grid dimensions for VIT and PI3 patches.

    Args:
        vit_patch_count (int): Number of VIT patches
        pi3_patch_count (int): Number of PI3 patches
        pi3_height (int, optional): Height of PI3 patch grid. If None, assumes square.
        pi3_width (int, optional): Width of PI3 patch grid. If None, assumes square.

    Returns:
        tuple: (vit_height, vit_width, pi3_height, pi3_width) grid dimensions
    """
    # Calculate VIT grid dimensions (assuming square VIT)
    vit_side = int(math.sqrt(vit_patch_count))
    vit_height = vit_width = vit_side

    # Calculate PI3 grid dimensions
    if pi3_height is None or pi3_width is None:
        # Assume square PI3 if dimensions not provided
        pi3_side = int(math.sqrt(pi3_patch_count))
        pi3_height = pi3_width = pi3_side
    else:
        # Verify provided dimensions match total patch count
        assert (
            pi3_height * pi3_width == pi3_patch_count
        ), f"PI3 dimensions {pi3_height}x{pi3_width} don't match patch count {pi3_patch_count}"

    return vit_height, vit_width, pi3_height, pi3_width
