import os
from glob import glob
from typing import Callable, Optional, Tuple, List, Dict

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RGBDepthDataset(Dataset):
    """
    Class for handling RGB/Depth datasets
    
    Args:
        rgb_paths: list of paths to rgb images
        depth_paths: list of paths to depth images
        transform_rgb: transform to apply to rgb images
        transform_depth: transform to apply to depth images
    """
    def __init__(
        self,
        rgb_dir: str,
        depth_dir: str,
        transform_rgb: Optional[Callable] = None,
        transform_depth: Optional[Callable] = None,
    ) -> None:
        """
        Initializes RGBDepthDataset.

        Args:
            rgb_dir: directory with rgb images
            depth_dir: directory with depth images
            transform_rgb: transform to apply to rgb images
            transform_depth: transform to apply to depth images
        """
        self.rgb_paths = sorted(glob(os.path.join(rgb_dir, "*.png")))
        self.depth_paths = sorted(glob(os.path.join(depth_dir, "*.png")))
        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth

    def __len__(self) -> int:
        """Return number of rgb images."""
        return len(self.rgb_paths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        """Get RGB/Depth images.
        
        Args:
            idx: index of image to get
        
        Returns:
            rgb: rgb image
            depth: depth image
        """
        rgb = Image.open(self.rgb_paths[idx]).convert("RGB")
        depth = Image.open(self.depth_paths[idx]).convert("L")

        if self.transform_rgb:
            rgb = self.transform_rgb(rgb)
        if self.transform_depth:
            depth = self.transform_depth(depth)

        return rgb, depth


class NYUDepthV2MatDataset(Dataset):
    """
    Class for handling NYU Depth V2 dataset.
    
    Args:
        mat_file_path: path to mat files
        transform_rgb: transform to apply to rgb images
        transform_depth: transform to apply to depth images
        img_key: key to get rgb images
        depth_key: key to get depth images
        images: rgb images
        depths: depth images
        max_depth: max depth normalization
    """
    def __init__(
        self,
        mat_file_path: str,
        transform_rgb: Optional[Callable] = None,
        transform_depth: Optional[Callable] = None,
        max_depth_percentile: float = 99.9,
        img_key: str = "images",
        depth_key: str = "depths",
    ) -> None:
        """
        Dataset handling for NYU Depth V2.
        
        Args:
            mat_file_path: path to mat files
            transform_rgb: transform to apply to rgb images
            transform_depth: transform to apply to depth images
            max_depth_percentile: percentil for max depth normalization
            img_key: key to get rgb images
            depth_key: key to get depth images
        """
        self.mat_file_path = mat_file_path
        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth
        self.img_key = img_key
        self.depth_key = depth_key

        print(f"Loading NYU Depth V2 data from: {mat_file_path}")

        with h5py.File(self.mat_file_path, "r") as f:
            self.images = np.array(f[self.img_key]).transpose(0, 3, 2, 1)
            self.depths = np.array(f[self.depth_key]).transpose(0, 2, 1)
        self.images = self.images
        self.depths = self.depths
        print(f"Loaded {len(self.images)} samples.")

        valid_depths = self.depths[self.depths > 1e-6]
        if valid_depths.size > 0:
            # Calculate percentile and store as a standard float
            self.max_depth = float(np.percentile(valid_depths, max_depth_percentile))
        else:
            self.max_depth = 10.0

    def __len__(self) -> int:
        """Get number of images."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get rgb and depth images.
        
        Args:
            idx: index of image to get
        
        Returns:
            rgb_tensor: rgb image
            depth_normalized: normalized depth image
        """
        # Get raw data for the index
        rgb_raw = self.images[idx]
        depth_raw = self.depths[idx]

        # Convert to PIL Images
        rgb_pil = Image.fromarray(rgb_raw, mode="RGB")
        depth_pil = Image.fromarray(depth_raw.astype(np.float32), mode="F")

        if self.transform_rgb:
            rgb_tensor = self.transform_rgb(rgb_pil)
        if self.transform_depth:
            depth_tensor = self.transform_rgb(depth_pil)

        # Ensure tensor and squeeze channel dim for processing
        if not isinstance(depth_tensor, torch.Tensor):
            raise TypeError("Depth is not a Tensor after transform.")
        if depth_tensor.dim() == 3 and depth_tensor.shape[0] == 1:
            depth_tensor = depth_tensor.squeeze(0)  # Shape [H, W]
        elif depth_tensor.dim() != 2:
            raise ValueError(f"Unexpected depth tensor shape: {depth_tensor.shape}")

        # --- Use self.max_depth calculated in __init__ ---
        # 1. Calculate valid mask based on original values
        valid_mask = (depth_tensor > 1e-6) & (
            depth_tensor <= self.max_depth
        )  # Shape [H, W]

        # 2. Clamp depth tensor
        depth_clamped = torch.clamp(
            depth_tensor, min=0.0, max=self.max_depth
        )  # Use pre-calculated float self.max_depth

        # 3. Normalize clamped depth
        depth_normalized = depth_clamped / self.max_depth  # Range [0, 1]

        # 4. Apply mask (set invalid regions to 0 in normalized depth)
        depth_normalized[~valid_mask] = 0.0

        # 5. Add channel dimension back for consistency [1, H, W]
        depth_normalized = depth_normalized.unsqueeze(0)
        valid_mask = valid_mask.unsqueeze(0)  # Also give mask channel dim [1, H, W]

        # 6. Return all three: rgb, normalized depth, and the mask
        return rgb_tensor, depth_normalized


def read_split_file(filepath: str) -> List[Dict[str, str]]:
    """
    Reads a Monodepth2 split file and returns a list of samples.

    Args:
        filepath: path to split file

    Returns:
        list of dicts with folders, frame indices, and sides
    """
    with open(filepath, "r") as f:
        lines = f.read().splitlines()

    samples = []
    for line in lines:
        parts = line.split()
        if len(parts) == 3:
            folder, frame_index, side = parts
            samples.append(
                {
                    "folder": folder,
                    "frame_index": int(frame_index),
                    "side": side,  # 'l' or 'r'
                }
            )
        elif len(parts) == 2:  # Handle cases where side might be missing (assume left?)
            folder, frame_index = parts
            samples.append(
                {
                    "folder": folder,
                    "frame_index": int(frame_index),
                    "side": "l",  # Default to left if side is missing
                }
            )
        else:
            print(f"Skipping malformed line: {line}")

    return samples