import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import time
from glob import glob
import os
from PIL import Image
import numpy as np

class RGBDepthDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, transform=None):
        self.rgb_paths = sorted(glob(os.path.join(rgb_dir, '*.png')))
        self.depth_paths = sorted(glob(os.path.join(depth_dir, '*.png')))
        self.transform = transform

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_paths[idx]).convert('RGB')
        depth = Image.open(self.depth_paths[idx]).convert('L')

        if self.transform:
            rgb = self.transform(rgb)
            depth = self.transform(depth)

        return rgb, depth

class NYUDepthV2MatDataset(Dataset):
    def __init__(self, mat_file_path, transform=None, max_depth_percentile=99.9, img_key='images', depth_key='depths'):

        self.mat_file_path = mat_file_path
        self.transform = transform # Applied in __getitem__
        self.img_key = img_key
        self.depth_key = depth_key

        print(f"Loading NYU Depth V2 data from: {mat_file_path}")

        with h5py.File(self.mat_file_path, 'r') as f:
            self.images = np.array(f[self.img_key]).transpose(3, 0, 1, 2)
            self.depths = np.array(f[self.depth_key]).transpose(2, 0, 1)
        print(f"Loaded {len(self.images)} samples.")

        valid_depths = self.depths[self.depths > 1e-6]
        if valid_depths.size > 0:
            # Calculate percentile and store as a standard float
            self.max_depth = float(np.percentile(valid_depths, max_depth_percentile))
        else:
            self.max_depth = 10.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get raw data for the index
        rgb_raw = self.images[idx]
        depth_raw = self.depths[idx]

        # Convert to PIL Images
        rgb_pil = Image.fromarray(rgb_raw, mode='RGB')
        depth_pil = Image.fromarray(depth_raw.astype(np.float32), mode='F')

        # Apply transformations (should include ToTensor)
        if self.transform:
            rgb_tensor = self.transform(rgb_pil)
            depth_tensor = self.transform(depth_pil) # Tensor [1, H, W] with original values

        # Ensure tensor and squeeze channel dim for processing
        if not isinstance(depth_tensor, torch.Tensor): raise TypeError("Depth is not a Tensor after transform.")
        if depth_tensor.dim() == 3 and depth_tensor.shape[0] == 1: depth_tensor = depth_tensor.squeeze(0) # Shape [H, W]
        elif depth_tensor.dim() != 2: raise ValueError(f"Unexpected depth tensor shape: {depth_tensor.shape}")


        # --- Use self.max_depth calculated in __init__ ---
        # 1. Calculate valid mask based on original values
        valid_mask = (depth_tensor > 1e-6) & (depth_tensor <= self.max_depth) # Shape [H, W]

        # 2. Clamp depth tensor
        depth_clamped = torch.clamp(depth_tensor, min=0.0, max=self.max_depth) # Use pre-calculated float self.max_depth

        # 3. Normalize clamped depth
        depth_normalized = depth_clamped / self.max_depth # Range [0, 1]

        # 4. Apply mask (set invalid regions to 0 in normalized depth)
        depth_normalized[~valid_mask] = 0.0

        # 5. Add channel dimension back for consistency [1, H, W]
        depth_normalized = depth_normalized.unsqueeze(0)
        valid_mask = valid_mask.unsqueeze(0) # Also give mask channel dim [1, H, W]

        # 6. Return all three: rgb, normalized depth, and the mask
        return rgb_tensor, depth_normalized