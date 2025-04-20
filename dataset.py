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
    def __init__(self, mat_file_path, transform=None, max_depth_eval=10.0, img_key='images', depth_key='depths'):
        self.mat_file_path = mat_file_path
        self.transform = transform
        self.max_depth_eval = max_depth_eval
        self.img_key = img_key
        self.depth_key = depth_key

        print(f"Loading NYU Depth V2 data from: {mat_file_path}")

        with h5py.File(self.mat_file_path, 'r') as f:
            self.images = np.array(f[self.img_key]).transpose(3, 2, 0, 1) # N C H W (assuming original C=3 is axis 2)
            self.depths = np.array(f[self.depth_key]).transpose(2, 0, 1) # N H W
        print(f"Loaded {len(self.images)} samples.")
        with h5py.File(self.mat_file_path, 'r') as f:
                self.images_raw = np.array(f[self.img_key]).transpose(3, 1, 0, 2) # N H W C
                self.depths_raw = np.array(f[self.depth_key]).transpose(2, 1, 0) # N H W
        print(f"Loaded {len(self.images_raw)} samples.")

    def __len__(self):
        return len(self.images_raw)

    def __getitem__(self, idx):
        rgb_raw_hwc = self.images_raw[idx] # Shape (H, W, C)
        depth_raw_hw = self.depths_raw[idx] # Shape (H, W)

        rgb_pil = Image.fromarray(rgb_raw_hwc, mode='RGB')
        depth_pil = Image.fromarray(depth_raw_hw.astype(np.float32), mode='F')

        if self.transform:
            rgb_tensor = self.transform(rgb_pil)
            depth_tensor_transformed = self.transform(depth_pil)
        depth_tensor_original_scale = depth_tensor_transformed.float() # Ensure float32

        valid_mask = (depth_tensor_original_scale > 1e-6) & (depth_tensor_original_scale <= self.max_depth_eval) # Shape [1, H, W]

        return rgb_tensor, depth_tensor_original_scale, valid_mask

def read_split_file(filepath):
    """Reads a Monodepth2 split file and returns a list of samples."""
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
    
    samples = []
    for line in lines:
        parts = line.split()
        if len(parts) == 3:
            folder, frame_index, side = parts
            samples.append({
                "folder": folder, 
                "frame_index": int(frame_index), 
                "side": side # 'l' or 'r'
            })
        elif len(parts) == 2: # Handle cases where side might be missing (assume left?)
             folder, frame_index = parts
             samples.append({
                "folder": folder, 
                "frame_index": int(frame_index), 
                "side": 'l' # Default to left if side is missing
            })
        else:
            print(f"Skipping malformed line: {line}")
            
    return samples

class KITTIRGBDepthDataset(Dataset):
    def __init__(self, data_path, split_file, transform=None, depth_subfolder="gt_depths"):
        super().__init__()
        self.data_path = data_path
        self.samples = read_split_file(split_file)
        self.transform = transform
        self.depth_subfolder = depth_subfolder
        
            
        print(f"{len(self.samples)} training samples.")

    def __len__(self):
        return len(self.samples)

    def get_rgb_path(self, sample_info):
        """Constructs the path to the RGB image."""
        # Camera ID 2 is left, 3 is right
        camera_id = 2 if sample_info["side"] == 'l' else 3 
        folder = sample_info["folder"]
        frame_index = sample_info["frame_index"]
        # Path: data_path / date_drive / image_0X / data / frame.png
        rgb_path = os.path.join(
            self.data_path,
            folder,
            f"image_0{camera_id}",
            "data",
            f"{frame_index:010d}.png"
        )
        return rgb_path

    def get_depth_path(self, sample_info):
         # Camera ID 2 is left, 3 is right - depth usually corresponds to left (image_02)
        camera_id_for_depth = 2 # Typically depth is generated for the left camera
        folder = sample_info["folder"]
        frame_index = sample_info["frame_index"]
        # Path: data_path / date_drive / depth_subfolder / image_02 / frame.png
        depth_path = os.path.join(
            self.data_path,
            folder,
            self.depth_subfolder, # e.g., "gt_depths" or "proj_depth/groundtruth"
            f"image_0{camera_id_for_depth}",
            f"{frame_index:010d}.png" 
        )
        return depth_path

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        rgb_path = self.get_rgb_path(sample_info)
        depth_path = self.get_depth_path(sample_info)

        rgb = Image.open(rgb_path).convert('RGB')
        
        # Load depth map (expected to be uint16 PNG)
        depth_png = Image.open(depth_path) 
        depth = np.array(depth_png, dtype=np.float32)

        mask = depth > 0
        depth = depth / 256.0 
        depth[~mask] = 0.0 # Set invalid depth to 0, or np.nan if preferred

        depth_pil = Image.fromarray(depth, mode='F')


        if self.transform:
            rgb_transformed = self.transform(rgb)
            depth_transformed = self.transform(depth_pil)
        return rgb_transformed, depth_transformed


class KITTIBenchmarkDataset(Dataset):
    def __init__(self, data_path, split_file, transform=None, max_depth_eval=80.0, 
                 depth_gt_subfolder="depth_maps/groundtruth", # Common location for Eigen GT
                 garg_crop=True, eigen_crop=False):
        super().__init__()
        self.data_path = data_path
        self.samples = read_split_file(split_file)
        self.transform = transform # Usually applied only to RGB for evaluation
        self.max_depth_eval = max_depth_eval
        self.depth_gt_subfolder = depth_gt_subfolder

        print(f"{len(self.samples)} benchmark samples.")
        print(f"Expecting GT depth maps in: {self.depth_gt_subfolder}")
        print(f"Max evaluation depth: {self.max_depth_eval}m")


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def get_rgb_path(self, sample_info):
        """Constructs the path to the RGB image."""
        # Camera ID 2 is left, 3 is right
        camera_id = 2 if sample_info["side"] == 'l' else 3
        # Split folder like '2011_09_26/2011_09_26_drive_0001_sync'
        date_folder = sample_info["folder"].split('/')[0] 
        drive_folder = sample_info["folder"]
        frame_index = sample_info["frame_index"]
        # Path: data_path / date / date_drive / image_0X / data / frame.png
        rgb_path = os.path.join(
            self.data_path,
            date_folder, # Need the date folder level
            drive_folder,
            f"image_0{camera_id}",
            "data",
            f"{frame_index:010d}.png"
        )
        return rgb_path

    def get_depth_gt_path(self, sample_info):
        # Camera ID 2 is left, 3 is right - GT depth usually corresponds to left (image_02)
        camera_id_for_depth = 2 
        # Split folder like '2011_09_26/2011_09_26_drive_0001_sync'
        date_folder = sample_info["folder"].split('/')[0]
        drive_folder = sample_info["folder"]
        frame_index = sample_info["frame_index"]
        # Path: data_path / depth_gt_subfolder / date_drive / image_02 / frame.png
        depth_path = os.path.join(
            self.data_path,
            self.depth_gt_subfolder, 
            drive_folder,
            f"image_0{camera_id_for_depth}",
            f"{frame_index:010d}.png"
        )
        return depth_path
        
    def apply_crop(self, img_or_depth_array):
        """Applies the specified crop to a numpy array (H, W, C) or (H, W)."""
        h, w = img_or_depth_array.shape[:2]

        crop_mask = np.zeros_like(img_or_depth_array, dtype=bool)
        if img_or_depth_array.ndim == 2: # Depth map
            crop_mask[int(0.40810811 * h):int(0.99189189 * h),
                        int(0.03594771 * w):int(0.96405229 * w)] = True
        else: # RGB image
                crop_mask[int(0.40810811 * h):int(0.99189189 * h),
                        int(0.03594771 * w):int(0.96405229 * w), :] = True
        if img_or_depth_array.ndim == 2: # Depth map
                cropped = img_or_depth_array[int(0.40810811 * h):int(0.99189189 * h),
                                            int(0.03594771 * w):int(0.96405229 * w)]
        else: # RGB - don't crop RGB usually, only GT depth
                cropped = img_or_depth_array 
            
        return cropped


    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        rgb_path = self.get_rgb_path(sample_info)
        depth_gt_path = self.get_depth_gt_path(sample_info)

        rgb_pil = Image.open(rgb_path).convert('RGB')
        
        # Load ground truth depth map (expected to be uint16 PNG)
        depth_gt_png = Image.open(depth_gt_path)
        depth_gt = np.array(depth_gt_png, dtype=np.float32)

        # Convert depth from uint16 representation to meters
        # Assumes depth = pixel_value / 256.0
        depth_gt = depth_gt / 256.0

        # Store original shape before potential cropping
        original_height, original_width = depth_gt.shape

        # Apply evaluation crop if specified (only to depth GT)
        if self.garg_crop or self.eigen_crop:
                depth_gt_cropped = self.apply_crop(depth_gt)
        else:
                depth_gt_cropped = depth_gt # No crop applied

        # Create valid mask based on GT > 0 and max_depth_eval
        # This mask is relative to the potentially *cropped* depth map
        valid_mask_cropped = (depth_gt_cropped > 1e-3) & (depth_gt_cropped <= self.max_depth_eval)

        # Apply transform to RGB image ONLY
        if self.transform:
            rgb_tensor = self.transform(rgb_pil)
        else:
            # Basic conversion if no transform provided
            rgb_tensor = torch.from_numpy(np.array(rgb_pil).transpose(2, 0, 1)).float() / 255.0

        # Convert depth map and mask to tensors
        # Depth tensor should keep original values, mask is boolean
        depth_tensor = torch.from_numpy(depth_gt_cropped).unsqueeze(0).float() # Add channel dim
        valid_mask_tensor = torch.from_numpy(valid_mask_cropped).unsqueeze(0) # Add channel dim, keep bool


        # Return RGB tensor, GT Depth tensor (cropped), and Valid Mask tensor (cropped)
        return rgb_tensor, depth_tensor, valid_mask_tensor

