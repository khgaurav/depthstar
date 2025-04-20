import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import argparse
import os
import time

from model import DepthSTAR
from dataset import NYUDepthV2MatDataset, KITTIBenchmarkDataset
def compute_depth_metrics(pred, gt, mask):
    """Computes depth metrics (RMSE, AbsRel, SqRel, LogRMSE, Accuracy thresholds)"""
    # Ensure inputs are tensors and mask is boolean
    pred = torch.as_tensor(pred)
    gt = torch.as_tensor(gt)
    mask = torch.as_tensor(mask, dtype=torch.bool)

    # Apply the mask to prediction and ground truth
    pred_m = pred[mask]
    gt_m = gt[mask]

    # Handle case where mask is empty
    if gt_m.numel() == 0:
        nan_tensor = torch.tensor(float('nan'), device=pred.device)
        return {
            'rmse': nan_tensor.item(), 'abs_rel': nan_tensor.item(), 'sq_rel': nan_tensor.item(),
            'log_rmse': nan_tensor.item(), 'delta1': nan_tensor.item(),
            'delta2': nan_tensor.item(), 'delta3': nan_tensor.item(),
            'count': 0
        }

    # --- Calculate metrics only on masked elements ---

    rmse = torch.sqrt(((gt_m - pred_m) ** 2).mean())

    valid_rel_mask = gt_m > 1e-4 # Use gt_m (already masked)
    pred_m_rel = pred_m[valid_rel_mask]
    gt_m_rel = gt_m[valid_rel_mask]

    if gt_m_rel.numel() == 0:
        nan_tensor = torch.tensor(float('nan'), device=pred.device)
        abs_rel = nan_tensor
        sq_rel = nan_tensor
        log_rmse = nan_tensor
        delta1 = nan_tensor
        delta2 = nan_tensor
        delta3 = nan_tensor
    else:
        # AbsRel (Mean Absolute Relative Error)
        abs_diff = torch.abs(gt_m_rel - pred_m_rel)
        abs_rel = (abs_diff / gt_m_rel).mean()

        sq_rel = (((gt_m_rel - pred_m_rel) / gt_m_rel) ** 2).mean()

        # LogRMSE (Root Mean Squared Logarithmic Error)
        valid_log_mask = (gt_m_rel > 1e-6) & (pred_m_rel > 1e-6)
        if valid_log_mask.sum() > 0:
            log_diff_sq = (torch.log(gt_m_rel[valid_log_mask]) - torch.log(pred_m_rel[valid_log_mask])) ** 2
            log_rmse = torch.sqrt(log_diff_sq.mean())
        else:
             log_rmse = torch.tensor(float('nan'), device=pred.device)


        # Delta Accuracies (percentage of pixels where max(gt/pred, pred/gt) < threshold)
        thresh = torch.maximum((gt_m_rel / pred_m_rel), (pred_m_rel / gt_m_rel))
        delta1 = (thresh < 1.25).float().mean()
        delta2 = (thresh < 1.25 ** 2).float().mean()
        delta3 = (thresh < 1.25 ** 3).float().mean()

    return {
        'rmse': rmse.item(),
        'abs_rel': abs_rel.item(),
        'sq_rel': sq_rel.item(),
        'log_rmse': log_rmse.item(),
        'delta1': delta1.item(),
        'delta2': delta2.item(),
        'delta3': delta3.item(),
        'count': mask.sum().item() # Number of valid pixels used (from original mask)
    }

def get_eigen_crop_mask(height, width):
    crop_height = 352
    crop_width = 1216

    # Check if the image is large enough for the crop
    if height < crop_height or width < crop_width:
        print(f"Warning: Image dimensions ({height}x{width}) are smaller than "
              f"the target Eigen crop ({crop_height}x{crop_width}). Cannot apply crop.")
        # Return an empty mask or handle as appropriate
        # Returning a mask of all False might be suitable
        return torch.from_numpy(np.ones((height, width), dtype=bool))
        # Alternatively, raise an error:
        # raise ValueError("Image dimensions too small for Eigen crop")

    # Calculate top-left corner of the crop
    # Crop from bottom
    top = height - crop_height
    # Crop from center horizontally
    left = (width - crop_width) // 2

    # Calculate bottom-right corner (exclusive index)
    bottom = top + crop_height
    right = left + crop_width

    # Create the mask
    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = True

    return torch.from_numpy(mask)

def evaluate_model_on_datasets(model_path: str, dataset_configs: list, eval_transform: transforms.Compose, batch_size: int = 8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model Loading ---
    input_size = None
    for t in eval_transform.transforms:
        if isinstance(t, transforms.Resize):
            size = t.size
            input_size = size[0] if isinstance(size, tuple) else size
            break

    model = DepthSTAR(
        use_residual_blocks=True,
        use_transformer=True,
        transformer_layers=8,
        transformer_heads=8,
        embed_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode

    results = {}

    for config in dataset_configs:
        dataset_name = config.get("name", "UnnamedDataset")
        dataset_type = config.get("type")
        dataset_path = config.get("path")


        print(f"\n--- Evaluating on Dataset: {dataset_name} ---")
        print(f"Type: {dataset_type}, Path: {dataset_path}")

        dataset = None
        if dataset_type == "nyu_v2_mat":
            # Pass the standardized evaluation transform to the dataset
            dataset = NYUDepthV2MatDataset(
                mat_file_path=dataset_path,
                transform=eval_transform,
                max_depth_eval=config.get("max_depth_eval", 10.0), # NYU specific cap for valid mask
                img_key=config.get("img_key", "images"),
                depth_key=config.get("depth_key", "depths")
            )
        elif dataset_type == "kitti":
                kitti_split_file = config.get("split_file")
                dataset = KITTIBenchmarkDataset(
                    data_path=dataset_path, # Base path to KITTI data (e.g., 'kitti_data/')
                    split_file=kitti_split_file, # Path to the specific split file (e.g., 'splits/eigen_benchmark/test_files.txt')
                    transform=eval_transform, # Apply transform only to RGB
                    max_depth_eval=config.get("max_depth_eval", 80.0), # KITTI typically 80m
                    depth_gt_subfolder=config.get("depth_gt_subfolder", "depth_maps/groundtruth"), # Adjust if GT depths are elsewhere
                )

        pin_mem = device.type == 'cuda'
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_mem)

        total_metrics = {k: 0.0 for k in ['rmse', 'abs_rel', 'sq_rel', 'log_rmse', 'delta1', 'delta2', 'delta3']}
        total_valid_pixel_count = 0 # Use a clearer name
        start_time = time.time()

        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                rgb_batch, target_depth_batch, valid_mask_batch = batch_data

                rgb_batch = rgb_batch.to(device)
                target_depth_batch = target_depth_batch.to(device)
                valid_mask_batch = valid_mask_batch.to(device, dtype=torch.bool) # Ensure mask is boolean

                pred_depth_batch = model(rgb_batch) # Output shape likely (B, 1, H, W) or (B, H, W)
                target_h, target_w = target_depth_batch.shape[-2:]

                if target_depth_batch.dim() == 4 and target_depth_batch.shape[1] == 1:
                    target_depth_batch = target_depth_batch.squeeze(1) # -> (B, H, W)
                if valid_mask_batch.dim() == 4 and valid_mask_batch.shape[1] == 1:
                    valid_mask_batch = valid_mask_batch.squeeze(1) # -> (B, H, W)

                # Handle prediction shape: Ensure it's (B, H, W) and matches target size
                if pred_depth_batch.dim() == 4 and pred_depth_batch.shape[1] == 1:
                    pred_depth_batch = pred_depth_batch.squeeze(1) # -> (B, H, W)
                elif pred_depth_batch.dim() != 3:
                    pred_depth_batch = pred_depth_batch.squeeze()

                # Resize prediction if its H, W doesn't match target
                if pred_depth_batch.shape[-2:] != (target_h, target_w):
                    # print(f"Warning: Prediction shape {pred_depth_batch.shape[-2:]} != Target shape {(target_h, target_w)}. Resizing prediction.")
                    pred_depth_batch = F.interpolate(pred_depth_batch.unsqueeze(1), # Add channel dim back for interpolate
                                                     size=(target_h, target_w),
                                                     mode='bilinear', # Use bilinear for predicted depth
                                                     align_corners=False).squeeze(1) # Remove channel dim

                # Ensure mask shape matches target/pred after potential squeezing/resizing
                if valid_mask_batch.shape != target_depth_batch.shape:
                     print(f"Warning: Mask shape {valid_mask_batch.shape} != Target shape {target_depth_batch.shape}. This shouldn't happen if dataset/transform is correct.")
                     # Attempt to resize mask using nearest neighbor if needed, but indicates an issue upstream.
                     valid_mask_batch = F.interpolate(valid_mask_batch.unsqueeze(1).float(),
                                                        size=(target_h, target_w),
                                                        mode='nearest').squeeze(1).bool()

                # --- Apply Eigen Crop for NYU/KITTI ---
                crop_mask = torch.ones_like(valid_mask_batch)
                if config.get("apply_eigen_crop", False):
                    crop_mask = get_eigen_crop_mask(target_h, target_w).to(device) # Create crop mask (H, W)
                    crop_mask = crop_mask.unsqueeze(0).expand_as(valid_mask_batch) # Expand to (B, H, W)

                # Combine original valid mask with the crop mask
                final_mask = valid_mask_batch & crop_mask

                # --- Calculate and Accumulate Metrics ---
                batch_metrics = compute_depth_metrics(pred_depth_batch, target_depth_batch, final_mask)
                num_valid_pixels_in_batch = batch_metrics['count']

                if num_valid_pixels_in_batch > 0:
                    total_valid_pixel_count += num_valid_pixels_in_batch
                    for k in total_metrics:
                        if not np.isnan(batch_metrics[k]):
                            total_metrics[k] += batch_metrics[k] * num_valid_pixels_in_batch
                        else:
                            pass

                if (i + 1) % 50 == 0 or (i + 1) == len(dataloader):
                    print(f"  Batch [{i+1}/{len(dataloader)}]")

        end_time = time.time()
        eval_time = end_time - start_time

        # --- Calculate Average Metrics ---
        avg_metrics = {}
        if total_valid_pixel_count > 0:
             avg_metrics = {k: (total_metrics[k] / total_valid_pixel_count) for k in total_metrics}
        else:
             avg_metrics = {k: float('nan') for k in total_metrics}


        print(f"--- Results for {dataset_name} ({len(dataset)} samples) ---")
        print(f"Evaluation Time: {eval_time:.2f} seconds")
        if total_valid_pixel_count > 0:
            print(f"  RMSE:       {avg_metrics['rmse']:.4f}")
            print(f"  AbsRel:     {avg_metrics['abs_rel']:.4f}")
            print(f"  SqRel:      {avg_metrics['sq_rel']:.4f}")
            print(f"  LogRMSE:    {avg_metrics['log_rmse']:.4f}")
            print(f"  Delta < 1.25:  {avg_metrics['delta1']:.4f}")
            print(f"  Delta < 1.25^2:{avg_metrics['delta2']:.4f}")
            print(f"  Delta < 1.25^3:{avg_metrics['delta3']:.4f}")
            print(f"  Evaluated on {total_valid_pixel_count} valid pixels (after masking/cropping)")
        else:
            print("  No valid pixels found for evaluation or error during processing.")

        results[dataset_name] = avg_metrics

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark')

    parser.add_argument('--model_path', type=str, default='../data/depth_model_32.pth')
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=8)

    # --- Dataset Paths ---
    # Use environment variables or direct paths. Make them optional arguments.
    parser.add_argument('--nyu_v2_path', type=str, default='../data/nyu_depth_v2_labeled.mat')
    # parser.add_argument('--kitti_path', type=str, default=os.environ.get("KITTI_PATH", None), help='Path to the base KITTI dataset directory.')

    # KITTI
    parser.add_argument('--kitti_path', type=str, default='/scratch/kothamachuharish.g/monodepth2/kitti_data')
    parser.add_argument('--kitti_split_file', type=str, default='/scratch/kothamachuharish.g/monodepth2/splits/eigen_benchmark/test_files.txt')
    parser.add_argument('--kitti_gt_subfolder', type=str, default="depth_maps/groundtruth", help='Subfolder within KITTI structure containing GT depth maps.')
    parser.add_argument('--kitti_max_depth', type=float, default=80.0, help='Maximum depth for KITTI evaluation.')
    parser.add_argument('--kitti_crop', type=str, default='garg', choices=['garg', 'eigen', 'none'], help='Type of crop to apply for KITTI evaluation (handled by dataset).')

    args = parser.parse_args()

    # --- Define Evaluation Transform ---
    # This should match the input requirements of the model being evaluated
    eval_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    # --- List of dataset configurations to test ---
    dataset_configurations = []

    dataset_configurations.append({
        "name": "NYU Depth V2",
        "type": "nyu_v2_mat",
        "path": args.nyu_v2_path,
        "max_depth_eval": 10.0, # Standard max depth for NYUv2 eval
        "apply_eigen_crop": True
    })
    # dataset_configurations.append({
    #     "name": f"KITTI (Split: {os.path.basename(args.kitti_split_file)}, Crop: {args.kitti_crop})",
    #     "type": "kitti",
    #     "path": args.kitti_path, # Base data directory
    #     "split_file": args.kitti_split_file, # Specific split file
    #     "max_depth_eval": args.kitti_max_depth,
    #     "depth_gt_subfolder": args.kitti_gt_subfolder, # Pass subfolder info
    # })

    # --- Run Evaluation ---
    all_results = evaluate_model_on_datasets(
        model_path=args.model_path,
        dataset_configs=dataset_configurations,
        eval_transform=eval_transform,
        batch_size=args.batch_size
    )

    # --- Print Summary ---
    print("\nBenchmark Summary")
    for name, metrics in all_results.items():
        print(f"Dataset: {name}")
        if all(np.isnan(v) for v in metrics.values()):
                print("  Metrics calculation failed or no valid data.")
        else:
            for key, value in metrics.items():
                print(f"  {key:<10}: {value:.4f}")
        print("-" * 30)