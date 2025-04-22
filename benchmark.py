import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import argparse
import os
import time
from model import DepthSTAR
from dataset import RGBDepthDataset, NYUDepthV2MatDataset

def compute_depth_metrics(pred, gt, mask):
    """Computes depth metrics (RMSE, AbsRel, SqRel, LogRMSE, Accuracy thresholds)"""
    pred_m = pred[mask]
    gt_m = gt[mask]

    # Ensure non-zero ground truth for relative errors and log
    valid_rel_mask = gt_m > 1e-4
    pred_m_rel = pred_m[valid_rel_mask]
    gt_m_rel = gt_m[valid_rel_mask]

    if gt_m_rel.numel() == 0:
        abs_rel = torch.tensor(float('nan'), device=pred.device)
        sq_rel = torch.tensor(float('nan'), device=pred.device)
        log_rmse = torch.tensor(float('nan'), device=pred.device)
        delta1 = torch.tensor(float('nan'), device=pred.device)
        delta2 = torch.tensor(float('nan'), device=pred.device)
        delta3 = torch.tensor(float('nan'), device=pred.device)
    else:
        thresh = torch.maximum((gt_m_rel / pred_m_rel), (pred_m_rel / gt_m_rel))
        delta1 = (thresh < 1.25).float().mean()
        delta2 = (thresh < 1.25 ** 2).float().mean()
        delta3 = (thresh < 1.25 ** 3).float().mean()

        abs_diff = torch.abs(gt_m_rel - pred_m_rel)
        abs_rel = (abs_diff / gt_m_rel).mean()
        sq_rel = (((gt_m_rel - pred_m_rel) ** 2) / gt_m_rel).mean()

        log_diff_sq = (torch.log(gt_m_rel) - torch.log(pred_m_rel)) ** 2
        log_rmse = torch.sqrt(log_diff_sq.mean())

    # RMSE computed on all valid pixels (gt > 0)
    if gt_m.numel() == 0:
        rmse = torch.tensor(float('nan'), device=pred.device)
    else:
        rmse = torch.sqrt(((gt_m - pred_m) ** 2).mean())


    return {
        'rmse': rmse.item(),
        'abs_rel': abs_rel.item(),
        'sq_rel': sq_rel.item(),
        'log_rmse': log_rmse.item(),
        'delta1': delta1.item(),
        'delta2': delta2.item(),
        'delta3': delta3.item(),
        'count': mask.sum().item() # Number of valid pixels used
    }

# --- Main Evaluation Function ---

def evaluate_model_on_datasets(model_path: str, dataset_configs: list, img_size: int, batch_size: int = 8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model Loading ---
    print(f"Loading model from: {model_path}")
    model = DepthSTAR(
        use_residual_blocks=True,
        use_transformer=True,
        transformer_layers=8,
        transformer_heads=8,
        embed_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        # transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    results = {}

    for config in dataset_configs:
        dataset_name = config.get("name")
        dataset_type = config.get("type")
        dataset_path = config.get("path")
        print(f"\n--- Evaluating on Dataset: {dataset_name} ---")
        print(f"Path: {dataset_path}")

        dataset = None
        if dataset_type == "nyu_v2_mat":
            dataset = NYUDepthV2MatDataset(
                mat_file_path=dataset_path,
                transform_rgb=transform,
                transform_depth=transform,
                img_key=config.get("img_key", "images"),
                depth_key=config.get("depth_key", "depths")
            )

        else:
            print(f"Unknown dataset type: {dataset_type}")
            continue

        if not dataset or len(dataset) == 0:
            print(f"Dataset {dataset_name} is empty or failed to load.")
            continue

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        total_metrics = {k: 0.0 for k in ['rmse', 'abs_rel', 'sq_rel', 'log_rmse', 'delta1', 'delta2', 'delta3']}
        total_count = 0
        start_time = time.time()

        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):

                # --- Data Handling ---
                if dataset_type == "nyu_v2_mat":
                    rgb_batch, target_depth_batch = batch_data
                    valid_mask_batch = target_depth_batch > 1e-8

                else: # Should not happen due to earlier checks
                    continue

                rgb_batch = rgb_batch.to(device)
                target_depth_batch = target_depth_batch.to(device)
                valid_mask_batch = valid_mask_batch.to(device)

                 # Remove channel dimension from mask and target if present (expected H, W)
                if valid_mask_batch.dim() == 4 and valid_mask_batch.shape[1] == 1:
                     valid_mask_batch = valid_mask_batch.squeeze(1)
                if target_depth_batch.dim() == 4 and target_depth_batch.shape[1] == 1:
                     target_depth_batch = target_depth_batch.squeeze(1)


                # --- Inference ---
                pred_depth_batch = model(rgb_batch)

                # Squeeze potential channel dimension from prediction if model outputs (B, 1, H, W)
                if pred_depth_batch.dim() == 4 and pred_depth_batch.shape[1] == 1:
                    pred_depth_batch = pred_depth_batch.squeeze(1)

                # Ensure prediction and target have same shape (B, H, W)
                if pred_depth_batch.shape != target_depth_batch.shape:
                     print(f"Warning: Prediction shape {pred_depth_batch.shape} != Target shape {target_depth_batch.shape}. Resizing prediction.")
                     pred_depth_batch = F.interpolate(pred_depth_batch.unsqueeze(1), # Add channel dim back for interpolate
                                                      size=target_depth_batch.shape[-2:],
                                                      mode='bilinear', # Use bilinear for predicted depth
                                                      align_corners=False).squeeze(1) # Remove channel dim

                # --- Calculate and Accumulate Metrics ---
                # Only compute metrics where the mask is valid
                batch_metrics = compute_depth_metrics(pred_depth_batch, target_depth_batch, valid_mask_batch)
                num_valid_pixels = batch_metrics['count']

                if num_valid_pixels > 0:
                    total_count += num_valid_pixels
                    for k in total_metrics:
                       if not np.isnan(batch_metrics[k]): # Avoid adding NaNs
                            # Weighted average: metric_value * num_pixels_in_batch
                            total_metrics[k] += batch_metrics[k] * num_valid_pixels

                if (i + 1) % 50 == 0:
                    print(f"  Batch [{i+1}/{len(dataloader)}]")

        end_time = time.time()
        eval_time = end_time - start_time

        # --- Calculate Average Metrics ---
        avg_metrics = {k: (total_metrics[k] / total_count if total_count > 0 else float('nan')) for k in total_metrics}

        print(f"--- Results for {dataset_name} ({len(dataset)} samples) ---")
        print(f"Evaluation Time: {eval_time:.2f} seconds")
        if total_count > 0:
            print(f"  RMSE:      {avg_metrics['rmse']:.4f}")
            print(f"  AbsRel:    {avg_metrics['abs_rel']:.4f}")
            print(f"  SqRel:     {avg_metrics['sq_rel']:.4f}")
            print(f"  LogRMSE:   {avg_metrics['log_rmse']:.4f}")
            print(f"  Delta < 1.25:  {avg_metrics['delta1']:.4f}")
            print(f"  Delta < 1.25^2:{avg_metrics['delta2']:.4f}")
            print(f"  Delta < 1.25^3:{avg_metrics['delta3']:.4f}")
            print(f"  Evaluated on {total_count} valid pixels")
        else:
            print("  No valid pixels found for evaluation.")

        results[dataset_name] = avg_metrics

    return results

# --- Main Execution ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark Depth Estimation Model')

    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=2)

    # Example using placeholder paths:
    parser.add_argument('--nyu_v2_path', type=str, default='/home/kothamachuharish.g/data/nyu_depth_v2_labeled.mat')
    parser.add_argument('--custom_rgbd_path', type=str, default='/home/kothamachuharish.g/Distill-Any-Depth/')

    args = parser.parse_args()

    # List of dataset configurations to test
    dataset_configurations = [
        {
            "name": "NYU Depth V2",
            "type": "nyu_v2_mat",
            "path": args.nyu_v2_path,
             "img_key": 'images',
             "depth_key": 'depths'
        }
    ]

    # --- Run Evaluation ---
    all_results = evaluate_model_on_datasets(
        model_path=f'/home/kothamachuharish.g/data/depth_model_cifar_{args.img_size}.pth',
        dataset_configs=dataset_configurations,
        img_size=args.img_size,
        batch_size=args.batch_size
    )

    print("\n--- Benchmark Summary ---")
    for name, metrics in all_results.items():
        print(f"Dataset: {name}")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        print("-" * 20)