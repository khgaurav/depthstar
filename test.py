import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from dataset import RGBDepthDataset
from model import DepthSTAR


def visualize_single_prediction(img_path: str, model_path: str) -> None:
    """
    Visualizes and saves a depth map prediction.
    
    Args:
        img_path: path to RGB image
        model_path: path to model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Input Image Transform ---
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()])

    # --- Model Loading ---
    print(f"Loading model from: {model_path}")
    model = DepthSTAR(
        use_residual_blocks=True,
        use_transformer=True,
        transformer_layers=8,
        transformer_heads=8,
        embed_dim=512,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    print(f"Loading image from: {img_path}")

    img_pil = Image.open(img_path).convert('RGB')
    rgb_tensor = transform(img_pil).unsqueeze(0).to(device)
    start_time = time.time()
    if device == torch.device("cuda"):
        torch.cuda.synchronize()
    with torch.no_grad():
        pred_depth_tensor = model(rgb_tensor)
    if device == torch.device("cuda"):
        torch.cuda.synchronize()
    end_time = time.time()
    print(f"Inference time:{end_time - start_time}s")

    pred_depth_numpy = pred_depth_tensor.squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(img_pil)
    axs[0].set_title("Input RGB")
    axs[0].axis("off")
    im = axs[1].imshow(pred_depth_numpy, cmap="inferno")
    axs[1].set_title("Predicted Depth")
    axs[1].axis("off")
    fig.colorbar(
        im, ax=axs[1], shrink=0.75, aspect=10, label="Predicted Depth (arbitrary scale)"
    )
    plt.tight_layout()

    base_path, img_filename = os.path.split(img_path)
    filename_no_ext, _ = os.path.splitext(img_filename)
    
    output_filename = f"{filename_no_ext}_out.png"
    output_path = os.path.join(base_path, output_filename)

    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved comparison image with scale to: {output_path}")
    plt.show()

    plt.close(fig)


def visualize_dataset_prediction(model_path: str):
    """
    Visualizes and saves a depth map prediction for a dataset.
    
    Args:
        model_path: path to model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Input Image Transform ---
    transform = transforms.Compose([
        # transforms.Resize((img_size, img_size)),
        transforms.ToTensor()])

    dataset = RGBDepthDataset('./Distill-Any-Depth/cifar_images', './Distill-Any-Depth/cifar_depths', transform, transform)

    # --- Model Loading ---
    print(f"Loading model from: {model_path}")
    model = DepthSTAR(
        use_residual_blocks=True,
        use_transformer=True,
        transformer_layers=8,
        transformer_heads=8,
        embed_dim=512,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    indices = random.sample(range(len(dataset)), 5)
    fig, axs = plt.subplots(5, 3, figsize=(9, 10))
    axs = axs.reshape(5, 3)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            rgb, gt_depth = dataset[idx]
            rgb_input = rgb.unsqueeze(0).cuda()
            pred_depth = model(rgb_input).squeeze().cpu().numpy()
            gt_depth = gt_depth.squeeze().numpy()

            axs[i][0].imshow(rgb.permute(1, 2, 0))
            axs[i][0].set_title("RGB")
            axs[i][1].imshow(gt_depth, cmap="inferno")
            axs[i][1].set_title("Ground Truth")
            axs[i][2].imshow(pred_depth, cmap="inferno")
            axs[i][2].set_title("Predicted")

            for ax in axs[i]:
                ax.axis("off")

    output_filename = "out.png"
    output_path = os.path.join("./data", output_filename)

    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved comparison image with scale to: {output_path}")
    plt.show()

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize")

    parser.add_argument('--image_path', type=str, default='img.png')
    parser.add_argument('--model_path', type=str, default='./data/best_depth_model.pth')
    args = parser.parse_args()

    # visualize_single_prediction(img_path=args.image_path, model_path=args.model_path)
    visualize_dataset_prediction(model_path=args.model_path)