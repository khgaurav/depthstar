import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import DepthSTAR
from dataset import RGBDepthDataset
import argparse
import os
import random

def visualize_single_prediction(img_path: str, model_path: str, img_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Input Image Transform ---
    transform = transforms.Compose([
        # transforms.Resize((img_size, img_size)),
        transforms.ToTensor()])

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

    print(f"Loading image from: {img_path}")
    img_pil = Image.open(img_path).convert('RGB')
    rgb_tensor = transform(img_pil).unsqueeze(0).to(device) # Add batch dimension

    with torch.no_grad():
        pred_depth_tensor = model(rgb_tensor)

    pred_depth_numpy = pred_depth_tensor.squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(img_pil.resize((img_size, img_size)))
    axs[0].set_title("Input RGB")
    axs[0].axis('off')
    im = axs[1].imshow(pred_depth_numpy, cmap='inferno')
    axs[1].set_title("Predicted Depth")
    axs[1].axis('off')
    fig.colorbar(im, ax=axs[1], shrink=0.75, aspect=10, label='Predicted Depth (arbitrary scale)')
    plt.tight_layout()
    
    base_path, img_filename = os.path.split(img_path)
    filename_no_ext, _ = os.path.splitext(img_filename)
    
    output_filename = f"{filename_no_ext}_out_{img_size}.png"
    output_path = os.path.join(base_path, output_filename)

    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved comparison image with scale to: {output_path}")
    plt.show()

    plt.close(fig)

def visualize_dataset_prediction(model_path: str, img_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Input Image Transform ---
    transform = transforms.Compose([
        # transforms.Resize((img_size, img_size)),
        transforms.ToTensor()])

    dataset = RGBDepthDataset('../Distill-Any-Depth/cifar_depths', '../Distill-Any-Depth/cifar_depths', transform, transform)

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
            axs[i][1].imshow(gt_depth, cmap='inferno')
            axs[i][1].set_title("Ground Truth")
            axs[i][2].imshow(pred_depth, cmap='inferno')
            axs[i][2].set_title("Predicted")

            for ax in axs[i]:
                ax.axis('off')
    
    
    output_filename = "out.png"
    output_path = os.path.join("../data", output_filename)

    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved comparison image with scale to: {output_path}")
    plt.show()

    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize')

    parser.add_argument('--image_path', type=str, default='/home/kothamachuharish.g/Distill-Any-Depth/cifar_images/img_0000.png')
    parser.add_argument('--model_path', type=str, default='../data/depth_model_cifar_32.pth')
    parser.add_argument('--img_size', type=int, default=32)
    args = parser.parse_args()

    visualize_dataset_prediction(model_path=args.model_path, img_size=args.img_size)