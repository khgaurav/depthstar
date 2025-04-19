import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from HybridDepthModel import HybridDepthModel
import argparse
import os

def visualize_single_prediction(img_path: str, model_path: str, img_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Input Image Transform ---
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()])

    # --- Model Loading ---
    print(f"Loading model from: {model_path}")
    model = HybridDepthModel(input_size=img_size).to(device)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize')

    parser.add_argument('--image_path', type=str, default='../data/image.jpg')
    parser.add_argument('--model_path', type=str, default='../data/depth_model_32.pth')
    parser.add_argument('--img_size', type=int, default=32)
    args = parser.parse_args()

    visualize_single_prediction(img_path=args.image_path, model_path=args.model_path, img_size=args.img_size)