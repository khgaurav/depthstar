import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset import NYUDepthV2MatDataset, RGBDepthDataset
from model import DepthSTAR

VAL_SPLIT = 0.1
LOG_INTERVAL = 100

<<<<<<< HEAD
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def train_model(
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-5,
    num_workers: int = 2,
    data_dir: str = "../Distill-Any-Depth",
    model_save_path: str = "../data",
):
    """
    Train and save model.
    
    Args:
        epochs: number of epochs to run for
        batch_size: batch size
        lr: learning rate
        num_workers: number of workers
        data_dir: directory of RGB/depth images
        model_save_path: directory to save model to
    """
    
=======
def train_model(epochs=100, batch_size=4, lr=1e-5, num_workers=2, data_dir='./Distill-Any-Depth', model_save_path='./data'):
>>>>>>> 1993bc795e89abc36266382b041b6c81c7300042
    print("--- Starting Training Configuration ---")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {lr}")
    print(f"Validation Split: {VAL_SPLIT*100:.1f}%")
    print(f"Number of Workers: {num_workers}")
    print(f"Data Directory: {data_dir}")
    print(f"Model Save Path: {model_save_path}")
    print("-" * 35)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform_rgb = transforms.Compose(
        [
            # transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform_depth = transforms.Compose(
        [
            # transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ]
    )

<<<<<<< HEAD
    image_path = os.path.join(data_dir, "images_hf_stream_224")
    depth_path = os.path.join(data_dir, "depths_hf_stream_224")
=======
    image_path = os.path.join(data_dir, 'cifar_images')
    depth_path = os.path.join(data_dir, 'cifar_depths')
>>>>>>> 1993bc795e89abc36266382b041b6c81c7300042

    full_dataset = RGBDepthDataset(
        image_path, depth_path, transform_rgb, transform_depth
    )
    # full_dataset = NYUDepthV2MatDataset('../data/nyu_depth_v2_labeled.mat', transform=transform)

    total_size = len(full_dataset)
    val_size = int(total_size * VAL_SPLIT)
    train_size = total_size - val_size

    print(f"Splitting dataset: Train={train_size}, Validation={val_size}")
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
    )

    print("Initializing model...")
    model = DepthSTAR(
        use_residual_blocks=True,
        use_transformer=True,
        transformer_layers=8,
        transformer_heads=8,
        embed_dim=512,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)
    criterion = nn.L1Loss()

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        processed_batches = 0

        for batch_idx, (rgb, depth) in enumerate(train_loader):
            rgb, depth = rgb.to(device), depth.to(device)

            optimizer.zero_grad()
            pred = model(rgb)

            loss = criterion(pred, depth)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            processed_batches += 1

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                print(
                    f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Train Loss (Batch): {loss.item():.4f}"
                )

        avg_train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():  # Disable gradient calculation for validation
            for rgb_val, depth_val in val_loader:
                rgb_val, depth_val = rgb_val.to(device), depth_val.to(device)
                pred_val = model(rgb_val)
                val_loss = criterion(pred_val, depth_val)
                running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        # scheduler.step(avg_val_loss)
        print(f"  Current Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"Epoch {epoch+1}/{epochs} Summary:")
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        print(f"  Avg Validation Loss: {avg_val_loss:.4f}")

        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
<<<<<<< HEAD
            save_filename = f"best_depth_model_2.pth"
=======
            save_filename = f'best_depth_model.pth'
>>>>>>> 1993bc795e89abc36266382b041b6c81c7300042
            save_filepath = os.path.join(model_save_path, save_filename)
            os.makedirs(model_save_path, exist_ok=True)
            torch.save(model.state_dict(), save_filepath)
            print(f"  Validation loss improved! Saving model to {save_filepath}")
        else:
            print(f"  Validation loss did not improve from {best_val_loss:.4f}.")

<<<<<<< HEAD

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="../Distill-Any-Depth")
    parser.add_argument("--model_save_path", type=str, default="../data")
=======
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default='./Distill-Any-Depth')
    parser.add_argument('--model_save_path', type=str, default='./data')
>>>>>>> 1993bc795e89abc36266382b041b6c81c7300042

    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        model_save_path=args.model_save_path,
    )
