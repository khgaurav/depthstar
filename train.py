import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from dataset import RGBDepthDataset, NYUDepthV2MatDataset
from model import DepthSTAR, HybridDepthModel
import argparse


def train_model(img_size=224, epochs=100):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_dataset = RGBDepthDataset('../Distill-Any-Depth/images/', '../Distill-Any-Depth/depths/', transform)
    # train_dataset = NYUDepthV2MatDataset('../data/nyu_depth_v2_labeled.mat', transform=transform)
    print(len(train_dataset))
    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = DepthSTAR(
        use_residual_blocks=True,
        use_transformer=True,
        transformer_layers=8,
        transformer_heads=8,
        embed_dim=512).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) 
    criterion = nn.L1Loss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for rgb, depth in dataloader:
            rgb, depth = rgb.to(device), depth.to(device)
            pred = model(rgb)
            loss = criterion(pred, depth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), f'../data/depth_model_{img_size}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()
    train_model(img_size=args.img_size, epochs=args.epochs)