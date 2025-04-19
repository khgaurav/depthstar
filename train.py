import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from Dataset import RGBDepthDataset, NYUDepthV2MatDataset
from HybridDepthModel import HybridDepthModel
import argparse


def train_model(img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # train_dataset = RGBDepthDataset('../Distill-Any-Depth/images/', '../Distill-Any-Depth/depths/', transform)
    train_dataset = NYUDepthV2MatDataset('../data/nyu_depth_v2_labeled.mat', transform=transform)
    print(len(train_dataset))
    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    model = HybridDepthModel(img_size).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    for epoch in range(100):
        model.train()
        total_loss = 0
        for rgb, depth in dataloader:
            rgb, depth = rgb.cuda(), depth.cuda()
            pred = model(rgb)
            loss = criterion(pred, depth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), f'../data/depth_model_nyu_{img_size}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--img_size', type=int, default=32)
    args = parser.parse_args()
    train_model(img_size=args.img_size)