# DepthStar: Quick Monocular Depth Estimation for Robotics Applications

This repository contains code and resources for **DepthStar**, a lightweight and efficient monocular depth estimation model optimized for robotics applications and real-time inference.

## Project Overview

DepthStar is an autoencoder-based model leveraging a hybrid convolutional-transformer architecture. It efficiently estimates depth maps from single RGB images and is particularly suitable for deployment on edge devices.

### Recommended Hardware
- NVIDIA GPU (e.g., T4, RTX GPUs)

## Setup Instructions

### Step 1: Clone the Repository
```bash
git clone https://github.com/khgaurav/depthstar.git
cd depthstar
```

### Step 2: Setup Python Environment
It's recommended to use Conda or virtualenv:

```bash
conda create -n depthstar python=3.10
conda activate depthstar
pip install -r requirements.txt
```

### Step 3: Prepare Datasets

#### NYUv2 Dataset
Download the NYU Depth V2 dataset `.mat` file and place it in the `data` directory:
- [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

#### CIFAR-10 Dataset
Used for generating synthetic depth maps (see synthetic depth generation script provided separately).

## Training DepthStar

Generating Dataset from CIFAR-10 Dataset
Open the generate_dataset.ipynb notebook and execute all the cells to download CIFAR dataset and generate the depth data from Distill-Any-Depth model.

Execute the training script:

```bash
python train.py --epochs 200 --batch_size 16 --lr 1e-4 --data_dir ./data --model_save_path ./models
```

This script will save the best-performing model based on validation accuracy into the specified `model_save_path`.

## Evaluating DepthStar

Use the benchmarking script to evaluate on datasets:

```bash
python benchmark.py --img_size 32 --batch_size 8 --nyu_v2_path ./data/nyu_depth_v2_labeled.mat
```

Evaluation results will display RMSE, MAE, and other metrics clearly in the terminal.

## Visualizing Predictions

To visualize depth predictions for individual images:

```bash
python test.py --image_path ./data/ing.png --model_path ./models/best_depth_model.pth --img_size 32
```

Visualizations will appear and save to the directory containing the input image.


## Contact
If you have questions, please contact:
- Keivalya Bhartendu Pandya (pandya.kei@northeastern.com)
- Gaurav Kothamachu Harish (kothamachuharish.g@northeastern.com)
- Rachel Lim (lim.rac@northeastern.com)

Thank you for checking out our project!

