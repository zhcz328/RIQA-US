# MONAI (Medical Open Network for AI) libraries for medical imaging processing and transformations.
from __future__ import annotations

import datetime

from monai.utils import set_determinism  # Utility to ensure reproducible results.
from monai.transforms import (
    EnsureChannelFirstD,  # Ensures the input image has channels as the first dimension.
    Compose,              # Allows combining multiple transformations.
    LoadImageD,           # Loads images from disk and wraps them in a dictionary.
    RandRotateD,          # Randomly rotates the image within a specified range.
    RandZoomD,            # Randomly zooms into the image within a specified range.
    ScaleIntensityRanged  # Scales image intensity to a specified range.
)
import monai
from monai.utils import set_determinism, first
from monai.networks.layers import Conv, Norm, Pool, same_padding
import torchinfo  # Library for summarizing the PyTorch model architecture.
from torchviz import make_dot  # Visualizes the computation graph of a PyTorch model.
from monai.data import DataLoader, Dataset, CacheDataset  # Utilities for handling datasets and data loading.
from monai.config import print_config  # Prints the MONAI configuration and environment info.
from monai.networks.blocks import Warp  # Warp block for applying a displacement field to images.
from monai.apps import MedNISTDataset  # Utility for working with the MedNIST dataset.
from monai.metrics import DiceMetric
import torch.nn.functional as F  # Functional interface in PyTorch, includes many useful operations like activations.
from tqdm import tqdm  # Progress bar library for iterating over large loops.
from models.siamese import Encoder, Predictor
from models.stn import stn_net
import argparse

# Core PyTorch imports
import torch  # PyTorch core library for building and training neural networks.
from torch import nn  # PyTorch module containing neural network components.
from collections.abc import Sequence  # Collection utilities for handling sequences.
from monai.networks.blocks import (
    Warp,                     # Warp block for applying a displacement field to images.
    Convolution               # Generic convolution block used in many MONAI network architectures.
)
from monai.networks.blocks.regunet_block import (
    RegistrationDownSampleBlock,  # Block for downsampling in a registration network.
    get_conv_block,               # Utility function to get a convolution block.
    get_deconv_block              # Utility function to get a deconvolution block.
)
from monai.networks.utils import meshgrid_ij  # Utility to generate a meshgrid for image coordinates.

# General-purpose imports for working with files, images, and metrics
import os  # Operating system interface for file handling and paths.
import cv2  # OpenCV library for image processing.
import torchmetrics  # Metrics library for evaluating PyTorch models.
from torch.autograd import Variable  # Enables automatic differentiation for tensor operations.
from scipy.spatial.distance import directed_hausdorff  # Computes the directed Hausdorff distance between point clouds.
import pandas as pd  # Data manipulation library, useful for handling tabular data.
import numpy as np  # Numerical operations on large, multi-dimensional arrays and matrices.
import matplotlib.pyplot as plt  # Plotting library for visualizing data.
import tempfile  # For creating temporary files and directories.
from glob import glob  # Unix-style pathname pattern expansion.
from monai.losses import *  # Import all loss functions provided by MONAI.
from monai.metrics import *  # Import all metrics provided by MONAI.
from piqa import SSIM  # Structural Similarity Index (SSIM) metric from PIQA.

# Print MONAI configuration to check the setup.
print_config()

# Set a fixed seed for reproducibility in data transformations, model training, etc.
set_determinism(42)

# Define the dataset directory name.
# dataset_name = '/archive/zhuchunzheng/US30K_split/CAMUS_EStoED_A2C'
dataset_name = '/archive/zhuchunzheng/GVSL/'

# Construct the root directory path for the dataset.
dataset_root_dir = f'{dataset_name}/'
print(f'Root directory: {dataset_root_dir}')

# Set batch sizes for training and testing.
training_batch_size = 4
testing_batch_size = 2

# Define the size of images to be processed.
image_size = 224

# Initialize previous model weights and pre-trained model flag.
previous_model_weight_size = 256
use_pretrained_model = 0

# Define the number of training epochs.
num_epochs = 1000

# Set the number of worker threads for data loading.
data_loader_workers = 0

# Define the experiment name for saving results.
experiment_name = "Reg2IQA"

# Print the number of available GPUs.
num_gpus = torch.cuda.device_count()
print(f'Number of GPUs available: {num_gpus}')

# Check and select the device (GPU if available, otherwise CPU).
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Selected device: {device}')

# Raise an exception if no GPU is available, indicating that CPU training will be too slow.
if not torch.cuda.is_available():
    raise Exception("GPU not available. Training on CPU may be too slow.")

# Print the name of the GPU device.
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f'Device name: {device_name}')


# Define a dataset class for handling grayscale images.
class EchoDataset(Dataset):
    def __init__(self, image_paths, img_size=image_size):
        """
        Initialize the dataset.

        Parameters:
        - image_paths (list of str): List of file paths to the images.
        - img_size (int): The size to which each image will be resized.
        """
        self.image_paths = image_paths
        self.img_size = img_size
        self.n_samples = len(image_paths)

    def __getitem__(self, index):
        """
        Retrieve an image from the dataset.

        Parameters:
        - index (int): The index of the image to retrieve.

        Returns:
        - image (numpy.ndarray): The processed image as a numpy array.
        """
        # Read the image in grayscale mode.
        image = cv2.imread(self.image_paths[index], cv2.IMREAD_GRAYSCALE)

        # Resize the image to the specified size.
        image = cv2.resize(image, (self.img_size, self.img_size))

        # Normalize the image pixel values to the range [0, 1].
        image = image / image.max()

        # Expand dimensions to add a channel dimension.
        image = np.expand_dims(image, axis=0)

        # Convert the image to float32 data type.
        image = image.astype(np.float32)

        return image

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
        - int: Number of samples in the dataset.
        """
        return self.n_samples


# Define a dataset class for handling masks associated with grayscale images.
class EchoDatasetMask(Dataset):
    def __init__(self, mask_paths, img_size=image_size):
        """
        Initialize the dataset.

        Parameters:
        - mask_paths (list of str): List of file paths to the mask images.
        - img_size (int): The size to which each mask image will be resized.
        """
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.n_samples = len(mask_paths)

    def __getitem__(self, index):
        """
        Retrieve a mask from the dataset.

        Parameters:
        - index (int): The index of the mask to retrieve.

        Returns:
        - mask (numpy.ndarray): The processed mask as a numpy array.
        """
        # Read the mask image in grayscale mode.
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)

        # Resize the mask to the specified size using nearest neighbor interpolation.
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        mask[mask == 100] = 1
        mask[mask == 200] = 2

        # Expand dimensions to add a channel dimension.
        mask = np.expand_dims(mask, axis=0)

        # Convert the mask to float32 data type.
        mask = mask.astype(np.float32)

        return mask

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
        - int: Number of samples in the dataset.
        """
        return self.n_samples

# Function to create data loaders for image datasets.
def get_batches(image_paths, batch_size, num_workers, pin_memory):
    """
    Create a DataLoader for the image dataset.

    Parameters:
    - image_paths (list of str): List of file paths to the images.
    - batch_size (int): Number of samples per batch.
    - num_workers (int): Number of subprocesses to use for data loading.
    - pin_memory (bool): Whether to pin memory for faster data transfer to GPU.

    Returns:
    - DataLoader: DataLoader object for the image dataset.
    """
    dataset = EchoDataset(image_paths=image_paths, img_size=image_size)
    data_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=False)
    return data_loader

# Function to create data loaders for mask datasets.
def get_batches_mask(mask_paths, batch_size, num_workers, pin_memory):
    """
    Create a DataLoader for the mask dataset.

    Parameters:
    - mask_paths (list of str): List of file paths to the masks.
    - batch_size (int): Number of samples per batch.
    - num_workers (int): Number of subprocesses to use for data loading.
    - pin_memory (bool): Whether to pin memory for faster data transfer to GPU.

    Returns:
    - DataLoader: DataLoader object for the mask dataset.
    """
    dataset = EchoDatasetMask(mask_paths=mask_paths, img_size=image_size)
    data_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=False)
    return data_loader

# Print the number of training and validation samples for images and masks.
print(f'Train Sample numbers (fixed_img) = {len(sorted(glob(os.path.join(dataset_root_dir, "train/fixed_img_jizhang/*.png"))))}')
print(f'Train Sample numbers (fixed_msk) = {len(sorted(glob(os.path.join(dataset_root_dir, "train/fixed_msk_jizhang/*.png"))))}')
print(f'Train Sample numbers (moving_img) = {len(sorted(glob(os.path.join(dataset_root_dir, "train/moving_img/*.png"))))}')
print(f'Train Sample numbers (moving_msk) = {len(sorted(glob(os.path.join(dataset_root_dir, "train/moving_msk/*.png"))))}')
print()
print(f'Val Sample numbers (fixed_img) = {len(sorted(glob(os.path.join(dataset_root_dir, "val/fixed_img_jizhang/*.png"))))}')
print(f'Val Sample numbers (fixed_msk) = {len(sorted(glob(os.path.join(dataset_root_dir, "val/fixed_msk_jizhang/*.png"))))}')
print(f'Val Sample numbers (moving_img) = {len(sorted(glob(os.path.join(dataset_root_dir, "val/moving_img/*.png"))))}')
print(f'Val Sample numbers (moving_msk) = {len(sorted(glob(os.path.join(dataset_root_dir, "val/moving_msk/*.png"))))}')
print()

# Create data loaders for the training dataset.
fixed_train_img_loader = get_batches(
    image_paths=sorted(glob(os.path.join(dataset_root_dir, "train/fixed_img_jizhang/*"))),
    batch_size=training_batch_size,
    num_workers=data_loader_workers,
    pin_memory=True
)

fixed_train_mask_loader = get_batches_mask(
    mask_paths=sorted(glob(os.path.join(dataset_root_dir, "train/fixed_msk_jizhang/*"))),
    batch_size=training_batch_size,
    num_workers=data_loader_workers,
    pin_memory=True
)

moving_train_img_loader = get_batches(
    image_paths=sorted(glob(os.path.join(dataset_root_dir, "train/moving_img/*"))),
    batch_size=training_batch_size,
    num_workers=data_loader_workers,
    pin_memory=True
)

moving_train_mask_loader = get_batches_mask(
    mask_paths=sorted(glob(os.path.join(dataset_root_dir, "train/moving_msk/*"))),
    batch_size=training_batch_size,
    num_workers=data_loader_workers,
    pin_memory=True
)

# Print data loader objects to verify creation.
print("Train IMG FIXED Loader:", fixed_train_img_loader)
print("Train MSK FIXED Loader:", fixed_train_mask_loader)
print("Train IMG Moving Loader:", moving_train_img_loader)
print("Train MSK Moving Loader:", moving_train_mask_loader)

# Create data loaders for the validation dataset.
fixed_val_img_loader = get_batches(
    image_paths=sorted(glob(os.path.join(dataset_root_dir, "val/fixed_img_jizhang/*"))),
    batch_size=testing_batch_size,
    num_workers=data_loader_workers,
    pin_memory=True
)

fixed_val_mask_loader = get_batches_mask(
    mask_paths=sorted(glob(os.path.join(dataset_root_dir, "val/fixed_msk_jizhang/*"))),
    batch_size=testing_batch_size,
    num_workers=data_loader_workers,
    pin_memory=True
)

moving_val_img_loader = get_batches(
    image_paths=sorted(glob(os.path.join(dataset_root_dir, "val/moving_img/*"))),
    batch_size=testing_batch_size,
    num_workers=data_loader_workers,
    pin_memory=True
)

moving_val_mask_loader = get_batches_mask(
    mask_paths=sorted(glob(os.path.join(dataset_root_dir, "val/moving_msk/*"))),
    batch_size=testing_batch_size,
    num_workers=data_loader_workers,
    pin_memory=True
)

# Print data loader objects to verify creation.
print("Val IMG FIXED Loader:", fixed_val_img_loader)
print("Val MSK FIXED Loader:", fixed_val_mask_loader)
print("Val IMG Moving Loader:", moving_val_img_loader)
print("Val MSK Moving Loader:", moving_val_mask_loader)

# Create a dictionary to store DataLoader objects for different datasets.
dataloaders = {
    'fixed_train_img': fixed_train_img_loader,
    'fixed_train_msk': fixed_train_mask_loader,
    'moving_train_img': moving_train_img_loader,
    'moving_train_msk': moving_train_mask_loader,
    'fixed_val_img': fixed_val_img_loader,
    'fixed_val_msk': fixed_val_mask_loader,
    'moving_val_img': moving_val_img_loader,
    'moving_val_msk': moving_val_mask_loader
}

# Example usage: Print the DataLoader objects to verify their creation.
for key, loader in dataloaders.items():
    print(f"{key} DataLoader: {loader}")
# Extract samples from each DataLoader
fixed_train_img_sample = first(dataloaders["fixed_train_img"])[0][0]
fixed_train_msk_sample = first(dataloaders["fixed_train_msk"])[0][0]
moving_train_img_sample = first(dataloaders["moving_train_img"])[0][0]
moving_train_msk_sample = first(dataloaders["moving_train_msk"])[0][0]

fixed_val_img_sample = first(dataloaders["fixed_val_img"])[0][0]
fixed_val_msk_sample = first(dataloaders["fixed_val_msk"])[0][0]
moving_val_img_sample = first(dataloaders["moving_val_img"])[0][0]
moving_val_msk_sample = first(dataloaders["moving_val_msk"])[0][0]

# Print shapes of the samples
print(f"fixed_train_img_sample shape: {fixed_train_img_sample.shape}")
print(f"fixed_train_msk_sample shape: {fixed_train_msk_sample.shape}")
print(f"moving_train_img_sample shape: {moving_train_img_sample.shape}")
print(f"moving_train_msk_sample shape: {moving_train_msk_sample.shape}")
print(f"fixed_val_img_sample shape: {fixed_val_img_sample.shape}")
print(f"fixed_val_msk_sample shape: {fixed_val_msk_sample.shape}")
print(f"moving_val_img_sample shape: {moving_val_img_sample.shape}")
print(f"moving_val_msk_sample shape: {moving_val_msk_sample.shape}")

# Print range of pixel values
print(f"fixed_train_img_sample Range: {fixed_train_img_sample.max()} {fixed_train_img_sample.min()}")
print(f"fixed_train_msk_sample Range: {fixed_train_msk_sample.max()} {fixed_train_msk_sample.min()} {np.unique(fixed_train_msk_sample)}")
print(f"moving_train_img_sample Range: {moving_train_img_sample.max()} {moving_train_img_sample.min()}")
print(f"moving_train_msk_sample Range: {moving_train_msk_sample.max()} {moving_train_msk_sample.min()} {np.unique(moving_train_msk_sample)}")
print(f"fixed_val_img_sample Range: {fixed_val_img_sample.max()} {fixed_val_img_sample.min()}")
print(f"fixed_val_msk_sample Range: {fixed_val_msk_sample.max()} {fixed_val_msk_sample.min()} {np.unique(fixed_val_msk_sample)}")
print(f"moving_val_img_sample Range: {moving_val_img_sample.max()} {moving_val_img_sample.min()}")
print(f"moving_val_msk_sample Range: {moving_val_msk_sample.max()} {moving_val_msk_sample.min()} {np.unique(moving_val_msk_sample)}")

# Plot samples in a grid
plt.figure(figsize=(15, 7))

# Fixed training images and masks
plt.subplot(2, 4, 1)
plt.title("fixed_train_img_sample")
plt.imshow(fixed_train_img_sample.squeeze(), cmap="gray")
plt.axis('off')

plt.subplot(2, 4, 2)
plt.title("fixed_train_msk_sample")
plt.imshow(fixed_train_msk_sample.squeeze(), cmap="gray")
plt.axis('off')

plt.subplot(2, 4, 3)
plt.title("moving_train_img_sample")
plt.imshow(moving_train_img_sample.squeeze(), cmap="gray")
plt.axis('off')

plt.subplot(2, 4, 4)
plt.title("moving_train_msk_sample")
plt.imshow(moving_train_msk_sample.squeeze(), cmap="gray")
plt.axis('off')

# Fixed validation images and masks
plt.subplot(2, 4, 5)
plt.title("fixed_val_img_sample")
plt.imshow(fixed_val_img_sample.squeeze(), cmap="gray")
plt.axis('off')

plt.subplot(2, 4, 6)
plt.title("fixed_val_msk_sample")
plt.imshow(fixed_val_msk_sample.squeeze(), cmap="gray")
plt.axis('off')

plt.subplot(2, 4, 7)
plt.title("moving_val_img_sample")
plt.imshow(moving_val_img_sample.squeeze(), cmap="gray")
plt.axis('off')

plt.subplot(2, 4, 8)
plt.title("moving_val_msk_sample")
plt.imshow(moving_val_msk_sample.squeeze(), cmap="gray")
plt.axis('off')

plt.tight_layout()
plt.show()





# Initialize the RegUNet model with specific hyperparameters

def CosLoss(data1, data2, Mean=True):
    data2 = data2.detach()
    cos = nn.CosineSimilarity(dim=1)
    if Mean:
        return -cos(data1, data2).mean()
    else:
        return -cos(data1, data2)


parser = argparse.ArgumentParser(description='Registration and Image quality assessment' )
parser.add_argument('--stn_mode', type=str, default='rotation_scale',
                        help='[affine, translation, rotation, scale, shear, rotation_scale, translation_scale, rotation_translation, rotation_translation_scale]')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate of others in SGD')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD')
args = parser.parse_args()

stn = stn_net(args).to(device)
STN_optimizer = torch.optim.SGD(stn.parameters(), lr=args.lr, momentum=args.momentum)

# Initialize a Warp layer for image transformation with bilinear interpolation and zero-padding
warp_layer = Warp(mode='bilinear', padding_mode='zeros').to(device)

# Define the image loss function using Global Mutual Information
# This loss function measures the mutual information between images
image_loss = GlobalMutualInformationLoss()


# Define a custom SSIM loss class inheriting from SSIM
class SSIMLoss(SSIM):
    # Override the forward method to compute the SSIM loss as 1 minus the SSIM score
    def forward(self, x, y):
        return 1. - super().forward(x, y)


# Define the regularization term using Bending Energy Loss
regularization = BendingEnergyLoss()

# Optionally, you can compute the mean Dice score directly if you have predictions and ground truth
# dice_metric = compute_meandice(y_pred, y, include_background=True)

# Set the random seed for reproducibility
torch.manual_seed(0)

# Loss function for binary classification
criterion = nn.BCELoss()

# Initialize variables for tracking performance metrics and losses during training
total_epochs = num_epochs  # Total number of epochs for training
train_epoch_losses, val_epoch_losses = [], []  # Lists to store average loss for each epoch (training and validation)
validation_interval = 1  # Interval for running validation (every epoch in this case)
best_dice_metric = -1  # Best Dice Similarity Coefficient (DSC) achieved so far
best_dice_epoch = -1  # Epoch at which the best DSC was achieved
dice_metric_history = []  # List to store DSC metrics for each validation epoch
lowest_loss = 1e10  # Initialize to a large value to track the lowest loss
best_dice_score = 0  # Highest DSC value achieved so far
epoch_counter = 1  # Counter for epochs, purpose specified later

train_total_losses = []
train_MI_losses = []  # To store Mutual Information (MI) loss for training data across epochs
train_cosloss = []
train_rec_loss = []
train_smooth_loss = []

val_total_losses = []
val_MI_losses = []  # To store Mutual Information loss for validation data across epochs
val_cosloss = []
val_rec_loss = []
val_smooth_loss = []

def MSE(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def gradient_loss_2d(s, penalty='l2'):
    # Compute the gradient of the image field in 2D (height and width)
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])  # Vertical gradient
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])  # Horizontal gradient

    if penalty == 'l2':
        dy = dy * dy
        dx = dx * dx

    # Average the gradients across the spatial dimensions
    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0  # Return average loss for horizontal and vertical gradients


# Training loop over all epochs
for epoch in range(total_epochs):
    # print("-" * 100)
    # print(f"Epoch {epoch + 1}/{total_epochs}")

    stn.train()  # Set model to training mode
    epoch_loss_sum, step_count = 0, 0  # Initialize sum of losses and step counter for the epoch

    # Initialize metrics for the current epoch
    epoch_MI_sum, cosloss_sum, rec_loss_sum, smooth_loss_sum = 0, 0, 0, 0

    # Iterate over batches of training data
    for fixed_train_img, _, moving_train_img, _ in tqdm(zip(fixed_train_img_loader, fixed_train_mask_loader, moving_train_img_loader, moving_train_mask_loader), disable=True):
        step_count += 1  # Increment step counter
        STN_optimizer.zero_grad()  # Reset the gradients for the optimizer
        fixed_train_img = fixed_train_img.to(device)
        moving_train_img = moving_train_img.to(device)
        fix_feat0, fix_feat00, fix_feat, _, _, _, _ = stn(fixed_train_img)
        move_feat0, move_feat00, move_feat, rec_img, grid1, grid2, grid3 = stn(moving_train_img)
        z1 = fix_feat
        z2 = move_feat
        contrast_loss = CosLoss(fix_feat0, move_feat0) + CosLoss(fix_feat00, move_feat00) + CosLoss(z1, z2)
        image_similarity_loss = image_loss(fix_feat0, move_feat0) + image_loss(fix_feat00, move_feat00) + image_loss(z1, z2)  # Image similarity loss 互信息
        smooth_loss = gradient_loss_2d(grid1) + gradient_loss_2d(grid2) + gradient_loss_2d(grid3)
        rec_loss = MSE(fixed_train_img, rec_img)
        total_loss = contrast_loss + image_similarity_loss + rec_loss + smooth_loss
        total_loss.backward()  # Backpropagate the total loss
        STN_optimizer.step()  # Update the generator parameters
        # print(image_similarity_loss)
        # Accumulate losses and metrics for this step
        epoch_loss_sum += total_loss.item()
        epoch_MI_sum += image_similarity_loss.item()
        cosloss_sum += contrast_loss.item()
        rec_loss_sum += rec_loss.item()
        smooth_loss_sum += rec_loss.item()

    # Print the learning rate for the optimizer
    # for param_group in STN_optimizer.param_groups:
    #     print("Learning rate: ", param_group['lr'])

    # Aggregate and average metrics and losses for the epoch
    avg_epoch_loss = epoch_loss_sum / step_count
    avg_epoch_MI = epoch_MI_sum / step_count
    avg_cosloss = cosloss_sum/step_count
    avg_recloss = rec_loss_sum/step_count
    avg_smoothloss = smooth_loss_sum / step_count

    # Store L2 and MI losses for training in lists
    train_total_losses.append(avg_epoch_loss)
    train_MI_losses.append(avg_epoch_MI)
    train_cosloss.append(avg_cosloss)
    train_rec_loss.append(avg_recloss)
    train_smooth_loss.append(avg_smoothloss)

    # Print statistics for the current epoch
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}: Average training total Loss: {avg_epoch_loss:.5f}")
        print(f"Epoch {epoch + 1}: Average training MI Loss: {avg_epoch_MI:.5f}")
        print(f"Epoch {epoch + 1}: Average training Cos Loss: {avg_cosloss:.5f}")
        print(f"Epoch {epoch + 1}: Average training Rec Loss: {avg_recloss:.5f}")
        print(f"Epoch {epoch + 1}: Average training Smooth Loss: {avg_smoothloss:.5f}")
        print("-" * 60)

    # stn.eval()
    # for fixed_train_img, _, moving_train_img, _ in tqdm(
    #         zip(fixed_val_img_loader, fixed_val_mask_loader, moving_val_img_loader, moving_val_mask_loader), disable=True):
    #     step_count += 1  # Increment step counter
    #     fixed_train_img = fixed_train_img.to(device)
    #     moving_train_img = moving_train_img.to(device)
    #     # fix_feat, _ = stn(fixed_train_img)
    #     # move_feat, rec_img = stn(moving_train_img)
    #     # z1 = fix_feat
    #     # z2 = move_feat
    #     # contrast_loss = CosLoss(z1, z2)
    #     # image_similarity_loss = image_loss(z1, z2)  # Image similarity loss 互信息
    #     # rec_loss = MSE(fixed_train_img, rec_img)
    #
    #     fix_feat0, fix_feat00, fix_feat, _, _, _, _ = stn(fixed_train_img)
    #     move_feat0, move_feat00, move_feat, rec_img, grid1, grid2, grid3 = stn(moving_train_img)
    #     z1 = fix_feat
    #     z2 = move_feat
    #     contrast_loss = CosLoss(fix_feat0, move_feat0) + CosLoss(fix_feat00, move_feat00) + CosLoss(z1, z2)
    #     image_similarity_loss = image_loss(fix_feat0, move_feat0) + image_loss(fix_feat00, move_feat00) + image_loss(z1,z2)  # Image similarity loss 互信息
    #     smooth_loss = gradient_loss_2d(grid1)+gradient_loss_2d(grid2)+gradient_loss_2d(grid3)
    #     rec_loss = MSE(fixed_train_img, rec_img)
    #
    #     total_loss = contrast_loss + image_similarity_loss + rec_loss + smooth_loss
    #     # Accumulate losses and metrics for this step
    #     epoch_loss_sum += total_loss.item()
    #     epoch_MI_sum += image_similarity_loss.item()
    #     cosloss_sum += contrast_loss.item()
    #     rec_loss_sum += rec_loss.item()
    #     smooth_loss_sum += rec_loss.item()

        # Print the learning rate for the optimizer
    # for param_group in STN_optimizer.param_groups:
    #     print("Learning rate: ", param_group['lr'])

        # Aggregate and average metrics and losses for the epoch
    avg_epoch_loss = epoch_loss_sum / step_count
    avg_epoch_MI = epoch_MI_sum / step_count
    avg_cosloss = cosloss_sum / step_count
    avg_recloss = rec_loss_sum / step_count
    avg_smoothloss = smooth_loss_sum / step_count

    # Store L2 and MI losses for training in lists
    # val_total_losses.append(avg_epoch_loss)
    # val_MI_losses.append(avg_epoch_MI)
    # val_cosloss.append(avg_cosloss)
    # val_rec_loss.append(avg_recloss)
    # val_rec_loss.append(avg_smoothloss)
    #
    # if (epoch + 1) % 10 == 0:
    #     # Print statistics for the current epoch
    #     print(f"Epoch {epoch + 1}: Average testing total Loss: {avg_epoch_loss:.5f}")
    #     print(f"Epoch {epoch + 1}: Average testing MI Loss: {avg_epoch_MI:.5f}")
    #     print(f"Epoch {epoch + 1}: Average testing Cos Loss: {avg_cosloss:.5f}")
    #     print(f"Epoch {epoch + 1}: Average testing Rec Loss: {avg_recloss:.5f}")
    #     print(f"Epoch {epoch + 1}: Average training Smooth Loss: {avg_smoothloss:.5f}")
    #     print("-" * 60)

# Save training and validation metrics to a CSV file
train_metrics_dataframe = pd.DataFrame({
    'train_total_losses': np.array(train_total_losses),
    'train_MI_losses': np.array(train_MI_losses),
    'train_cosloss': np.array(train_cosloss),
    'train_rec_loss': np.array(train_rec_loss),
    'train_smooth_loss': np.array(train_smooth_loss)
})
val_metrics_dataframe = pd.DataFrame({
    'train_total_losses': np.array(train_total_losses),
    'train_MI_losses': np.array(train_MI_losses),
    'train_cosloss': np.array(train_cosloss),
    'train_rec_loss': np.array(train_rec_loss),
    'train_smooth_loss': np.array(train_smooth_loss)
})
# metrics_dataframe = pd.DataFrame({
#     'Validation_L2_Loss': np.array(val_L2_losses),
#     'Training_L2_Loss': np.array(train_L2_losses),
#     'Validation_MI_Loss': np.array(val_MI_losses),
#     'Training_MI_Loss': np.array(train_MI_losses),
# })
# Get the current time and format it as a string
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# Create a file name with a timestamp
file_name = f'stn_model_{current_time}.pth'
# Save the model's state_dict
torch.save(stn.state_dict(), file_name)
print(f'Model saved as {file_name}')
train_metrics_dataframe.to_csv(experiment_name + 'train.csv')
val_metrics_dataframe.to_csv(experiment_name + 'val.csv')


