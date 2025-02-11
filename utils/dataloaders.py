from utils.datasets import EchoDataset, EchoDatasetMask
from monai.data import DataLoader
from glob import glob

import os


# Function to create data loaders for image datasets.
def get_batches(image_paths, img_size, batch_size, num_workers, pin_memory=True):
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
    dataset = EchoDataset(image_paths=image_paths, img_size=img_size)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             shuffle=False)
    return data_loader


# Function to create data loaders for mask datasets.
def get_batches_mask(mask_paths, img_size, batch_size, num_workers, pin_memory):
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
    dataset = EchoDatasetMask(mask_paths=mask_paths, img_size=img_size)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             shuffle=False)
    return data_loader


def get_dataloaders(args):
    root_path = args["dataset_path"]
    trn_batch_size = args["trn_batch_size"]
    val_batch_size = args["val_batch_size"]
    img_size = args["img_size"]
    num_workers = args["num_workers"]
    print(f'Dataset path: {root_path}')
    print(f'Training batch size: {trn_batch_size}')
    print(f'Validation batch size: {val_batch_size}')
    print(f'Image size: {img_size}')
    print("-" * 50)

    # Print the number of training and validation samples for images and masks.
    print(f'Train Sample numbers (fixed_img) = {len(glob(os.path.join(root_path, "train/fixed_img_jizhang/*.png")))}')
    print(f'Train Sample numbers (moving_img) = {len(glob(os.path.join(root_path, "train/moving_img/*.png")))}')
    print(f'Val Sample numbers (fixed_img) = {len(glob(os.path.join(root_path, "val/fixed_img_jizhang/*.png")))}')
    print(f'Val Sample numbers (moving_img) = {len(glob(os.path.join(root_path, "val/moving_img/*.png")))}')
    print("-" * 50)

    fixed_train_img_loader = get_batches(
        image_paths=sorted(glob(os.path.join(root_path, "train/fixed_imgs/*"))),
        img_size=img_size,
        batch_size=trn_batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    moving_train_img_loader = get_batches(
        image_paths=sorted(glob(os.path.join(root_path, "train/moving_imgs/*"))),
        img_size=img_size,
        batch_size=trn_batch_size,
        num_workers=trn_batch_size,
        pin_memory=True
    )

    print("Train IMG FIXED Loader:", fixed_train_img_loader)
    print("Train IMG Moving Loader:", moving_train_img_loader)

    fixed_val_img_loader = get_batches(
        image_paths=sorted(glob(os.path.join(root_path, "val/fixed_imgs/*"))),
        img_size=img_size,
        batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    moving_val_img_loader = get_batches(
        image_paths=sorted(glob(os.path.join(root_path, "val/moving_imgs/*"))),
        img_size=img_size,
        batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    print("Val IMG FIXED Loader:", fixed_val_img_loader)
    print("Val IMG Moving Loader:", moving_val_img_loader)

    dataloaders = {
        'fixed_train_img_loader': fixed_train_img_loader,
        'moving_train_img_loader': moving_train_img_loader,
        'fixed_val_img_loader': fixed_val_img_loader,
        'moving_val_img_loader': moving_val_img_loader,
    }

    return dataloaders
