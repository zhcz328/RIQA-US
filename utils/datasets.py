from monai.data import Dataset
import numpy as np
import cv2


# Define a dataset class for handling grayscale images.
class EchoDataset(Dataset):
    def __init__(self, image_paths, img_size=224):
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
    def __init__(self, mask_paths, img_size=224):
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
