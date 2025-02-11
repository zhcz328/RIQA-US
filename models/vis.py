import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch.nn.functional as F
import random


def save_richer_transformations(image_path, output_dir, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Apply combined richer STN transformations, including non-rigid deformations, to an input image
    and save the transformed images in the specified directory.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the output images.
        device (str): Device to perform the transformations ("cuda" or "cpu").
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess the input image
    transform = T.Compose([
        T.Resize((224, 224)),  # Resize the image to a fixed size
        T.ToTensor(),          # Convert the image to a tensor
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Define random affine transformation
    def random_affine():
        """
        Generate a random affine transformation matrix with translation, rotation, scaling, and shearing.
        """
        tx = random.uniform(-0.1, 0.1)  # Translation: ±10% of image size
        ty = random.uniform(-0.1, 0.1)
        angle = random.uniform(-15, 15)  # Rotation: ±15 degrees
        scale_x = random.uniform(0.85, 1.15)  # Scaling: 85% ~ 115%
        scale_y = random.uniform(0.85, 1.15)
        shear_x = random.uniform(-0.1, 0.1)  # Shearing: ±10%
        shear_y = random.uniform(-0.1, 0.1)

        cos_a = torch.cos(torch.tensor(angle * 3.14159265 / 180))
        sin_a = torch.sin(torch.tensor(angle * 3.14159265 / 180))

        theta = torch.tensor([
            [scale_x * cos_a + shear_x * sin_a, -sin_a + shear_y * cos_a, tx],
            [sin_a + shear_x * cos_a, scale_y * cos_a + shear_y * sin_a, ty],
        ], dtype=torch.float)

        return theta

    def random_non_rigid(grid_size=10):
        """
        Generate random non-rigid deformations by perturbing grid points.
        """
        # Generate base grid
        batch_size, _, height, width = input_tensor.size()
        grid = F.affine_grid(
            torch.eye(2, 3).unsqueeze(0).repeat(batch_size, 1, 1).to(device),
            input_tensor.size(),
            align_corners=False,
        )

        # Downsample the grid to grid_size x grid_size
        grid = grid.view(1, height, width, 2)  # Original grid shape
        downsampled_grid = F.interpolate(
            grid.permute(0, 3, 1, 2),  # Change to NCHW
            size=(grid_size, grid_size),
            mode="bilinear",
            align_corners=False,
        )
        downsampled_grid = downsampled_grid.permute(0, 2, 3, 1)  # Back to NHWC

        # Add random perturbation to the downsampled grid
        perturb = (torch.rand_like(downsampled_grid) - 0.5) * 0.13  # ±10% perturbation
        perturbed_grid = downsampled_grid + perturb

        # Upsample the perturbed grid back to original size
        upsampled_grid = F.interpolate(
            perturbed_grid.permute(0, 3, 1, 2),  # Change to NCHW
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        upsampled_grid = upsampled_grid.permute(0, 2, 3, 1)  # Back to NHWC
        return F.grid_sample(input_tensor, upsampled_grid, mode="bilinear", align_corners=False)

    # Define random non-rigid deformation
    def random_non_rigid_center(grid_size=10, sigma=0.22):
        """
        Generate random non-rigid deformations focused on the center region.
        """
        # Generate base grid
        batch_size, _, height, width = input_tensor.size()
        grid = F.affine_grid(
            torch.eye(2, 3).unsqueeze(0).repeat(batch_size, 1, 1).to(device),
            input_tensor.size(),
            align_corners=False,
        )

        # Downsample the grid to grid_size x grid_size
        grid = grid.view(1, height, width, 2)  # Original grid shape
        downsampled_grid = F.interpolate(
            grid.permute(0, 3, 1, 2),  # Change to NCHW
            size=(grid_size, grid_size),
            mode="bilinear",
            align_corners=False,
        )
        downsampled_grid = downsampled_grid.permute(0, 2, 3, 1)  # Back to NHWC

        # Create a weight mask to focus on the center
        y = torch.linspace(-1, 1, grid_size, device=device)
        x = torch.linspace(-1, 1, grid_size, device=device)
        xv, yv = torch.meshgrid(x, y, indexing='ij')
        weight_mask = torch.exp(-((xv ** 2 + yv ** 2) / sigma))  # Gaussian-like weight mask
        weight_mask = weight_mask.unsqueeze(0).unsqueeze(-1)  # Shape: (1, grid_size, grid_size, 1)

        # Add random perturbation to the downsampled grid
        perturb = (torch.rand_like(downsampled_grid) - 0.5) * 0.2  # ±10% perturbation
        perturb *= weight_mask  # Apply the mask to the perturbation
        perturbed_grid = downsampled_grid + perturb

        # Upsample the perturbed grid back to original size
        upsampled_grid = F.interpolate(
            perturbed_grid.permute(0, 3, 1, 2),  # Change to NCHW
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        upsampled_grid = upsampled_grid.permute(0, 2, 3, 1)  # Back to NHWC

        # Apply the deformation to the input tensor
        return F.grid_sample(input_tensor, upsampled_grid, mode="bilinear", align_corners=False)

    # Helper to apply transformations
    def apply_affine_transformation(tensor, theta):
        batch_size = tensor.size(0)
        theta = theta.view(batch_size, 2, 3).to(device)
        grid = F.affine_grid(theta, tensor.size(), align_corners=False)
        return F.grid_sample(tensor, grid, mode='bilinear', padding_mode='reflection', align_corners=False)

    # Denormalize and convert tensors to PIL images
    def denormalize(tensor):
        tensor = tensor.squeeze(0).cpu().permute(1, 2, 0)  # Remove batch dim, move channels
        tensor = tensor * 0.5 + 0.5  # Reverse normalization
        tensor = torch.clamp(tensor, 0, 1)  # Clamp values to valid range [0, 1]
        return tensor

    # Save original image
    original_img = denormalize(input_tensor)
    original_img_pil = Image.fromarray((original_img.numpy() * 255).astype("uint8"))
    original_img_pil.save(os.path.join(output_dir, "original_image.png"))

    # Apply affine transformations and non-rigid deformations
    for i in range(5):  # Apply 5 affine transformations
        theta = random_affine()
        transformed_tensor = apply_affine_transformation(input_tensor, theta)
        transformed_img = denormalize(transformed_tensor)
        transformed_img_pil = Image.fromarray((transformed_img.numpy() * 255).astype("uint8"))
        transformed_img_pil.save(os.path.join(output_dir, f"affine_transformed_{i + 1}.png"))

    for i in range(5):  # Apply 5 non-rigid deformations
        transformed_tensor = random_non_rigid(grid_size=10)
        transformed_img = denormalize(transformed_tensor)
        transformed_img_pil = Image.fromarray((transformed_img.numpy() * 255).astype("uint8"))
        transformed_img_pil.save(os.path.join(output_dir, f"non_rigid_transformed_{i + 1}.png"))


# Example usage:
save_richer_transformations("/archive/zhuchunzheng/GVSL/train/11.jpg", "/archive/zhuchunzheng/GVSL/train/fixed_img_jizhang_transform")
