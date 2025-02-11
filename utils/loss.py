import torch
from monai.losses import GlobalMutualInformationLoss, BendingEnergyLoss
from piqa import SSIM


def cos_loss(data1, data2, is_mean=True):
    data2 = data2.detach()
    cos = torch.nn.CosineSimilarity(dim=1)
    if is_mean:
        return -cos(data1, data2).mean()
    else:
        return -cos(data1, data2)


class SSIMLoss(SSIM):
    """Define a custom SSIM loss class inheriting from SSIM"""

    # Override the forward method to compute the SSIM loss as 1 minus the SSIM score
    def forward(self, x, y):
        return 1. - super().forward(x, y)


def MSE(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def gradient_loss_2d(s, penalty='l2'):
    """Compute the gradient of the image field in 2D (height and width)"""

    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])  # Vertical gradient
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])  # Horizontal gradient

    if penalty == 'l2':
        dy = dy * dy
        dx = dx * dx

    # Average the gradients across the spatial dimensions
    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0  # Return average loss for horizontal and vertical gradients


def normalize_loss(loss, min_loss, max_loss):
    return (loss - min_loss) / (max_loss - min_loss)


# Define the image loss function using Global Mutual Information
# This loss function measures the mutual information between images
image_loss = GlobalMutualInformationLoss()

# Define the regularization term using Bending Energy Loss
regularization = BendingEnergyLoss()
