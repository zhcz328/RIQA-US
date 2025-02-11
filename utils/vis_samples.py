from monai.utils import first
from matplotlib import pyplot as plt


def vis_samples(dataloaders):
    try:
        # Extract samples from each DataLoader
        fixed_train_img_sample = first(dataloaders["fixed_train_img_loader"])[0][0]
        moving_train_img_sample = first(dataloaders["moving_train_img_loader"])[0][0]

        # Print shapes of the samples
        print(f"fixed_train_img_sample shape: {fixed_train_img_sample.shape}")
        print(f"moving_train_img_sample shape: {moving_train_img_sample.shape}")

        # Print range of pixel values
        print(f"fixed_train_img_sample Range: {fixed_train_img_sample.max()} {fixed_train_img_sample.min()}")
        print(f"moving_train_img_sample Range: {moving_train_img_sample.max()} {moving_train_img_sample.min()}")

    except Exception as e:
        pass

    fixed_val_img_sample = first(dataloaders["fixed_val_img_loader"])[0][0]
    moving_val_img_sample = first(dataloaders["moving_val_img_loader"])[0][0]

    # Print shapes of the samples
    print(f"fixed_val_img_sample shape: {fixed_val_img_sample.shape}")
    print(f"moving_val_img_sample shape: {moving_val_img_sample.shape}")


    print(f"fixed_val_img_sample Range: {fixed_val_img_sample.max()} {fixed_val_img_sample.min()}")
    print(f"moving_val_img_sample Range: {moving_val_img_sample.max()} {moving_val_img_sample.min()}")

    # Plot samples in a grid
    plt.figure(figsize=(15, 7))
    try:
        # Fixed training images and masks
        plt.subplot(2, 2, 1)
        plt.title("fixed_train_img_sample")
        plt.imshow(fixed_train_img_sample.squeeze(), cmap="gray")
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.title("moving_train_img_sample")
        plt.imshow(moving_train_img_sample.squeeze(), cmap="gray")
        plt.axis('off')
    except:
        pass

    plt.subplot(2, 2, 3)
    plt.title("fixed_val_img_sample")
    # Fixed validation images and masks
    plt.subplot(2, 2, 3)
    plt.title("fixed_val_img_sample")
    plt.imshow(fixed_val_img_sample.squeeze(), cmap="gray")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("moving_val_img_sample")
    plt.imshow(moving_val_img_sample.squeeze(), cmap="gray")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
