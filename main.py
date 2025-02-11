import os
import datetime
import pandas as pd
import numpy as np

from tqdm import tqdm

from utils.dataloaders import get_dataloaders
from utils.loss import *
from utils.build import get_R_STN_model, get_R_STN_optimizer


def train(model, optimizer, epoch, dataloaders):
    model.train()  # Set model to training mode
    epoch_loss_sum, step_count = 0, 0  # Initialize sum of losses and step counter for the epoch
    # Initialize metrics for the current epoch
    epoch_MI_sum, cos_loss_sum, rec_loss_sum, smooth_loss_sum = 0, 0, 0, 0
    device = next(model.parameters()).device
    # Iterate over batches of training data
    for fixed_img, moving_img, in tqdm(
            zip(dataloaders["fixed_train_img_loader"], dataloaders["moving_train_img_loader"]), disable=True):
        step_count += 1  # Increment step counter
        optimizer.zero_grad()  # Reset the gradients for the optimizer
        fixed_img = fixed_img.to(device)
        moving_img = moving_img.to(device)
        fix_feat_low, fix_feat_mid, fix_feat_high, _, _, _, _ = model(fixed_img)
        move_feat_low, move_feat_mid, move_feat_high, rec_img, grid1, grid2, grid3 = model(moving_img)
        contrast_loss = cos_loss(fix_feat_low, move_feat_low) + \
                        cos_loss(fix_feat_mid, move_feat_mid) + \
                        cos_loss(fix_feat_high, move_feat_high)

        image_similarity_loss = image_loss(fix_feat_low, move_feat_low) + \
                                image_loss(fix_feat_mid, move_feat_mid) + \
                                image_loss(fix_feat_high, move_feat_high)  # Image similarity loss

        smooth_loss = gradient_loss_2d(grid1) + gradient_loss_2d(grid2) + gradient_loss_2d(grid3)
        rec_loss = MSE(fixed_img, rec_img)
        total_loss = contrast_loss + image_similarity_loss + rec_loss + smooth_loss
        total_loss.backward()  # Back-propagate the total loss
        optimizer.step()  # Update the generator parameters

        # Accumulate losses and metrics for this step
        epoch_loss_sum += total_loss.item()
        epoch_MI_sum += image_similarity_loss.item()
        cos_loss_sum += contrast_loss.item()
        rec_loss_sum += rec_loss.item()
        smooth_loss_sum += rec_loss.item()

    # Aggregate and average metrics and losses for the epoch
    avg_epoch_loss = epoch_loss_sum / step_count
    avg_epoch_MI = epoch_MI_sum / step_count
    avg_cos_loss = cos_loss_sum / step_count
    avg_rec_loss = rec_loss_sum / step_count
    avg_smooth_loss = smooth_loss_sum / step_count

    # Store L2 and MI losses for training in lists
    train_total_loss.append(avg_epoch_loss)
    train_MI_loss.append(avg_epoch_MI)
    train_cos_loss.append(avg_cos_loss)
    train_rec_loss.append(avg_rec_loss)
    train_smooth_loss.append(avg_smooth_loss)

    # Print statistics for the current epoch
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}: Average training total Loss: {avg_epoch_loss:.5f}")
        print(f"Epoch {epoch + 1}: Average training MI Loss: {avg_epoch_MI:.5f}")
        print(f"Epoch {epoch + 1}: Average training Cos Loss: {avg_cos_loss:.5f}")
        print(f"Epoch {epoch + 1}: Average training Rec Loss: {avg_rec_loss:.5f}")
        print(f"Epoch {epoch + 1}: Average training Smooth Loss: {avg_smooth_loss:.5f}")
        print("-" * 60)


@torch.no_grad()
def validate(model, epoch, dataloaders):
    model.eval()
    epoch_loss_sum, step_count = 0, 0
    epoch_MI_sum, cos_loss_sum, rec_loss_sum, smooth_loss_sum = 0, 0, 0, 0
    device = next(model.parameters()).device
    for fixed_img, moving_img in tqdm(
            zip(dataloaders["fixed_val_img_loader"], dataloaders["moving_val_img_loader"]), disable=True):
        step_count += 1  # Increment step counter
        fixed_img = fixed_img.to(device)
        moving_img = moving_img.to(device)

        fix_feat_low, fix_feat_mid, fix_feat_high, _, _, _, _ = model(fixed_img)
        move_feat_low, move_feat_mid, move_feat_high, rec_img, grid1, grid2, grid3 = model(moving_img)
        contrast_loss = cos_loss(fix_feat_low, move_feat_low) + \
                        cos_loss(fix_feat_mid, move_feat_mid) + \
                        cos_loss(fix_feat_high, move_feat_high)

        image_similarity_loss = image_loss(fix_feat_low, move_feat_low) + \
                                image_loss(fix_feat_mid, move_feat_mid) + \
                                image_loss(fix_feat_high,
                                           move_feat_high)  # Image similarity loss

        smooth_loss = gradient_loss_2d(grid1) + gradient_loss_2d(grid2) + gradient_loss_2d(grid3)
        rec_loss = MSE(fixed_img, rec_img)

        total_loss = contrast_loss + image_similarity_loss + rec_loss + smooth_loss

        # Accumulate losses and metrics for this step
        epoch_loss_sum += total_loss.item()
        epoch_MI_sum += image_similarity_loss.item()
        cos_loss_sum += contrast_loss.item()
        rec_loss_sum += rec_loss.item()
        smooth_loss_sum += rec_loss.item()

    # Aggregate and average metrics and losses for the epoch
    avg_epoch_loss = epoch_loss_sum / step_count
    avg_epoch_MI = epoch_MI_sum / step_count
    avg_cos_loss = cos_loss_sum / step_count
    avg_rec_loss = rec_loss_sum / step_count
    avg_smooth_loss = smooth_loss_sum / step_count

    # Store L2 and MI losses for training in lists
    val_total_loss.append(avg_epoch_loss)
    val_MI_loss.append(avg_epoch_MI)
    val_cos_loss.append(avg_cos_loss)
    val_rec_loss.append(avg_rec_loss)
    val_smooth_loss.append(avg_smooth_loss)

    # Print statistics for the current epoch
    print(f"Epoch {epoch + 1}: Average validation total Loss: {avg_epoch_loss:.5f}")
    print(f"Epoch {epoch + 1}: Average validation MI Loss: {avg_epoch_MI:.5f}")
    print(f"Epoch {epoch + 1}: Average validation Cos Loss: {avg_cos_loss:.5f}")
    print(f"Epoch {epoch + 1}: Average validation Rec Loss: {avg_rec_loss:.5f}")
    print(f"Epoch {epoch + 1}: Average validation Smooth Loss: {avg_smooth_loss:.5f}")
    print("-" * 60)


def save(model, train_metrics_df, val_metrics_df):
    # Get the current time and format it as a string
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Create a file name with a timestamp
    os.makedirs("./output", exist_ok=True)
    model_saved_path = f'./output/r_stn_model_{current_time}.pth'
    train_metrics_df_saved_path = f'./output/train_metrics_{current_time}.csv'
    val_metrics_df_saved_path = f'./output/val_metrics_{current_time}.csv'

    # Save the model's state_dict
    torch.save(model.state_dict(), model_saved_path)
    print(f'Model saved at {model_saved_path}')

    train_metrics_df.to_csv(train_metrics_df_saved_path)
    val_metrics_df.to_csv(val_metrics_df_saved_path)
    print(f'Metrics saved at ./output/')


if __name__ == "__main__":
    import yaml

    with open("./configs/train.yaml", 'r') as stream:
        args = yaml.load(stream, Loader=yaml.FullLoader)

    # init model
    R_STN = get_R_STN_model(args)
    # init optimizer
    optimizer = get_R_STN_optimizer(R_STN, args)

    # Set the random seed for reproducibility
    torch.manual_seed(0)

    # training metrics records
    train_total_loss = []
    train_MI_loss = []
    train_cos_loss = []
    train_rec_loss = []
    train_smooth_loss = []

    # validation metrics records
    val_total_loss = []
    val_MI_loss = []
    val_cos_loss = []
    val_rec_loss = []
    val_smooth_loss = []

    dataloaders = get_dataloaders(args)
    if args["vis_samples"]:
        from utils.vis_samples import vis_samples

        vis_samples(dataloaders)

    for epoch in tqdm(range(args['epochs'])):
        train(R_STN, optimizer, epoch, dataloaders)
        if (epoch + 1) % 5 == 0:
            validate(R_STN, epoch, dataloaders)

    # training and validation metrics
    train_metrics_df = pd.DataFrame({
        'train_total_loss': np.array(train_total_loss),
        'train_MI_loss': np.array(train_MI_loss),
        'train_cos_loss': np.array(train_cos_loss),
        'train_rec_loss': np.array(train_rec_loss),
        'train_smooth_loss': np.array(train_smooth_loss)
    })

    val_metrics_df = pd.DataFrame({
        'val_total_loss': np.array(val_total_loss),
        'val_MI_loss': np.array(val_MI_loss),
        'val_cos_loss': np.array(val_cos_loss),
        'val_rec_loss': np.array(val_rec_loss),
        'val_smooth_loss': np.array(val_smooth_loss)
    })

    save(R_STN, train_metrics_df, val_metrics_df)
