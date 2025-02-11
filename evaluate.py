import os
import json
import numpy as np
from tqdm import tqdm

from utils.dataloaders import get_dataloaders
from utils.loss import *
from utils.build import get_R_STN_model, get_R_STN_optimizer


def calculate_quality_score(cosine_loss, image_similarity_loss, smooth_loss):
    # set min and max
    image_similarity_min = -4.496496201
    image_similarity_max = -0.017213229
    cosine_min = -2.993062496
    cosine_max = -1.630403519
    smooth_min = 0.038905706
    smooth_max = 0.482037395

    # normalize loss
    cosine_loss_normalized = normalize_loss(cosine_loss, cosine_min, cosine_max)
    image_similarity_loss_normalized = normalize_loss(image_similarity_loss, image_similarity_min, image_similarity_max)
    smooth_loss_normalized = normalize_loss(smooth_loss, smooth_min, smooth_max)

    #  Calculate the overall score (weights can be adjusted as needed)
    quality_score = 1 / 3 * cosine_loss_normalized + 1 / 3 * image_similarity_loss_normalized + 1 / 3 * smooth_loss_normalized

    return 1 - quality_score


def load_state_dict(model, args):
    import os
    weights_path = args["weights_path"]
    if not os.path.exists(weights_path) or not os.path.isfile(weights_path):
        weights_paths = [f for f in os.listdir("./output") if f.endswith('.pth')]
        assert len(weights_paths) > 0, "Cannot find any weights."
        weights_paths.sort(reverse=True)
        weights_path = str(os.path.join("./output", weights_paths[0]))

    model.load_state_dict(torch.load(weights_path))
    model.eval().to(args["device"])
    print(f'Model loaded from {weights_path}')
    return model


@torch.no_grad()
def evaluate(model, epoch, dataloaders):
    epoch_results = {}
    model.eval()
    device = next(model.parameters()).device
    fixed_imgs = torch.tensor(np.stack(
        [dataloaders["fixed_val_img_loader"].dataset[i] for i in
         range(len(dataloaders["fixed_val_img_loader"].dataset))],
        axis=0))
    fixed_imgs = fixed_imgs.to(device)
    fix_feat_low, fix_feat_mid, fix_feat_high, _, _, _, _ = model(fixed_imgs)
    fix_feat_low = fix_feat_low.mean(0).unsqueeze(0)
    fix_feat_mid = fix_feat_mid.mean(0).unsqueeze(0)
    fix_feat_high = fix_feat_high.mean(0).unsqueeze(0)

    for idx, (moving_img) in enumerate(tqdm(dataloaders["moving_val_img_loader"])):
        moving_img = moving_img.to(device)
        move_feat_low, move_feat_mid, move_feat_high, rec_img, grid1, grid2, grid3 = model(moving_img)

        contrast_loss = cos_loss(fix_feat_low, move_feat_low) + \
                        cos_loss(fix_feat_mid, move_feat_mid) + \
                        cos_loss(fix_feat_high, move_feat_high)

        image_similarity_loss = image_loss(fix_feat_low, move_feat_low) + \
                                image_loss(fix_feat_mid, move_feat_mid) + \
                                image_loss(fix_feat_high, move_feat_high)  # Image similarity loss

        smooth_loss = gradient_loss_2d(grid1) + gradient_loss_2d(grid2) + gradient_loss_2d(grid3)

        quality_score = calculate_quality_score(contrast_loss.item(), image_similarity_loss.item(), smooth_loss.item())

        epoch_results[dataloaders["moving_val_img_loader"].dataset.image_paths[idx].split("/")[-1]] = quality_score

    return epoch_results


if __name__ == "__main__":
    import yaml

    with open("./configs/evaluate.yaml", 'r') as stream:
        args = yaml.load(stream, Loader=yaml.FullLoader)

    # init model
    R_STN = get_R_STN_model(args)
    # init optimizer
    optimizer = get_R_STN_optimizer(R_STN, args)
    # load state dict
    R_STN = load_state_dict(R_STN, args)

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

    results = {}

    for epoch in tqdm(range(args['epochs'])):
        results = dict(results, **evaluate(R_STN, epoch, dataloaders))

    os.makedirs("./results", exist_ok=True)
    with open("./results/evaluate_results.json", 'w', encoding="utf-8") as fp:
        json.dump(results, fp)
