import torch
from models.r_stn import r_stn_net


def get_R_STN_model(args):
    R_STN = r_stn_net(args).to(args["device"])
    print(f"R_STN mode: {args['mode']}")
    return R_STN


def get_R_STN_optimizer(R_STN, args):
    print(f"SGD optimizer: lr = {args['lr']}, momentum = {args['momentum']}")
    return torch.optim.SGD(R_STN.parameters(), lr=args["lr"], momentum=args["momentum"])
