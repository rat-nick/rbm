from math import sqrt
import torch

mae = torch.nn.L1Loss()
mse = torch.nn.MSELoss()


def softmax_to_onehot(v):
    imax = torch.argmax(v, 1, keepdim=True)
    return torch.zeros_like(v).scatter(1, imax, 1)


def onehot_to_ratings(v):
    return torch.argmax(v, dim=1)


def absolute_error(o, r):
    r = r[o.sum(dim=1) > 0]
    o = o[o.sum(dim=1) > 0]

    r = onehot_to_ratings(r).float()
    o = onehot_to_ratings(o).float()

    return torch.sum(torch.abs(o - r)).item()


def squared_error(o, r):
    r = r[o.sum(dim=1) > 0]
    o = o[o.sum(dim=1) > 0]

    r = onehot_to_ratings(r).float()
    o = onehot_to_ratings(o).float()
    return torch.sum(torch.square(o - r)).item()


def reconstruction_rmse(o, r):
    r = r[o.sum(dim=1) > 0]
    o = o[o.sum(dim=1) > 0]

    r = onehot_to_ratings(r).float()
    o = onehot_to_ratings(o).float()

    return sqrt(mse(o, r).item())


def reconstruction_mae(o, r):
    r = r[o.sum(dim=1) > 0]
    o = o[o.sum(dim=1) > 0]

    r = onehot_to_ratings(r).float()
    o = onehot_to_ratings(o).float()

    return mae(o, r).item()
