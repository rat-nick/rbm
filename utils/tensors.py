from math import sqrt
import torch
import matplotlib.pyplot as plt

mae = torch.nn.L1Loss()
mse = torch.nn.MSELoss()


def softmax_to_onehot(v):
    imax = torch.argmax(v, 1, keepdim=True)
    return torch.zeros_like(v).scatter(1, imax, 1)


def softmax_to_rating(v):
    ratings = torch.arange(1, v.shape[0] + 1).float()
    return torch.dot(v, ratings).item()


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


def ratings_softmax(v, num_ratings=5):
    v = v.reshape(v.shape[0] // num_ratings, num_ratings)
    v = torch.softmax(v, dim=1)
    return v


if __name__ == "__main__":
    v1 = torch.randint(high=10, size=(5,)).float()
    v2 = torch.randint(high=10, size=(5,)).float()
    v3 = sqrt(mse(v1, v2).item())
    v4 = mae(v1, v2)
    print(v1, v2, v3, v4)
