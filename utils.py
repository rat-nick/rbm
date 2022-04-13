import torch


def softmax_to_onehot(v):
    imax = torch.argmax(v, 1, keepdim=True)
    return torch.zeros_like(v).scatter(1, imax, 1)


def reconstruction_rmse(o, r):
    r = r[o.sum(dim=1) > 0]
    o = o[o.sum(dim=1) > 0]
    # print(o.shape, r.shape)
    return torch.sqrt(torch.sum((o - r) ** 2) / len(o)).item()
