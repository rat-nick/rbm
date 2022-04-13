from sklearn.model_selection import train_test_split
from dataset import Dataset
from rbm import RBM
import torch
import config
from utils import *

if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = "cpu"


dataset = Dataset(device=device)
tensorData = dataset.getDatasetAsTensor()
trainset, testset = train_test_split(tensorData, train_size=0.75, random_state=42)
print("Dataset loaded")


rbm = RBM(trainset.shape[1], n_hidden=10, device=device, learning_rate=0.1)

for epoch in range(1, config.epochs + 1):

    for vin in trainset:

        v0 = vin
        vk = vin

        for k in range(10):
            _, hk = rbm.sample_h(vk)  # forward pass
            _, vk = rbm.sample_v(hk)  # backward pass
            vk[v0.sum(dim=1) == 0] = v0[v0.sum(dim=1) == 0]

        rbm.train(v0, vk)

    print(f"Epoch {epoch} done.", end=" ")
    rmse = 0
    for vin in testset:
        rec = rbm.reconstruct(vin)
        # print(reconstruction_rmse(vin, rec))
        rmse += reconstruction_rmse(vin, rec)
    print(rmse / len(testset))
