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


dataset = Dataset(device=device, name="ml-1m")
tensorData = dataset.getDatasetAsTensor()
trainset, testset = train_test_split(tensorData, train_size=0.9, random_state=42)
print(trainset.shape, testset.shape)


rbm = RBM(
    n_visible=trainset.shape[1] * trainset.shape[2],
    n_hidden=10,
    device=device,
    learning_rate=0.1,
    momentum=0.5,
)

for epoch in range(1, config.epochs + 1):
    print(f"Epoch\t{epoch}", end="\t")
    se_val = 0
    ae_val = 0
    ratings_count = 0
    trainset = trainset[torch.randperm(len(trainset))]
    for user in range(0, len(trainset), config.batch_size):

        minibatch = trainset[user : user + config.batch_size]
        rbm.apply_gradient(minibatch=minibatch, t=(epoch // 20 + 1))

    se = 0
    ae = 0
    ratings_count = 0
    for vin in testset:
        rec = rbm.reconstruct(vin)
        ratings_count += len(vin[vin.sum(dim=1) > 0])
        se += squared_error(vin, rec)
        ae += absolute_error(vin, rec)

    print("TestRMSE\t", sqrt(se / ratings_count), end="\t")
    print("TestMAE\t", ae / ratings_count)
