from sklearn.model_selection import train_test_split
from dataset import Dataset
from rbm import RBM
import torch
import config

if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = "cpu"


dataset = Dataset(device=device)
tensorData = dataset.getDatasetAsTensor()
tensorData = tensorData[tensorData.sum(dim=[1, 2]) != 0]
trainset, testset = train_test_split(tensorData, train_size=0.8)
print("Dataset loaded")


rbm = RBM(trainset.shape[1], n_hidden=50, device=device, learning_rate=0.1)

for epoch in range(1, config.epochs + 1):
    i = 1
    for case in trainset:

        goodSample = case
        badSample = case

        for k in range(10):
            _, hk = rbm.sample_h(badSample)  # forward pass
            _, badSample = rbm.sample_v(hk)  # backward pass

        rbm.train(goodSample, badSample)
        i += 1
        # if i % 100 == 0:
        # calculate reconstruction rmse
        # rec = rbm.reconstruct(case)
        # rec = rec[case.sum(dim=1) > 0]
        # case = case[case.sum(dim=1) > 0]
        # case = torch.argmax(case, dim=1)
        # rec = torch.argmax(rec, dim=1)

        # error = torch.sqrt(torch.sum((case - rec) ** 2) / len(case))
        # print(error)
    print(f"Epoch {epoch} done.", end=" ")
    rmse = 0
    for case in testset:
        reconstruction = rbm.reconstruct(case)
        reconstruction = reconstruction[case.sum(dim=1) > 0]
        case = case[case.sum(dim=1) > 0]
        case = torch.argmax(case, dim=1) + 1
        reconstruction = torch.argmax(reconstruction, dim=1) + 1

        # print(f"Case:{case}")
        # print(f"Reco:{reconstruction}")
        # print(f"C-R :{case-reconstruction}")
        rmse += torch.sqrt(torch.sum((case - reconstruction) ** 2) / len(case))
    print(rmse / len(testset))
