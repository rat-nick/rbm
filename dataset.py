from numpy import int16, int8
import pandas as pd
from surprise.dataset import Dataset as sDataset
import torch


class Dataset:
    def __init__(self, device: str = "cpu", name="ml-100k") -> None:
        self.load(name)
        self.device = device

    def load(self, name):
        ds = sDataset.load_builtin(name=name, prompt=True)
        self.data = ds

    def getDatasetAsTensor(self):
        """
        Convert ratings into a tensor of shape `(u, m, 5)` where u is the number of users, and m is the number of items.
        """

        trainset = self.data.build_full_trainset()
        self.df = pd.DataFrame(trainset.all_ratings())
        # print(self.df)
        antiset = pd.DataFrame(trainset.build_anti_testset(fill=0))
        antiset[0] = antiset[0].apply(trainset.to_inner_uid)
        antiset[1] = antiset[1].apply(trainset.to_inner_iid)
        # print(antiset)
        self.df = pd.concat([self.df, antiset])
        self.df.sort_values(by=[0, 1], inplace=True)
        # print(self.df)
        self.df.columns = ["user", "item", "rating"]
        self.df["rating"] = self.df["rating"].astype("int")
        self.df = pd.concat([self.df, pd.get_dummies(self.df.rating)], axis=1)
        self.df = self.df.drop(["rating", 0], axis=1)

        u = len(trainset.all_users())
        i = len(trainset.all_items())
        # print(u, i)
        k = 5

        data = self.df.to_numpy(dtype=int16)
        t = torch.Tensor(data[:, 2:].reshape(u, i, 5))
        return t.to(device=self.device)


if __name__ == "__main__":
    data = Dataset()
    t = data.getDatasetAsTensor()
    # print(t)
