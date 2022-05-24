from random import sample
from turtle import shape
from surprise import AlgoBase, PredictionImpossible
from surprise.dataset import DatasetAutoFolds, Dataset

from sklearn.model_selection import train_test_split
import torch
import numpy as np
from rbm import RBM
from utils.data import ratingsToTensor
from utils.tensors import softmax_to_rating


class RBMAlgorithm(AlgoBase):
    def __init__(self, model: RBM, split_ratio: float = 0.9):
        self.model = model
        self.split_ratio = split_ratio

        AlgoBase.__init__(self)

    def fit(self, dataTensor: DatasetAutoFolds):
        AlgoBase.fit(self, dataTensor)
        self.trainset = dataTensor.build_full_trainset()

        dataTensor = ratingsToTensor(dataTensor)
        self.ratings = dataTensor
        self.predictions = np.ndarray([dataTensor.shape[0], dataTensor.shape[1]])
        train, test = train_test_split(
            dataTensor,
            train_size=self.split_ratio,
            random_state=42,
        )

        self.model.fit(train, test)
        for user in self.trainset.all_users():
            v = self.model.reconstruct(self.ratings[user])
            for item in self.trainset.all_items():
                self.predictions[user, item] = softmax_to_rating(v[item])
        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible("User and/or item is unknown.")

        return self.predictions[u, i]

    def predict(self, uid, iid, r_ui=None, clip=True, verbose=False):
        return super().predict(uid, iid, r_ui, clip, verbose)


if __name__ == "__main__":
    data = Dataset.load_builtin("ml-1m")
    items = data.build_full_trainset().n_items

    rbm = RBM(
        n_visible=5 * items,
        n_hidden=100,
        learning_rate=0.005,
        verbose=True,
        early_stopping=True,
        patience=5,
        l2=0.0001,
        l1=0.001,
        max_epoch=50,
        batch_size=10,
        momentum=0.2,
    )
    rbm.h *= 0.0
    algo = RBMAlgorithm(model=rbm, split_ratio=0.9)
    algo.fit(dataTensor=data)
    print(algo.model.rmse)
    pred = algo.predict(str(234), str(1184))
    print(pred)
