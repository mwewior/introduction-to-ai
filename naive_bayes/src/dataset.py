import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer


SEED = 318407
np.random.seed(seed=SEED)


class DataSetSKL:
    def __init__(self) -> None:
        features, target = load_iris(return_X_y=True)
        shapeDS = (features.shape[0], features.shape[1]+1)
        ds = np.zeros(shape=shapeDS)
        ds[:, 0:-1] = features
        ds[:, -1] = target

        shuffle = np.random.permutation(shapeDS[0])
        ds = ds[shuffle]

        self.features = ds[:, 0:-1]
        self.target = ds[:, -1]
        self.data = ds

        # lb = LabelBinarizer()
        # bin_target = np.array([number[0] for number in lb.fit_transform(self.target)])
        # self.bin_target = bin_target


