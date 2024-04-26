from ucimlrepo import fetch_ucirepo

import numpy as np

# import pandas as pd
# import copy

seed = 318407
# seed = 271102
# seed = 231219
np.random.seed(seed)


class DataSet:
    def get_data(self, id: int = 53):
        iris = fetch_ucirepo(id=id)
        dataset = iris.data
        return dataset

    def __convertAbstract(self, vector, dictionary):
        output_vector = []
        for i in range(len(vector)):
            translated_item = dictionary[vector[i]]
            output_vector.append(translated_item)
        return output_vector

    def convertNameToInt(self, vector):
        forward_dict = {
            "Iris-setosa": 0,
            "Iris-versicolor": 1,
            "Iris-virginica": 2,
        }
        return self.__convertAbstract(vector, forward_dict)

    def convertIntToName(self, vector):
        backward_dict = {
            0: "Iris-setosa",
            1: "Iris-versicolor",
            2: "Iris-virginica",
        }
        return self.__convertAbstract(vector, backward_dict)

    def splitData(self, DataFrame):
        # features = np.array([[-1.0, -1.0, -1.0, -1.0]], dtype="float64")
        # target = np.array([-1], dtype="int64")

        features = []
        targets = []
        for subset in DataFrame:
            features.append(
                np.array(subset.to_numpy()[:, 0:4], dtype='float64')
            )
            targets.append(
                np.array(subset.to_numpy()[:, -1], dtype='int64')
            )

        # features = features[1:]
        # target = target[1:]

        # features = features.astype(dtype="float64")
        # targets = targets.astype(dtype="int64")

        return features, targets

    def joinData(self, DataFrame: np.array):
        FrameType = DataFrame[0].dtype
        outputArray = np.array([DataFrame[0][0]], dtype=FrameType)
        for subset in DataFrame:
            outputArray = np.concatenate((outputArray, subset))
        outputArray = outputArray[1:]
        outputArray = outputArray.astype(dtype=FrameType)
        return outputArray

    def __init__(self, K: int = 5) -> None:
        dataset = self.get_data()
        original = dataset.original
        trans_target = self.convertNameToInt(original["class"])

        original.loc[:, "class"] = np.array(trans_target)
        shuffled = dataset.original.sample(frac=1)
        groupedSets = np.array_split(shuffled, K + 1)

        self.testData = groupedSets.pop(-1)
        self.trainData = groupedSets

        self.trainFeatures, self.trainTarget = self.splitData(self.trainData)
        self.joinedTrainFeatures = self.joinData(self.trainFeatures)
        self.joinedTrainTarget = self.joinData(self.trainTarget)

        self.testFeatures = np.array(
            self.testData.to_numpy()[:, 0:4], dtype='float64')
        self.testTarget = np.array(
            self.testData.to_numpy()[:, -1], dtype='int64')
        # self.testFeatures, self.testTarget = self.splitData(self.testData)
        # self.testFeatures = self.joinData(self.testFeatures)
        # self.testTarget = self.joinData(self.testTarget)
