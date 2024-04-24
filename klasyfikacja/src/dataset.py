import copy
from ucimlrepo import fetch_ucirepo

import pandas as pd
import numpy as np

np.random.seed(318407)


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
            'Iris-setosa': 0,
            'Iris-versicolor': 1,
            'Iris-virginica': 2
        }
        return self.__convertAbstract(vector, forward_dict)

    def convertIntToName(self, vector):
        backward_dict = {
            0: 'Iris-setosa',
            1: 'Iris-versicolor',
            2: 'Iris-virginica'
        }
        return self.__convertAbstract(vector, backward_dict)

    def __init__(self, K: int = 5) -> None:
        dataset = self.get_data()
        original = dataset.original
        trans_target = self.convertNameToInt(original['class'])
        original.loc[:, 'class'] = np.array(trans_target)
        shuffled = dataset.original.sample(frac=1)
        groupedSets = np.array_split(shuffled, K + 1)
        self.testData = groupedSets.pop(-1)
        self.testTarget = self.testData['class']
        self.crossData = groupedSets
        # self.crossTarget = self.crossData['class']
        self.crossTarget = copy.deepcopy(groupedSets)
        for i in range(K):
            currTarget = self.crossData[i]['class']
            # self.crossTarget.loc[:, 'class'] = np.array(currTarget)
            self.crossTarget[i] = currTarget


ds = DataSet(K=5)
groupT = ds.testData
groupC = ds.crossData
targetT = ds.testTarget
targetC = ds.crossTarget

# c0 = groupC[0]
# c1 = groupC[1]
# c2 = groupC[2]
# c3 = groupC[3]
# c4 = groupC[4]

# c0targ = targetC[0]
# c1targ = targetC[1]
# c2targ = targetC[2]
# c3targ = targetC[3]
# c4targ = targetC[4]

# print("night\n\n")
# print(groupT)
# print("\n\n ---------------------------------------------------- \n\n")
# print(c0)
# print("\n\n ---------------------------------------------------- \n\n")
# print(c1)
# print("\n\n ---------------------------------------------------- \n\n")
# print(c2)
# print("\n\n ---------------------------------------------------- \n\n")
# print(c3)
# print("\n\n ---------------------------------------------------- \n\n")
# print(c4)
# print("\n\n -------------------------------------------------------------------------------------------------------- \n\n\n\n")
# print(targetT)
# print("\n\n ---------------------------------------------------- \n\n")
# print(c0targ)
# print("\n\n ---------------------------------------------------- \n\n")
# print(c1targ)
# print("\n\n ---------------------------------------------------- \n\n")
# print(c2targ)
# print("\n\n ---------------------------------------------------- \n\n")
# print(c3targ)
# print("\n\n ---------------------------------------------------- \n\n")
# print(c4targ)
# print("\n\n -------------------------------------------------------------------------------------------------------- \n\n\n\n")


# def learning():
#     # tutaj to jakoś się uczy, daje parametry potem
#     return 0


# def testing():
#     # dla uzyskanych parametrów klasyfikacji przeprowadzamy
#     # badanie na danych testowych i oceniamy jakość
#     return 0


# parameters = []
# for i in range(K):
#     train_group = copy.deepcopy(cross_group)
#     validation_group = train_group.pop(i)
#     parameters.append(learning())
# test_results = testing()
