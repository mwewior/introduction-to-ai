# import copy
from ucimlrepo import fetch_ucirepo

# import pandas as pd
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
        grouped_sets = np.array_split(shuffled, K + 1)
        self.test_group = grouped_sets.pop(-1)
        self.cross_group = grouped_sets


# ds = DataSet(K=5)
# tgroup = ds.test_group
# cgroup = ds.cross_group
# c0 = cgroup[0]
# c1 = cgroup[1]
# c2 = cgroup[2]
# c3 = cgroup[3]
# c4 = cgroup[4]

# print(tgroup)
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
# print("\n\n ---------------------------------------------------- \n\n")


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
