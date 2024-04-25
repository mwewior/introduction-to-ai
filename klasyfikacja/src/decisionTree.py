from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from dataset import DataSet
import numpy as np

"""
- kryterium oceny
- technika podziau węzła
- maks. głębokość drzewa
"""


clf = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=2)

ds = DataSet(K=5)

iris = load_iris()
# print(iris)
irisData = iris.data
irisTarget = iris.target

print(irisData)
print(irisTarget)
scoresOrigin = cross_val_score(clf, iris.data, iris.target, cv=5)
print(scoresOrigin)

# trainData = np.array([[]], dtype='float64')
trainDataAll = ds.trainData
i = 0
for subset in trainDataAll:
    if i == 0:
        trainData = subset.to_numpy()[:, 0:4]
        target = subset.to_numpy()[:, -1]
    else:
        trainData = np.concatenate((trainData, subset.to_numpy()[:, 0:4]))
        target = np.concatenate((target, subset.to_numpy()[:, -1]))
    i += 1
# target = ds.trainTarget

print(trainData)
print(target)
scoresMy = cross_val_score(clf, trainData, target, cv=5)
print(scoresMy)

# Dlaczego to wywala błąd?


# dataset = DataSet(K=5)
# data = dataset.test_group
# data_target = dataset.convert(data[['class']])
# data[['class']] = data_target
# dataset.cross_group[2].iloc[:, -1]
# cross_val_score(clf, dataset.test_group, dataset.test_group[['class']])
