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

clf = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=6)

ds = DataSet(K=5)

iris = load_iris()
# print(iris)
irisData = iris.data
irisTarget = iris.target

trainDataAll = ds.trainData

trainData = np.array([[-1.0, -1.0, -1.0, -1.0]], dtype='float64')
target = np.array([-1], dtype='int64')

for subset in trainDataAll:
    trainData = np.concatenate((trainData, subset.to_numpy()[:, 0:4]))
    target = np.concatenate((target, subset.to_numpy()[:, -1]))

trainData = trainData[1:]
target = target[1:]

trainData = trainData.astype(dtype='float64')
target = target.astype(dtype='int64')


print(irisData)
print('\n')
print(irisTarget)

print(f'\n\n{"-"*100}\n{"="*100}\n{"-"*100}\n\n')

print(trainData)
print('\n')
print(target)


scoresOrigin = cross_val_score(clf, iris.data, iris.target, cv=5)
print(f'\n\n{scoresOrigin}\n\n')
scoresMy = cross_val_score(clf, trainData, target, cv=5)
print(f'\n\n{scoresMy}\n\n')

# Dlaczego to wywala błąd?
