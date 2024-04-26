from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from dataset import DataSet

"""
- kryterium oceny
- technika podziau węzła
- maks. głębokość drzewa
"""

clf = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=6)

ds = DataSet(K=5)

iris = load_iris()
irisData = iris.data
irisTarget = iris.target

trainFeatures = ds.trainFeatures
trainTarget = ds.trainTarget

print(irisData)
print('\n')
print(irisTarget)

print(f'\n\n{"-"*100}\n{"="*100}\n{"-"*100}\n\n')

print(trainFeatures)
print('\n')
print(trainTarget)


scoresOrigin = cross_val_score(clf, iris.data, iris.target, cv=5)
print(f'\n\n{scoresOrigin}\n\n')
scoresMy = cross_val_score(clf, trainFeatures, trainTarget, cv=5)
print(f'\n\n{scoresMy}\n\n')
