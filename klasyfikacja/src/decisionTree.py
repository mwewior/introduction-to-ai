from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from dataset import DataSet
"""
- kryterium oceny
- technika podziau węzła
- maks. głębokość drzewa
"""


clf = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=2)

iris = load_iris()
# print(iris)
# print(iris.data)
# print(iris.target)

cross_val_score(clf, iris.data, iris.target, cv=10)

# dataset = DataSet(K=5)
# data = dataset.test_group
# data_target = dataset.convert(data[['class']])
# data[['class']] = data_target
# dataset.cross_group[2].iloc[:, -1]
# cross_val_score(clf, dataset.test_group, dataset.test_group[['class']])
