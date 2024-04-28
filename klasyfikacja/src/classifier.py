from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np

from dataset import DataSet

FOLDS = 5

clfTree = DecisionTreeClassifier(
    criterion="entropy", splitter="best", max_depth=5
)

clfSVM = SVC(
    C=1, kernel="poly", tol=10e-16  # , max_iter=int(10e4)
)

"""
linear: nie działa
poly: nie działa
precomputed: idk, to trzeba jakoś inaczej
rbf: działa, dobrze
sigmoid: działa, słabo
"""

ds = DataSet(K=FOLDS)

Features = ds.trainFeatures
Targets = ds.trainTarget


clf = clfSVM
accuracies = []
for k in range(FOLDS):
    # if k != 2:
    trainFeatures = ds.joinData(Features, k)
    trainTargets = ds.joinData(Targets, k)
    testFeatures = Features[k]
    testTargets = Targets[k]

    # print(f'\n -- {k} --')
    clf.fit(trainFeatures, trainTargets)
    # print('done')
    testPrediction = clf.predict(testFeatures)

    TP = 0
    FN = 0
    for i in range(len(testTargets)):
        if (testPrediction[i] - testTargets[i]) == 0:
            TP += 1
        else:
            FN += 1
    accuracy = TP/(TP+FN)
    accuracies.append(accuracy)
    print(f'Fold {k}: accuracy = {accuracy}')

avgAccuracy = np.mean(accuracies)
stddevAccuracy = np.std(accuracies)
print(f'\nMean: {avgAccuracy}\nstandard deviation: {stddevAccuracy}')
