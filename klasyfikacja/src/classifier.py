from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np
from typing import List

from dataset import DataSet
from observation import Observation

FOLDS = 5


def trueClassification(classes: List[Observation], target: int):
    classes[target].TP += 1
    for c in classes:
        if c != classes[target]:
            c.TN += 1
        c.update()


def falseClassification(classes: List[Observation], target: int, predict: int):
    classes[target].FN += 1
    classes[predict].FP += 1
    for c in classes:
        if c != classes[target] and c != classes[predict]:
            c.TN += 1
        c.update()


clfTree = DecisionTreeClassifier(
    criterion="entropy", splitter="best", max_depth=4
)

clfSVM = SVC(
    C=1, kernel="linear", tol=10e-16, max_iter=int(10e6)
)

"""
trzeba dać ograniczenie iteracji!!!

linear: nie działa
poly: nie działa
precomputed: idk, to trzeba jakoś inaczej
rbf: działa, dobrze
sigmoid: działa, słabo
"""

ds = DataSet(K=FOLDS)

Features = ds.trainFeatures
Targets = ds.trainTarget


clf = clfTree

if clf == clfTree:
    print(
        f'\ncriterion: {clf.criterion},' +
        f'\n splitter: {clf.splitter},' +
        f'\nmax depth: {clf.max_depth}' +
        '\n'
    )
elif clf == clfSVM:
    print(
        f'\nkernel: {clf.kernel},' +
        f'\n     C: {clf.C},' +
        f'\n   tol: {clf.tol}' +
        f'\n  iter: {clf.max_iter}' +
        '\n'
    )

accuracies = []
for k in range(FOLDS):
    trainFeatures = ds.joinData(Features, k)
    trainTargets = ds.joinData(Targets, k)
    testFeatures = Features[k]
    testTargets = Targets[k]

    clf.fit(trainFeatures, trainTargets)
    testPrediction = clf.predict(testFeatures)

    oSetosa = Observation()
    oVersicolor = Observation()
    oVirginica = Observation()
    observation = [oSetosa, oVersicolor, oVirginica]

    numerosity = len(testTargets)
    AccuracyPOSITIVE = 0

    target_str = ""
    predict_str = ""
    where_error_str = ""

    for i in range(numerosity):
        predict_str += f'{testPrediction[i]} '
        target_str += f'{testTargets[i]} '

        targetClass = testTargets[i]
        predictClass = testPrediction[i]

        if (predictClass == targetClass):
            AccuracyPOSITIVE += 1
            where_error_str += "  "
            trueClassification(observation, targetClass)
            # observation[targetClass].TP += 1
            # for c in observation:
            #     if c != observation[targetClass]:
            #         c.TN += 1
        else:
            where_error_str += "X "
            falseClassification(observation, targetClass, predictClass)
            # observation[targetClass].FN += 1
            # observation[predictClass].FP += 1
            # for c in observation:
            #     if c != observation[targetClass] and c != observation[predictClass]:  # noqa
            #         c.TN += 1

    accuracy = AccuracyPOSITIVE / numerosity

    print(f'\nFold {k}: accuracy = {round(accuracy, 5)}')
    print(f'errors:    {where_error_str}')
    print(f'predicted: {predict_str}')
    print(f'target:    {target_str}')

    accuracies.append(accuracy)

avgAccuracy = np.mean(accuracies)
stddevAccuracy = np.std(accuracies)
print(f'\nMean: {round(avgAccuracy, 5)}\nstandard deviation: {stddevAccuracy}')
