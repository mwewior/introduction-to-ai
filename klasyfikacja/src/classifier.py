from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np
from typing import List

from dataset import DataSet
from observation import Observation


def trueClassification(classes: List[Observation], target: int) -> None:
    classes[target].TP += 1
    for c in classes:
        if c != classes[target]:
            c.TN += 1
        c.update()


def falseClassification(classes: List[Observation], target: int, predict: int) -> None:  # noqa
    classes[target].FN += 1
    classes[predict].FP += 1
    for c in classes:
        if c != classes[target] and c != classes[predict]:
            c.TN += 1
        c.update()


def printInfo(clf) -> None:
    info = ""
    if clf == clfTree:
        info += f'\ncriterion: {clf.criterion}, '
        info += f'\n splitter: {clf.splitter}, '
        info += f'\nmax depth: {clf.max_depth} '
    elif clf == clfSVM:
        info += f'\nkernel: {clf.kernel}, '
        info += f'\n     C: {clf.C}, '
        info += f'\n   tol: {clf.tol}, '
        info += f'\n  iter: {clf.max_iter} '
    info += '\n'
    print(info)


clfTree = DecisionTreeClassifier(
    criterion="entropy", splitter="random", max_depth=5
)

clfSVM = SVC(
    C=1, kernel="linear", tol=10e-16, max_iter=int(10e6)
)


FOLDS = 5
DIGITS = 5


ds = DataSet(K=FOLDS)

Features = ds.trainFeatures
Targets = ds.trainTarget


clf = clfTree
printInfo(clf)


oSetosa = Observation(name="Setosa")
oVersicolor = Observation(name="Versicolor")
oVirginica = Observation(name="Virginica")
observations = [oSetosa, oVersicolor, oVirginica]
accuracies = []


for k in range(FOLDS):
    trainFeatures = ds.joinData(Features, k)
    trainTargets = ds.joinData(Targets, k)
    testFeatures = Features[k]
    testTargets = Targets[k]

    clf.fit(trainFeatures, trainTargets)
    testPrediction = clf.predict(testFeatures)

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
            trueClassification(observations, targetClass)
        else:
            where_error_str += "X "
            falseClassification(observations, targetClass, predictClass)

    accuracy = AccuracyPOSITIVE / numerosity
    accuracies.append(accuracy)

    print(f'Fold {k}: accuracy = {round(accuracy, DIGITS)}')
    # print(f'Fold {k}: accuracy = {round(accuracy, DIGITS)}')
    # print(f'errors:    {where_error_str}')
    # print(f'predicted: {predict_str}')
    # print(f'target:    {target_str}\n')


for obs in observations:
    print('')
    print(f'Class: {obs.name}')
    print(f' accuracy: {round(obs.accuracy(), DIGITS)}')
    print(f'precision: {round(obs.precision(), DIGITS)}')
    print(f'   recall: {round(obs.recall(), DIGITS)}')
    print(f'       F1: {round(obs.F1(), DIGITS)}')

avgAccuracy = np.mean(accuracies)
stddevAccuracy = np.std(accuracies)
print(
    f'\nOverall mean accuracy: {round(avgAccuracy, DIGITS)}\n' +
    f'standard deviation: {stddevAccuracy}\n'
    )
