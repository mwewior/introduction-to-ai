from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import numpy as np
from typing import List
import copy

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
    info += '\n\n'
    print(info)


def printPredictions(accuracy, errors, predictions, targets):
    info = ""
    info += f'\naccuracy = {round(accuracy, DIGITS)}'
    info += f'\nerrors:    {errors}'
    info += f'\npredicted: {predictions}'
    info += f'\ntarget:    {targets}'
    print(info)


def printMetricsPerFold(observations: List[Observation]):
    info = ""
    for obs in observations:
        info += f'\nClass: {obs.name}'
        info += f'\n accuracy: {round(obs.accuracy(), DIGITS)}'
        info += f'\nprecision: {round(obs.precision(), DIGITS)}'
        info += f'\n   recall: {round(obs.recall(), DIGITS)}'
        info += f'\n       F1: {round(obs.F1(), DIGITS)}'
        info += '\n'
    print(info)


def printOverallMetrix(data):
    info = ""
    for className in data:
        info += f'\nClass: {className}'
        info += '\n           mean \t standard deviation'
        info += '\n accuracy: '
        info += f'{round(data[className]["accuracy"]["mean"], DIGITS)}, \t '
        info += f'{round(data[className]["accuracy"]["std"], DIGITS)}'
        info += '\nprecision: '
        info += f'{round(data[className]["precision"]["mean"], DIGITS)}, \t '
        info += f'{round(data[className]["precision"]["std"], DIGITS)}'
        info += '\n   recall: '
        info += f'{round(data[className]["recall"]["mean"], DIGITS)}, \t '
        info += f'{round(data[className]["recall"]["std"], DIGITS)}'
        info += '\n       F1: '
        info += f'{round(data[className]["F1"]["mean"], DIGITS)}, \t '
        info += f'{round(data[className]["F1"]["std"], DIGITS)}'
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


# oSetosa = Observation(name="Setosa")
# oVersicolor = Observation(name="Versicolor")
# oVirginica = Observation(name="Virginica")
# observations = [oSetosa, oVersicolor, oVirginica]


accuracies = []

single_dict = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'F1': []
}

metrics = {
    'Setosa': copy.deepcopy(single_dict),
    'Versicolor': copy.deepcopy(single_dict),
    'Virginica': copy.deepcopy(single_dict)
}

metricStatistic = {
    'Setosa': copy.deepcopy(single_dict),
    'Versicolor': copy.deepcopy(single_dict),
    'Virginica': copy.deepcopy(single_dict)
}

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

    oSetosa = Observation(name="Setosa")
    oVersicolor = Observation(name="Versicolor")
    oVirginica = Observation(name="Virginica")
    observations = [oSetosa, oVersicolor, oVirginica]

    for i in range(numerosity):
        targetClass = testTargets[i]
        predictClass = testPrediction[i]

        predict_str += f'{predictClass} '
        target_str += f'{targetClass} '

        if (predictClass == targetClass):
            AccuracyPOSITIVE += 1
            where_error_str += "  "
            trueClassification(observations, targetClass)
        else:
            where_error_str += "X "
            falseClassification(observations, targetClass, predictClass)

    accuracy = AccuracyPOSITIVE / numerosity
    accuracies.append(accuracy)

    for obs in observations:
        metrics[obs.name]["accuracy"].append(obs.accuracy())
        metrics[obs.name]["precision"].append(obs.precision())
        metrics[obs.name]["recall"].append(obs.recall())
        metrics[obs.name]["F1"].append(obs.F1())

    print(f'Fold {k}: overall fold accuracy = {round(accuracy, DIGITS)}')
    # printPredictions(accuracy, where_error_str, predict_str, target_str)
    # printMetricsPerFold(observations)

for obs in observations:
    metricStatistic[obs.name]["accuracy"] = {
        "mean": np.mean(metrics[obs.name]["accuracy"]),
        "std": np.std(metrics[obs.name]["accuracy"])
        }
    metricStatistic[obs.name]["precision"] = {
        "mean": np.mean(metrics[obs.name]["precision"]),
        "std": np.std(metrics[obs.name]["precision"])
        }
    metricStatistic[obs.name]["recall"] = {
        "mean": np.mean(metrics[obs.name]["recall"]),
        "std": np.std(metrics[obs.name]["recall"])
        }
    metricStatistic[obs.name]["F1"] = {
        "mean": np.mean(metrics[obs.name]["F1"]),
        "std": np.std(metrics[obs.name]["F1"])
        }


printOverallMetrix(metricStatistic)


avgAccuracy = np.mean(accuracies)
stddevAccuracy = np.std(accuracies)
print(
    f'\nOverall mean accuracy: {round(avgAccuracy, DIGITS)}\n' +
    f'standard deviation: {stddevAccuracy}\n'
    )
