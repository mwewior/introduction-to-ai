from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import numpy as np
from typing import List
import copy
import yaml

from dataset import DataSet
from observation import Observation
import printData


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


def loadParams(file_path: str = "../parameters.yaml"):
    with open(file_path, 'r') as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)
    return parameters


parameters = loadParams()
FOLDS = parameters["FOLDS"]
DIGITS = parameters["DIGITS"]


clfTree = DecisionTreeClassifier(
    criterion="entropy", splitter="random", max_depth=5
)

clfSVM = SVC(
    C=1, kernel="linear", tol=10e-16, max_iter=int(10e6)
)

clf = clfTree
printData.printInfo(clf, clfTree, clfSVM)


ds = DataSet(K=FOLDS)

Features = ds.trainFeatures
Targets = ds.trainTarget


# classificationSetosa = Observation(name="Setosa")
# classificationVersicolor = Observation(name="Versicolor")
# classificationVirginica = Observation(name="Virginica")
# classification = [
#     classificationSetosa,
#     classificationVersicolor,
#     classificationVirginica
# ]


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

metricStatistic = copy.deepcopy(metrics)


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
    print(f'Fold {k+1}: overall fold accuracy = {round(accuracy, DIGITS)}')
    # printPredictions(accuracy, where_error_str, predict_str, target_str)
    # printMetricsPerFold(observations)

    for obs in observations:
        metrics[obs.name]["accuracy"].append(obs.accuracy())
        metrics[obs.name]["precision"].append(obs.precision())
        metrics[obs.name]["recall"].append(obs.recall())
        metrics[obs.name]["F1"].append(obs.F1())


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


printData.printOverallMetrix(metricStatistic)


avgAccuracy = np.mean(accuracies)
stddevAccuracy = np.std(accuracies)
print(
    f'\nOverall mean accuracy: {round(avgAccuracy, DIGITS)}\n' +
    f'standard deviation: {stddevAccuracy}\n'
    )
