# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC

import numpy as np
from typing import List
import copy

try:
    from dataset import DataSet, loadParams
    from observation import Observation
    # import printData
except ModuleNotFoundError:
    from src.dataset import DataSet, loadParams
    from src.observation import Observation
    # from src import printData


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


parameters = loadParams()
FOLDS = parameters["FOLDS"]
DIGITS = parameters["DIGITS"]
SEED = parameters["SEED"]
np.random.seed(SEED)


def classification(clf, dataSet=None):
    if dataSet is None:
        # local dataset division
        ds = DataSet(K=FOLDS)
        Features = ds.trainFeatures
        Targets = ds.trainTarget
    else:
        # sklearn dataset ???
        ds = dataSet
        Features = ds[0]
        Targets = ds[1]

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

    mergedClasses = copy.deepcopy(single_dict)

    def statistics(stats, data):
        stats["accuracy"] = {
            "mean": np.mean(data["accuracy"]),
            "std": np.std(data["accuracy"]),
            "data": data["accuracy"]
            }
        stats["precision"] = {
            "mean": np.mean(data["precision"]),
            "std": np.std(data["precision"]),
            "data": data["precision"]
            }
        stats["recall"] = {
            "mean": np.mean(data["recall"]),
            "std": np.std(data["recall"]),
            "data": data["recall"]
            }
        stats["F1"] = {
            "mean": np.mean(data["F1"]),
            "std": np.std(data["F1"]),
            "data": data["F1"]
            }
        return stats

    for obs in observations:
        mergedClasses["accuracy"] += metrics[obs.name]["accuracy"]
        mergedClasses["precision"] += metrics[obs.name]["precision"]
        mergedClasses["recall"] += metrics[obs.name]["recall"]
        mergedClasses["F1"] += metrics[obs.name]["F1"]

        metricStatistic[obs.name] = statistics(
            stats=metricStatistic[obs.name],
            data=metrics[obs.name]
        )

    mergedStatistics = copy.deepcopy(single_dict)
    mergedStatistics = statistics(
        stats=mergedStatistics, data=mergedClasses
    )

    return mergedStatistics
