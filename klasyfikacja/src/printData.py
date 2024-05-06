from typing import List
try:
    from observation import Observation
    from dataset import loadParams
except ModuleNotFoundError:
    from src.observation import Observation
    from src.dataset import loadParams

parameters = loadParams()
DIGITS = parameters["DIGITS"]
SEED = parameters["SEED"]


def printInfo(clf, clfTree, clfSVM) -> None:
    info = f"\nSeed: {SEED}\n"
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
    info += f'\n    Overall accuracy = {round(accuracy, DIGITS)}'
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


def printStatisticMetricsMerged(data):
    info = ""
    info += '\n---------------------- SUMMARY -----------------------\n\n'
    info += '\n           mean \t standard deviation'
    info += '\n accuracy: '
    info += f' {round(data["accuracy"]["mean"], DIGITS)},\t '
    info += f' {round(data["accuracy"]["std"], DIGITS)}'
    info += '\nprecision: '
    info += f' {round(data["precision"]["mean"], DIGITS)},\t '
    info += f' {round(data["precision"]["std"], DIGITS)}'
    info += '\n   recall: '
    info += f' {round(data["recall"]["mean"], DIGITS)},\t '
    info += f' {round(data["recall"]["std"], DIGITS)}'
    info += '\n       F1: '
    info += f' {round(data["F1"]["mean"], DIGITS)},\t '
    info += f' {round(data["F1"]["std"], DIGITS)}'
    info += '\n'
    print(info)


def printStatisticMetricsEach(data):
    info = ""
    for className in data:
        info += f'\nClass: {className}'
        info += '\n           mean \t standard deviation'
        info += '\n accuracy: '
        info += f' {round(data[className]["accuracy"]["mean"], DIGITS)},\t '
        info += f' {round(data[className]["accuracy"]["std"], DIGITS)}'
        info += '\nprecision: '
        info += f' {round(data[className]["precision"]["mean"], DIGITS)},\t '
        info += f' {round(data[className]["precision"]["std"], DIGITS)}'
        info += '\n   recall: '
        info += f' {round(data[className]["recall"]["mean"], DIGITS)},\t '
        info += f' {round(data[className]["recall"]["std"], DIGITS)}'
        info += '\n       F1: '
        info += f' {round(data[className]["F1"]["mean"], DIGITS)},\t '
        info += f' {round(data[className]["F1"]["std"], DIGITS)}'
        info += '\n'
    print(info)


def printOverallAccuracy(avgAccuracy, stddevAccuracy):
    info = ""
    info += f'Overall mean accuracy: {round(avgAccuracy, DIGITS)}\n'
    info += f'   Standard deviation: {round(stddevAccuracy, DIGITS)}\n'
    print(info)
