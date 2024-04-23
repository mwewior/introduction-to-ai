import copy
from ucimlrepo import fetch_ucirepo

# import pandas as pd
import numpy as np

np.random.seed(318407)


K = 5


iris = fetch_ucirepo(id=53)
dataset = iris.data

original = dataset.original
feature = dataset.features
target = dataset.targets

shuffled = original.sample(frac=1)
grouped_sets = np.array_split(shuffled, K + 1)

test_group = grouped_sets.pop(-1)
cross_group = grouped_sets


def learning():
    # tutaj to jakoś się uczy, daje parametry potem
    return 0


def testing():
    # dla uzyskanych parametrów klasyfikacji przeprowadzamy
    # badanie na danych testowych i oceniamy jakość
    return 0


parameters = []
for i in range(K):
    train_group = copy.deepcopy(cross_group)
    validation_group = train_group.pop(K)
    parameters.append(learning())
test_results = testing()
