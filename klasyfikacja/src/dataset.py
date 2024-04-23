from ucimlrepo import fetch_ucirepo
# import pandas as pd
import numpy as np
np.random.seed(318407)


iris = fetch_ucirepo(id=53)
dataset = iris.data

original = dataset.original
feature = dataset.features
target = dataset.targets

shuffled = original.sample(frac=1)
grouped_sets = np.array_split(shuffled, 5)

validation_group = grouped_sets.pop(-1)
training_group = grouped_sets
