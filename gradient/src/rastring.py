from function import Function
import numpy as np


class Rastrigin(Function):
    def __init__(self, bounds, dimensions, beta, initial_position):
        super().__init__(bounds, dimensions, beta, initial_position)
        self._name = "Rastrigin"

    def q(self, x_vector):
        d = self._dimensions
        partial_cost = lambda x: x ** 2 - 10 * np.cos(2 * np.pi * x) # noqa

        cost = 10 * d
        for i in range(d):
            cost += partial_cost(x_vector[i])

        return cost

    def grad(self, x_vector, i):
        x = x_vector[i]
        gradient = 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
        return gradient
