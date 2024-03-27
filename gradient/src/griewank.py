from function import Function
import numpy as np


class Griewank(Function):
    def __init__(self, bounds, dimensions, beta, initial_position):
        super().__init__(bounds, dimensions, beta, initial_position)
        self._name = "Griewank"

    def q(self, x_vector):
        d = self._dimensions

        SUM = 0
        for i in range(d):
            SUM += x_vector[i] ** 2
        SUM = SUM / 4000

        PI = 1
        for i in range(d):
            PI = PI * np.cos(x_vector[i] * np.sqrt(i + 1) / (i + 1))

        return SUM - PI + 1

    def grad(self, x_vector, i):
        x = x_vector
        PI_rest = 1
        for j in range(self.d()):
            if j != i:
                PI_rest = PI_rest * np.cos(x[j] * np.sqrt(j + 1) / (j + 1))

        return x[i] / 2000 + np.sin(
            x[i] * np.sqrt(i + 1) / (i + 1)
        ) * PI_rest * np.sqrt(i + 1) / (i + 1)
