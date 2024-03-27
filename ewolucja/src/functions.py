import numpy as np


class Function:
    def __init__(self, bounds, dimensions, global_optimum=0.0):
        self._name = None
        self._bounds = bounds
        self._dimensions = dimensions
        self._global_optimum = global_optimum

    def name(self) -> str:
        return self._name

    def dimension(self) -> int:
        return self._dimensions

    def bounds(self) -> list:
        return self._bounds

    def bound(self) -> float:
        return max(self.bounds())

    def global_optimum(self) -> float:
        return self._global_optimum

    def get_bounds(self) -> tuple:
        return self.bounds()[0], self.bounds()[1]

    def q(self) -> float:
        pass


class Rastrigin(Function):
    def __init__(self, bounds, dimensions):
        super().__init__(bounds, dimensions)
        self._name = "Rastrigin"

    def q(self, x_vector) -> float:
        d = self.dimension()
        partial_cost = lambda x: x ** 2 - 10 * np.cos(2 * np.pi * x)  # noqa

        cost = 10 * d
        for i in range(d):
            cost += partial_cost(x_vector[i])

        return cost


class Griewank(Function):
    def __init__(self, bounds, dimensions):
        super().__init__(bounds, dimensions)
        self._name = "Griewank"

    def q(self, x_vector) -> float:
        d = self.dimension()

        SUM = 0
        for i in range(d):
            SUM += x_vector[i] ** 2
        SUM = SUM / 4000

        PI = 1
        for i in range(d):
            PI = PI * np.cos(x_vector[i] * np.sqrt(i + 1) / (i + 1))

        return SUM - PI + 1


class DropWave(Function):
    def __init__(self, bounds, dimensions, global_optimum=0.0):
        super().__init__(bounds, dimensions, global_optimum)
        self._name = "DropWave"

    def q(self, x_vector) -> float:
        d = self.dimension()

        square = 0
        for i in range(d):
            square += x_vector[i] ** 2

        numerator = 1 + np.cos(12 * np.sqrt(square))
        denominator = 0.5 * square + 2

        return 1 - numerator / denominator


class Quadratic(Function):
    def __init__(self, bounds, dimensions, global_optimum=0.0):
        super().__init__(bounds, dimensions, global_optimum)
        self._name = "Quadratic"

    def q(self, x_vector) -> float:
        d = self._dimensions

        cost = 0
        for i in range(d):
            cost += (x_vector[i]) ** 2

        return cost
