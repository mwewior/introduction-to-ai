from function import Function


class Quadratic(Function):
    def __init__(
        self, bounds, dimensions, beta, initial_position, global_optimum=0
    ):
        super().__init__(
            bounds, dimensions, beta, initial_position, global_optimum
        )
        self._name = "Quadratic"

    def q(self, x_vector):
        d = self._dimensions

        cost = 0
        for i in range(d):
            cost += (x_vector[i]) ** 2

        return cost

    def grad(self, x_vector, i):
        x = x_vector[i]
        gradient = 2 * x
        return gradient
