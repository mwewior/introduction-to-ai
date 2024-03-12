import copy


class Function:
    def __init__(
        self, bounds, dimensions, beta, initial_position, global_optimum=0.0
    ):
        self._name = None
        self._bounds = bounds
        self._dimensions = dimensions
        self._beta = beta
        self._current_position = initial_position
        self._global_optimum = global_optimum

    def name(self):
        return self._name

    def d(self):
        return self._dimensions

    def bounds(self):
        return self._bounds

    def position(self):
        return self._current_position

    def set_position(self, new_position):
        self._current_position = new_position

    def beta(self):
        return self._beta

    def set_beta(self, new_beta):
        self._beta = new_beta

    def global_optimum(self):
        return self._global_optimum

    def grad(self, x_vector, i):
        return None

    def update_position(self):
        min_bound = self._bounds[0]
        max_bound = self._bounds[1]
        d = self.d()
        x = copy.deepcopy(self.position())

        for i in range(d):
            upd = self.beta() * self.grad(x, i)
            x[i] = x[i] - upd

            while x[i] > max_bound:
                diff = x[i] - max_bound
                x[i] = min_bound + diff
            while x[i] < min_bound:
                diff = x[i] - min_bound
                x[i] = max_bound + diff

        self.set_position(x)
        return x
