import math
import copy
import numpy as np
import matplotlib.pyplot as plt


class Function:
    def __init__(self, domain, dimensions, beta, initial_position):
        self._domain = domain
        self._dimensions = dimensions
        self._beta = beta
        self._current_position = initial_position

    def d(self):
        return self._dimensions

    def position(self):
        return self._current_position

    def set_position(self, new_position):
        self._current_position = new_position

    def beta(self):
        return self._beta

    def grad(self, x_vector, i):
        return None

    def update_position(self):
        min_bound = self._domain[0]
        max_boud = self._domain[1]
        d = self.d()
        x = copy.deepcopy(self.position())
        for i in range(d):
            upd = self.beta() * self.grad(x, i)
            x[i] = x[i] - upd
            if x[i] < min_bound:
                x[i] = min_bound
            if x[i] > max_boud:
                x[i] = max_boud

        self.set_position(x)
        return x


class Rastrigin(Function):
    def __init__(self, domain, dimensions, beta, initial_position):
        super().__init__(domain, dimensions, beta, initial_position)

    def name(self):
        return "Rastrigin"

    def q(self, x_vector):
        d = self._dimensions
        partial_cost = lambda x: x ** 2 - 10 * math.cos(2 * math.pi * x)

        cost = 10 * d
        for i in range(d):
            cost += partial_cost(x_vector[i])

        return cost

    def f(self, x):
        return self.q(x)

    def grad(self, x_vector, i):
        x = x_vector[i]
        gradient = 2 * x + 20 * math.pi * math.sin(2 * math.pi * x)
        return gradient


class Griewank(Function):
    def __init__(self, domain, dimensions, beta, initial_position):
        super().__init__(domain, dimensions, beta, initial_position)

    def name(self):
        return "Griewank"

    def q(self, x_vector):
        d = self._dimensions

        SUM = 0
        for i in range(d):
            SUM += x_vector[i] ** 2 / 4000

        PI = 1
        for i in range(d):
            PI = PI * math.cos(x_vector[i] / math.sqrt(i + 1))

        return SUM - PI + 1

    def grad(self, x_vector, i):
        x = x_vector
        PI_rest = 1
        for j in range(self.d()):
            if j != i:
                PI_rest = PI_rest * math.cos(x[j] / math.sqrt(j + 1))

        return x[i] / 2000 + math.sin(
            x[i] / math.sqrt(i + 1)
        ) * PI_rest / math.sqrt(i + 1)


class Test:
    def __init__(self, max_iterations, function):
        self._max_iterations = max_iterations
        self._function = function

    def max_interations(self):
        return self._max_iterations

    def function(self):
        return self._function

    def initial_position(self):
        return self.function().position()

    def do_test(self):
        this_func = self.function()
        route_x = []
        route_y = []
        route_q = []
        for t in range(self.max_interations()):
            route_x.append(this_func.position()[0])
            route_y.append(this_func.position()[1])
            route_q.append(this_func.q(this_func.position()))
            this_func.update_position()

        return route_x, route_y, route_q


class Plotter:
    def __init__(self, dimensions, steps, name):
        self._dimensions = dimensions
        self._steps = steps
        self._name = name

    def name(self):
        return self._name

    def d(self):
        return self._dimensions

    def steps(self):
        return self._steps

    def bounds(self):
        if self.name() == "Rastrigin":
            return [-5.12, 5.12]
        else:
            return [-5, 5]

    def q(self, x_vector):
        d = self.d()
        if self.name() == "Rastrigin":
            partial_cost = lambda x: x ** 2 - 10 * math.cos(2 * math.pi * x)
            cost = 10 * d

            for i in range(d):
                cost += partial_cost(x_vector[i])

            return cost

        if self.name() == "Griewank":
            SUM = 0
            for i in range(d):
                SUM += x_vector[i] ** 2
            SUM = SUM / 4000

            PI = 1
            for i in range(d):
                PI = PI * math.cos(x_vector[i] / math.sqrt(i + 1))

            return SUM - PI + 1


    def plot_route(self, route_x, route_y):
        plt.scatter(route_x[1:-1], route_y[1:-1])
        plt.plot(route_x, route_y)
        plt.scatter(route_x[0], route_y[0], color="white")
        plt.scatter(route_x[-1], route_y[-1], color="red")


    def plot(self, route_x=None, route_y=None):
        step = self.steps()
        bounds = self.bounds()
        diff = bounds[1] - bounds[0]
        Z = np.zeros((step + 1, step + 1))
        for i in range(step + 1):
            x = bounds[0] + i * diff / step
            for j in range(step + 1):
                y = bounds[0] + j * diff / step
                # meshgrid[i].append(q(d, [x, y]))
                Z[i][j] = self.q([x, y])

        x = np.linspace(bounds[0], bounds[1], step + 1)
        y = np.linspace(bounds[0], bounds[1], step + 1)

        X, Y = np.meshgrid(x, y)

        fig = plt.figure()
        plt.ion()
        contour = plt.contourf(X, Y, Z, cmap="viridis", levels=100)

        # Dodanie kolorowej skali
        cbar = plt.colorbar(contour)
        cbar.set_label("Function value")

        plt.grid(True)
        plt.xlabel("X")
        plt.ylabel("Y")

        if not (route_x is None or route_y is None):
            self.plot_route(route_x, route_y)

        plt.show()


RastriginPlot = Plotter(2, 400, "Rastrigin")
RastriginPlot.plot()

GriewankPlot = Plotter(2, 400, "Griewank")
GriewankPlot.plot()
