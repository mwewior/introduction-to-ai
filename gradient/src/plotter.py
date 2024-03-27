import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, dimensions, steps, function, bounds=None):
        self._dimensions = dimensions
        self._steps = steps
        self._function = function
        self._bounds = bounds

    def funciton(self):
        return self._function

    def name(self):
        return self.funciton().name()

    def d(self):
        return self._dimensions

    def steps(self):
        return self._steps

    def bounds(self):
        if self._bounds is not None:
            return self._bounds
        return self.funciton().bounds()

    def q(self, x_vector):
        return self.funciton().q(x_vector)

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
            y = bounds[0] + i * diff / step
            for j in range(step + 1):
                x = bounds[0] + j * diff / step
                Z[i][j] = self.q([x, y])

        x = np.linspace(bounds[0], bounds[1], step + 1)
        y = np.linspace(bounds[0], bounds[1], step + 1)

        X, Y = np.meshgrid(x, y)

        plt.figure()
        plt.ion()

        contour = plt.contourf(X, Y, Z, cmap="plasma", levels=100)
        cbar = plt.colorbar(contour)
        cbar.set_label("Function value")

        plt.grid(True)
        plt.xlabel("x1")
        plt.ylabel("x2", rotation=0)

        if not (route_x is None or route_y is None):
            self.plot_route(route_x, route_y)

        plt.show()
