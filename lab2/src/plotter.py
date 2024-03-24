import matplotlib.pyplot as plt
import numpy as np

from functions import Function


PLOT_STEPS = 200
PLOT_COLOR = "turbo"
PARTICLES_COLOR = "white"
BEST_COLOR = "magenta"


class Plotter:
    def __init__(self, function: Function, steps: int, bounds: list = None):
        self._function = function
        self._steps = steps
        self._dimensions = function.dimension()
        self._bounds = bounds
        self._axes = None

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

    def draw_grid(self, drow_now=False):

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

        # fig = plt.figure()
        plt.figure()

        contour = plt.contourf(X, Y, Z, cmap=PLOT_COLOR, levels=100)
        cbar = plt.colorbar(contour)
        cbar.set_label("Wartość funkcji")

        plt.grid(True)
        plt.xlabel("X")
        plt.ylabel("Y")

    def draw_online(self):
        plt.ion()

        self.draw_grid(self)

        # Wyświetlenie punktów
        plot_points = plt.scatter([], [], c=PARTICLES_COLOR, marker=".")
        plot_best_point = plt.scatter([], [], c=BEST_COLOR, marker="*")

        plt.show()
        return plot_points, plot_best_point

    def draw_result(
        self, best_position, best_fitness, min_bound, max_bound, fun
    ):

        self.draw_grid(self)
        # Oznaczenie znalezionego punktu
        plt.scatter(
            [best_position[0]],
            [best_position[1]],
            color="w",
            s=50,
            label="Best Position",
        )

        plt.show()

    def off(self):
        plt.ioff()

    def pause(self):
        plt.pause(0.1)
