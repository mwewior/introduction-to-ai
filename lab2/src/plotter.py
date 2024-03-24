import matplotlib.pyplot as plt
import numpy as np

from functions import Function
from evolution import Evolution


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

    # def axes(self, fig):
    #     # fig = plt.figure()
    #     ax = fig.add_subplot()
    #     ax.set_xlim(self.funciton().bounds()[0], self.funciton().bounds()[1])
    #     ax.set_ylim(self.funciton().bounds()[0], self.funciton().bounds()[1])
    #     return ax

    def plot(self, drow_now=False):

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

        # fig = plt.figure()
        # ax = self.axes(fig)
        contour = plt.contourf(X, Y, Z, cmap=PLOT_COLOR, levels=100)
        cbar = plt.colorbar(contour)
        # contour = ax.contourf(X, Y, Z, cmap=PLOT_COLOR, levels=100)
        # cbar = fig.colorbar(contour)
        cbar.set_label("Function value")

        plt.grid(True)
        plt.xlabel("x1")
        plt.ylabel("x2", rotation=0)

        # ax.grid(True)
        # plt.xlabel("x1")
        # plt.ylabel("x2", rotation=0)

        if not drow_now:
            plt.show()

    def draw_points(self, evolution: Evolution, fig=None):
        # plt.ion()
        # self.plot(drow_online=True)

        X = []
        Y = []
        best_fitnesses = evolution.best_fitness().position()
        for particle in evolution.population():
            X.append(particle.position()[0])
            Y.append(particle.position()[1])
        # ax = self.axes(fig)
        plt.scatter(X, Y, c=PARTICLES_COLOR, marker=".")
        plt.scatter(
            best_fitnesses[0], best_fitnesses[1], c=BEST_COLOR, marker="*"
        )
        # ax.scatter(X, Y, c=PARTICLES_COLOR, marker=".")
        # ax.scatter(
        #   best_fitnesses[0], best_fitnesses[1], c=BEST_COLOR, marker="*"
        # )
        plt.show()

    # def draw_online(
    #     self, evolution: Evolution, gif: bool = True, t: int = None
    #     ):
    #     X = []
    #     Y = []
    #     best_fitnesses = evolution.best_fitness().position()
    #     for particle in evolution.population():
    #         X.append(particle.position()[0])
    #         Y.append(particle.position()[1])
    #     plt.scatter(X, Y, c=PARTICLES_COLOR, marker=".")

    #     if gif:

    #     plt.show()
