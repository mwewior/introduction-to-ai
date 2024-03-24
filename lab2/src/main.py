import numpy as np
import matplotlib.pyplot as plt

from functions import Rastrigin, Griewank, DropWave
from evolution import Evolution
from plotter import Plotter


RAST_BOUNDS = [-5.12, 5.12]
GRIE_BOUNDS = [-50, 50]
DROP_BOUNDS = [-4, 4]
DIMENSIONS = 2


DRAW = True
PLOT_STEPS = 200
PLOT_COLOR = "turbo"
PARTICLES_COLOR = "white"
BEST_COLOR = "magenta"
# viridis # plasma # inferno # magma # cividis
# cubehelix # gist_earth # turbo # terrain


ITERATIONS = 50
POPULATION = 25
PC = 0.4
PM = 0.7
SIGMA = 0.5
ELITE_SUCCESSION = 1


testing_functions = [
    DropWave(DROP_BOUNDS, DIMENSIONS),
    Griewank(GRIE_BOUNDS, DIMENSIONS),
    Rastrigin(RAST_BOUNDS, DIMENSIONS),
]


for testfun in testing_functions:

    plot = Plotter(testfun, PLOT_STEPS)
    draw = DRAW

    sigma = SIGMA
    if testfun.name() == "Griewank":
        sigma = 2

    population = Evolution(testfun, ITERATIONS, POPULATION, PM, PC, sigma)
    best_fitnesses = []

    if draw:
        plot_points, plot_best_point = plot.draw_online()

    t = 0
    while t < population.iterations():
        t += 1
        population.new_generation()
        best_fitnesses.append(population.best_fitness().value())

        if draw:
            plt.title(f"Generation: {t+1}")
            x, y = population.positions()
            plot_points.set_offsets(np.column_stack((x, y)))
            plot_best_point.set_offsets(
                np.column_stack((
                    population.best_fitness().position()[0],
                    population.best_fitness().position()[1],
                ))
            )
            plot.pause()

    if draw:
        plot.off()
    print(population.best_fitness().value())
plt.show(block=True)
