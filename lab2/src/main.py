from functions import Rastrigin, Griewank, DropWave

# from particle import Particle
from evolution import Evolution
from plotter import Plotter


RAST_BOUNDS = [-5.12, 5.12]
GRIE_BOUNDS = [-50, 50]
DROP_BOUNDS = [-4, 4]
DIMENSIONS = 2


PLOT_STEPS = 200
PLOT_COLOR = "turbo"
PARTICLES_COLOR = "white"
BEST_COLOR = "magenta"
# viridis # plasma # inferno # magma # cividis
# cubehelix # gist_earth # turbo # terrain


ITERATIONS = 1000
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
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.set_xlim(testfun.bounds()[0], testfun.bounds()[1])
    # ax.set_ylim(testfun.bounds()[0], testfun.bounds()[1])

    sigma = SIGMA
    if testfun.name() == "Griewank":
        sigma = 2

    population = Evolution(testfun, ITERATIONS, POPULATION, PM, PC, sigma)
    best_fitnesses = []
    t = 0

    # plot.plot(False)

    while t < population.iterations():
        t += 1
        population.new_generation()
        best_fitnesses.append(population.best_fitness().value())

        # X = []
        # Y = []
        # for particle in population.population():
        #     X.append(particle.position()[0])
        #     Y.append(particle.position()[1])
        # scat = ax.scatter(X, Y, c=PARTICLES_COLOR, marker=".")
        # plt.title(f'Generation: {t+1}')
        # # plt.savefig(
        #     f'/home/szczygiel/pop/projekt/pop_projekt/tmp_figures/fig{i}.png'
        # )
        # plt.pause(0.001)
        # scat.remove()

    # plt.show()

    print(population.best_fitness().value())
    # plot.draw_online(population)

    plot.plot(True)
    plot.draw_points(population)
