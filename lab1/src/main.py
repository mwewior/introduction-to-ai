from griewank import Griewank
from rastring import Rastrigin
from quadratic import Quadratic
from thistest import ThisTest as Test
from experiment import Experiment
from plotter import Plotter


DIMENSIONS = 2
SHOW_PLOTS = False


dimensions = DIMENSIONS
COUNT = 200
show_plots = SHOW_PLOTS


bounds = [-5.0, 5.0]
iterations = 500

step = 0.1
domain = 4
epsilon = 10 ** (-11)

beta_init = 1.4
x_init = [[-0.4, 1.0], [0.7, -2.9], [1.5, 2.2], [0.0, 4.0]]
distribution_bound = bounds[1]


test_func = Griewank(bounds, DIMENSIONS, beta=None, initial_position=None)
current_experiment = Experiment(
    test_func, iterations, step, domain, epsilon, show_plots
)
G_beta_L1 = current_experiment.beta_experiment(x_init)
G_dist_L1 = current_experiment.initial_position_experiment(
    COUNT, beta_init, distribution_bound
)
G_beta_MSE = current_experiment.beta_experiment(x_init, MSE=True)
G_dist_MSE = current_experiment.initial_position_experiment(
    COUNT, beta_init, distribution_bound, MSE=True
)


bounds = [-5.12, 5.12]
iterations = 500

step = 0.0001
domain = 0.016
epsilon = 10 ** (-11)

beta_init = 0.004
x_init = [[-0.1, 0.2], [-4.3, -4.3], [-1.0, -1.0], [1.5, 0.5]]
distribution_bound = bounds[1]

test_func = Rastrigin(bounds, DIMENSIONS, beta=None, initial_position=None)
current_experiment = Experiment(
    test_func, iterations, step, domain, epsilon, show_plots
)
R_beta_L1 = current_experiment.beta_experiment(x_init)
R_dist_L1 = current_experiment.initial_position_experiment(
    COUNT, beta_init, distribution_bound
)
R_beta_MSE = current_experiment.beta_experiment(x_init, MSE=True)
R_dist_MSE = current_experiment.initial_position_experiment(
    COUNT, beta_init, distribution_bound, MSE=True
)


STEPS = 500

# ## Griewank:

bound = 5.0
gbounds = [-bound, bound]
x_init = [0.7, -2.9]
beta = 1.3

function = Griewank(gbounds, DIMENSIONS, beta, x_init)
test = Test(50, function)
route_x, route_y, route_q = test.do_test()
if show_plots:
    plot = Plotter(DIMENSIONS, STEPS, test.function(), bounds=None)
    plot.plot(route_x, route_y)
    close_plot = Plotter(DIMENSIONS, STEPS, test.function(), bounds=[-3, 2])
    close_plot.plot(route_x, route_y)


# ## Rastrigin:

bound = 5.12
rbounds = [-bound, bound]
x_init = [1.5, 0.5]
beta = 0.003

function = Rastrigin(rbounds, DIMENSIONS, beta, x_init)
test = Test(50, function)
route_x, route_y, route_q = test.do_test()
if show_plots:
    plot = Plotter(DIMENSIONS, STEPS, test.function(), bounds=None)
    plot.plot(route_x, route_y)
    close_plot = Plotter(
        DIMENSIONS, STEPS, test.function(), bounds=[-0.1, 1.75]
    )
    close_plot.plot(route_x, route_y)


dimensions = DIMENSIONS
COUNT = 200
STEPS = 500
show_plots = SHOW_PLOTS

bounds = [-4.0, 4.0]
x_init = [1.7, 0.5]
beta = 0.25


function = Quadratic(bounds, DIMENSIONS, beta=beta, initial_position=x_init)
test = Test(800, function)
route_x, route_y, route_q = test.do_test()
if show_plots:
    plot = Plotter(DIMENSIONS, STEPS, test.function(), bounds=bounds)
    plot.plot(route_x, route_y)
