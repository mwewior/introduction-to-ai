from thistest import ThisTest as Test
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class Experiment:
    def __init__(
        self,
        function,
        iterations: int,
        step: float,
        experiment_domain: float,
        epsilon: float,
        show: bool,
    ):
        self._function = function
        self._iterations = iterations
        self._step = step
        self._domain = experiment_domain
        self._epsilon = epsilon
        self._show = show

    def test_function(self):
        return self._function

    def iterations(self):
        return self._iterations

    def step(self):
        return self._step

    def domain(self):
        return self._domain

    def epsilon(self):
        return self._epsilon

    def show_flag(self):
        return self._show

    def specify_function(self, function, x=None, beta=None, MSE=False):
        if x is not None:
            function.set_position(x)
        if beta is not None:
            function.set_beta(beta)

    def plot(self, data, xlabel, scatter=False):
        X = data[0]
        Y = data[1::]
        plt.figure()
        plt.grid(True)
        for y in Y:
            plt.plot(X, y)
        if scatter:
            for y in Y:
                plt.scatter(X, y, c="#ff7f0e")
        plt.ylabel("Error")
        plt.xlabel(xlabel)
        plt.show()

    def beta_experiment(self, init_points, MSE=False):
        beta_and_qs = []
        for _ in range(len(init_points) + 1):
            beta_and_qs.append([])

        int_range = int(self.domain() / self.step())

        for i in range(int_range + 1):

            beta = (i + 1) * self.step()
            beta_and_qs[0].append(beta)

            for t in range(len(init_points)):
                test_func = self.test_function()
                self.specify_function(test_func, init_points[t], beta)

                test = Test(self.iterations(), test_func, self.epsilon())
                route_x, route_y, route_q = test.do_test()

                if MSE:
                    error = route_q[-1] ** 2
                else:
                    error = abs(route_q[-1])

                beta_and_qs[t + 1].append(error)

        if self.show_flag():
            self.plot(beta_and_qs, xlabel="Beta")

        return beta_and_qs

    def initial_position_experiment(
        self, inits_count, beta, distribution_bound, MSE=False
    ):
        xy_results = [[], [], []]
        # bound = self.test_function().bounds()[1]

        for _ in range(inits_count):

            # x1 = np.random.uniform(-bound, bound)
            # x2 = np.random.uniform(-bound, bound)
            # x2 = np.random.normal(0, 3.0, 1)
            # x1 = np.random.normal(0, 3.0, 1)
            X = get_truncated_normal(
                0, 4, -distribution_bound, distribution_bound
            )
            x1 = X.rvs()
            x2 = X.rvs()
            position = [x1, x2]
            norm = np.sqrt(x1 ** 2 + x2 ** 2)

            xy_results[0].append(position)
            xy_results[1].append(norm)

            test_func = self.test_function()
            self.specify_function(test_func, position, beta)

            test = Test(self.iterations(), test_func, self.epsilon())
            route_x, route_y, route_q = test.do_test()

            if MSE:
                error = route_q[-1] ** 2
            else:
                error = abs(route_q[-1])

            xy_results[2].append(error)

        zipped = list(zip(xy_results[1], xy_results[2]))
        dist_q = sorted(zipped, key=lambda x: x[0])
        xs_and_qs = [[i for i, j in dist_q], [j for i, j in dist_q]]

        if self.show_flag():
            self.plot(
                xs_and_qs, xlabel="Distance from global optimum", scatter=True
            )

        return xs_and_qs
