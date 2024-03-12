class ThisTest:
    def __init__(self, max_iterations, function, epsilon=None):
        self._max_iterations = max_iterations
        self._last_iteration = max_iterations
        self._function = function
        self._epsilon = epsilon
        self._routes = None

    def max_interations(self):
        return self._max_iterations

    def function(self):
        return self._function

    def initial_position(self):
        return self.function().position()

    def set_routes(self, routes):
        self._routes = routes

    def get_routes(self):
        return self._routes

    def set_last_iteration(self, iteration_id):
        self._last_iteration = iteration_id

    def get_last_iteration(self):
        return self._last_iteration

    def epsilon(self):
        return self._epsilon

    def close_enough(self, last_qs):
        if len(last_qs) < 3:
            return False
        if abs(last_qs[-1] - last_qs[-2]) < self.epsilon():
            return True
        return False

    def break_faster(self, route_q, t):
        if self.epsilon() is not None:
            if self.close_enough(route_q):
                self.set_last_iteration(t)
                return True
        return False

    def do_test(self):
        this_func = self.function()
        route_x = []
        route_y = []
        route_q = []

        for t in range(self.max_interations()):
            x = this_func.position()
            q = this_func.q(x)
            route_x.append(x[0])
            route_y.append(x[1])
            route_q.append(q)
            this_func.update_position()

            if self.break_faster(route_q, t):
                break

        self.set_routes([route_x, route_y, route_q])

        return route_x, route_y, route_q
