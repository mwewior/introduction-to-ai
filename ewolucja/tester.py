import numpy as np


class MyClass:
    def __init__(self, name):
        name = name

    def x(self):
        pass


class Child(MyClass):
    def __init__(self, name, x):
        super().__init__(name)
        self._x = x

    def x(self):
        return self._x ** 2


class MyO:
    def __init__(self, func: MyClass):
        self._func = func

    def func(self):
        return self._func.x()


p = MyClass("parent")
c = Child("c", 12)

# xp = p.x()
# xc = c.x()

t1 = MyO(p)
t2 = MyO(c)

print(t1.func())
print(t2.func())


RAST_BOUNDS = [-5.12, 5.12]
GRIE_BOUNDS = [-50, 50]
DROP_BOUNDS = [-5.12, 5.12]

DIMENSIONS = 2
PLOT_STEPS = 400

ITERATIONS = 1000
POPULATION = 20
PC = 0
PM = 0.1
SIGMA = 0.5
ELITE_SUCCESSION = 1


class Function:
    def __init__(self, bounds, dimensions, global_optimum=0.0):
        self._name = None
        self._bounds = bounds
        self._dimensions = dimensions
        self._global_optimum = global_optimum

    def name(self):
        return self._name

    def dimension(self):
        return self._dimensions

    def bounds(self):
        return self._bounds

    def bound(self):
        return max(self.bounds())

    def global_optimum(self):
        return self._global_optimum

    def q(self):
        pass

    def update_position(self):
        pass


class Griewank(Function):
    def __init__(self, bounds, dimensions):
        super().__init__(bounds, dimensions)
        self._name = "Griewank"

    # TODO

    def q(self, x_vector):
        d = self.dimension()

        SUM = 0
        for i in range(d):
            SUM += x_vector[i] ** 2
        SUM = SUM / 4000
        PI = 1
        for i in range(d):
            PI = PI * np.cos(x_vector[i] * np.sqrt(i + 1) / (i + 1))

        return SUM - PI + 1


class Particle:
    def __init__(self, id: int, position: list, function: Function):
        self._id = id
        self._position = position
        self._value = None
        self._function = function
        self._rank = None

    def id(self):
        return self._id

    def position(self):
        return self._position

    def value(self):
        x = self.position()
        self._value = self._function.q(x)
        return self._value

    def set_rank(self, rank):
        self._rank = rank

    def rank(self):
        return self._rank

    def print(self):
        print(f"x = {self.position()}; q = {self.value()}; r = {self.rank()}")


class Evolution:
    def __init__(
        self,
        function: Function,
        iterations: int,
        population_count: int,
        mutation_factor: float,
        crossing_factor: float,
        mutation_sigma: float = None,
    ):
        def generate_population():
            bound = function.bound()
            population = []
            for id in range(population_count):
                x1 = np.random.uniform(-bound, bound)
                x2 = np.random.uniform(-bound, bound)
                population.append(Particle(id + 1, [x1, x2], function))
            return population

        self._iterations = iterations
        self._population_count = population_count
        self._population = generate_population()
        self._mutation_factor = mutation_factor
        self._mutation_sigma = mutation_factor
        self._crossing_factor = crossing_factor

    def iterations(self):
        return self._iterations

    def population(self):
        return self._population

    def population_count(self):
        return self._population_count

    def pm(self):
        return self._mutation_factor

    def pc(self):
        return self._crossing_factor

    def sigma(self):
        return self._mutation_sigma

    def fitness_func(self):
        self.population().sort(key=lambda p: p.value())
        best_fitness = self.population()[0]
        best_fitness.set_rank(1)
        best_value = best_fitness.value()
        for particle in self.population()[1::]:
            r = 1 + (particle.value() - best_value) ** 2
            particle.set_rank(r)

        # sorted_population = self.population().sort(key = lambda p: p.value())
        # return sorted_population

    def which_to_cross(self):
        pass

    def crossing(self):
        pass

    def mutation(self):
        pass

    def new_population(self):
        pass


# pop = Evolution(
#     Griewank(GRIE_BOUNDS, DIMENSIONS), ITERATIONS, POPULATION, PM, PC, SIGMA
# )  # noqa
# # for particle in pop.population():
# #     print(f'BASE: x1, x2 = ({particle.position()}), q = {particle.value()}, R = {particle.rank()}') # noqa
# # pop.fitness_func()
# # for particle in pop.population():
# #     print(
# #         f"SORT: x1, x2 = ({particle.position()}), q = {particle.value()}, R = {particle.rank()}" # noqa
# #     )

# for particle in pop.population():
#     particle.print()

# population = pop.population()
# sorted_population = population.sort(key=lambda p: p.value())
# for particle in sorted_population:
#     particle.print()

pop = Evolution(
    Griewank(GRIE_BOUNDS, DIMENSIONS), ITERATIONS, POPULATION, PM, PC, SIGMA
)
pop.fitness_func()
for particle in pop.population():
    print(
        f"SORT: x1, x2 = ({particle.position()}), q = {particle.value()}, R = {particle.rank()}" # noqa
    )
