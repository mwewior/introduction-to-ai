import numpy as np
import random
import copy

from functions import Function
from particle import Particle


class Evolution:
    def __init__(
        self,
        function: Function,
        iterations: int,
        population_count: int,
        mutation_factor: float,
        crossing_factor: float,
        mutation_sigma: float = None,
        elite_succession: int = 1,
    ):
        def generate_population() -> list:
            bound = function.bound()
            population = []
            for id in range(population_count):
                x1 = np.random.uniform(-bound, bound)
                x2 = np.random.uniform(-bound, bound)
                particle = Particle(id + 1, [x1, x2], function)
                particle.set_value()
                population.append(particle)
            return population

        self._function = function
        self._iterations = iterations
        self._population_count = population_count
        self._population = generate_population()
        self._mutation_factor = mutation_factor
        self._mutation_sigma = mutation_sigma
        self._crossing_factor = crossing_factor
        self._elite_successors = elite_succession
        self._current_best_fitness = None
        self._history_best_fitness = None

    def function(self) -> Function:
        return self._function

    def iterations(self) -> int:
        return self._iterations

    def pm(self) -> float:
        return self._mutation_factor

    def pc(self) -> float:
        return self._crossing_factor

    def sigma(self) -> float:
        return self._mutation_sigma

    def elites_count(self) -> int:
        return self._elite_successors

    def population(self) -> list:
        return self._population

    def nominal_population_count(self) -> int:
        return self._population_count

    def current_population_count(self) -> int:
        return len(self.population())

    def set_new_population(self, population):
        self._population = population

    def sort_population(self):
        self.population().sort(key=lambda p: p.value())
        self._current_best_fitness = self.population()[0]

    def positions(self):
        x = []
        y = []
        for particle in self.population():
            x.append(particle.position()[0])
            y.append(particle.position()[1])
        return x, y

    def best_fitness(self) -> Particle:
        return self._current_best_fitness

    def best_history_fitness(self) -> Particle:
        history_best = self._history_best_fitness
        if history_best is None:
            return self.best_fitness()
        return history_best

    def set_best_fitness(self, best_fitness):
        self._current_best_fitness = best_fitness
        if self._history_best_fitness is None:
            self._history_best_fitness = best_fitness
        elif best_fitness.value() < self._history_best_fitness.value():
            self._history_best_fitness = best_fitness

    def fitness_func(self):
        for particle in self.population():
            particle.set_value()
        self.sort_population()

        best_fitness = self.best_history_fitness()
        best_fitness.set_rank(1)
        best_value = best_fitness.value()

        for particle in self.population()[1::]:
            r = 1 + (particle.value() - best_value) ** 2
            particle.set_rank(r)

        """if we would know global optimum at start:"""
        # for particle in self.population():
        #     r = 0
        #     for d in range(self.function().dimension()):
        #         r += particle.position()[d]**2
        #     particle.set_rank(np.sqrt(r))

    def crossover(self, first_parent: Particle) -> Particle:
        a = np.random.random()
        if a < self.pc():

            second_parent = random.choice(self.population())
            possible_position = []

            for parent in [first_parent, second_parent]:
                for d in range(self.function().dimension()):
                    possible_position.append(parent.position()[d])

            particle = Particle(
                first_parent.id(), [None, None], self.function()
            )

            for d in range(self.function().dimension()):
                particle.set_position(d, random.choice(possible_position))

        else:
            particle = first_parent

        particle.fix_position()
        return particle

    def mutation(self, particle: Particle) -> Particle:
        a = np.random.random()
        if a < self.pm():
            for i in range(self.function().dimension()):
                mut = self.sigma() * np.random.randn()
                particle.position()[i] += mut
            particle.fix_position()
        return particle

    def reproduction(self):
        reproduction = []
        while (
            len(reproduction)
            < self.nominal_population_count() - self.elites_count()
        ):
            particle1 = random.choice(self.population())
            particle2 = random.choice(self.population())

            if particle1.rank() <= particle2.rank():
                reproduction.append(copy.deepcopy(particle1))
            else:
                reproduction.append(copy.deepcopy(particle2))

        self.set_new_population(reproduction)

    def succession(self, new_population: list):
        for index in range(self.elites_count()):
            new_population.insert(index, self.best_fitness())
        self.set_new_population(new_population)

    def new_generation(self):
        self.fitness_func()
        self.reproduction()
        successing_population = []

        for particle in self.population():
            crossed_particle = self.crossover(particle)
            mutated_particle = self.mutation(crossed_particle)
            successing_population.append(mutated_particle)

        self.succession(successing_population)
