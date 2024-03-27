from functions import Function

# import functions


class Particle:
    def __init__(self, id: int, position: list, function: Function):
        self._id = id
        self._position = position
        self._function = function
        self._rank = None
        self.q = None

    def function(self) -> Function:
        return self._function

    def id(self) -> int:
        return self._id

    def position(self) -> list:
        return self._position

    def value(self) -> float:
        return self.q

    def rank(self) -> float:
        return self._rank

    def set_position(self, index, value):
        self._position[index] = value

    def set_positions(self, values):
        if len(values) == self.function().dimension():
            self._position = values

    def set_value(self):
        x = self.position()
        self.q = self.function().q(x)

    def set_rank(self, rank):
        self._rank = rank

    def fix_position(self):
        min_bound, max_bound = self.function().get_bounds()
        """Metoda naprawy rozwiązania przez zawijanie"""
        for d in range(self.function().dimension()):
            while self.position()[d] > max_bound:
                difference = self.position()[d] - max_bound
                self.set_position(d, min_bound + difference)
            while self.position()[d] < min_bound:
                difference = self.position()[d] - min_bound
                self.set_position(d, max_bound + difference)

    def print(self):
        print(
            f"{self.id()} | {self.position()} | {self.value()} | {self.function().name()}" # noqa
        )
