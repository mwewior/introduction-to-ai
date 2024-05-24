import numpy as np


class Q:
    def __init__(
        self,
        map_shape: tuple[int] = (64, 4),
        learning_rate: float = 0.1,
        discount: float = 0.8,
        policy: str = "Eps-greedy",     # "Boltzman"
        eps: float = 0.5,
        T: float = 1.0,
    ):
        self.Q = np.zeros(shape=map_shape)
        self.learning_rate = learning_rate  # beta (or alpha)
        self.discount = discount            # gamma
        self.policy = policy
        self.eps = eps
        self.T = T
        self.state = 0

    def epsGreedy(self) -> int:
        randval = np.random.random()
        if randval < self.eps:
            move = np.random.randint(0, 4)
        else:
            move = self.Q[self.state, :].argmax()
        return move

    def boltzman(self) -> int:
        moves = list(range(4))
        p = []
        for m in moves:
            p.append(np.exp(self.Q[self.state, m] / self.T))

        denominator = sum(p)
        for i in moves:
            p[i] = p[i] / denominator

        move = np.random.choice(moves, p=p)
        return move

    def chooseAction(self) -> int:
        if self.policy == "Eps-greedy":
            return self.epsGreedy()
        if self.policy == "Boltzman":
            return self.boltzman()

    def updateQ(self, next_state: int, action: int, reward: float) -> None:
        lr = self.learning_rate
        gamma = self.discount

        prev = (1 - lr) * self.Q[self.state, action]
        updated = lr * (reward + gamma * max(self.Q[next_state, :]))

        self.Q[self.state, action] = prev + updated
        self.state = next_state

    def reshapeQ(self):
        Qmoves = []
        for i in range(self.Q.shape[1]):
            Qmoves.append(np.array(self.Q[:, i]))
            Qmoves[i].reshape(8, 8)
        return np.stack(Qmoves, axis=1)

    def showBestStrategy():
        # ma zwrócić macierz gdzie po prostu będzie mapa, a na niej cyferki,
        # które mówią który ruch najlepiej wykonać w danym miejscu
        pass
