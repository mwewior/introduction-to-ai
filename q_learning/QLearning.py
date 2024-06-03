import numpy as np
import gym
from multiprocessing import Queue


class QL:
    def __init__(
        self,
        map_shape: tuple[int] = (64, 4),
        learning_rate: float = 0.9,
        discount: float = 0.9,
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
        self.rewardCounter = 0
        self.moves = []

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

    def updateState(self, next_state):
        self.state = next_state

    def updateQ(self, next_state: int, action: int, reward: float) -> None:
        lr = self.learning_rate
        gamma = self.discount

        prev = (1 - lr) * self.Q[self.state, action]
        updated = lr * (reward + gamma * max(self.Q[next_state, :]))

        self.Q[self.state, action] = prev + updated
        self.updateState(next_state)

    def changeParameters(
        self,
        learningRate: float = None,
        discount: float = None,
        eps: float = None,
        T: int = None
    ):
        if learningRate is not None:
            self.learning_rate = learningRate
        if discount is not None:
            self.discount = discount
        if eps is not None:
            self.eps = eps
        if T is not None:
            self.T = T

    def getLastMoves(self, length):
        return self.moves[-length:]


def run(
    env: gym.Env,   Q: QL = None,
    Tmax: int = 50, Emax: int = 20000
):

    moves = []
    for e in range(Emax):

        observation, info = env.reset()
        Q.updateState(observation)
        episodeMoves = 0
        moves.append(0)

        for t in range(Tmax):
            action = Q.chooseAction()
            episodeMoves += 1
            observation, reward, terminated, truncated, info = env.step(action)

            # if truncated:
            #     reward = reward * -0.1

            Q.updateQ(observation, action, reward)

            if terminated:
                if reward == 1:
                    Q.rewardCounter += 1
                    moves[e] = episodeMoves
                else:
                    moves[e] = "D"
                break

            if truncated or t == Tmax-1:
                moves[e] = "_"
                break

        # print(f"Episode: {e} | Last reward: {reward} | Rewards gained: {Q.rewardCounter}")    # noqa
        # if round(e % 1e3) == 0:
        #     print(f"Episode: {int(e / 1e3)}e3 | Rewards gained: {Q.rewardCounter}")           # noqa

    Q.moves = moves
    env.close()


def run_parralel(
    seed: int,      queue: Queue,
    env: gym.Env,   Q: QL = None,
    Tmax: int = 50, Emax: int = 20000
):

    np.random.seed(seed)

    moves = []
    for e in range(Emax):

        observation, info = env.reset()
        Q.updateState(observation)
        episodeMoves = 0
        moves.append(0)

        for t in range(Tmax):
            action = Q.chooseAction()
            episodeMoves += 1
            observation, reward, terminated, truncated, info = env.step(action)

            # if truncated:
            #     reward = reward * -0.1

            Q.updateQ(observation, action, reward)

            if reward == 1:
                Q.rewardCounter += 1
                moves[e] = episodeMoves
                break

            if terminated:
                moves[e] = "D"
                break

            if truncated or t == Tmax-1:
                moves[e] = "X"
                break

    Q.moves = moves
    result = {
        'seed': seed,
        'Q': Q.Q,
        'rewardCounter': Q.rewardCounter,
        'moves': Q.moves
    }
    queue.put(result)
    env.close()
