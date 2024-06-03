import gym
import QLearning

import numpy as np


SEED = np.random.seed(318407)


mapsize = 64
actionTypes = 4
QSHAPE = (mapsize, actionTypes)


def print_results(Q: QLearning.QL, print_all: bool = False):
    print(Q.Q)
    print(f"\nRewards count: {Q.rewardCounter}")
    indexes = "indeksy:\n"
    values = "moves:\n"
    i = 0
    s = 0
    for m in Q.moves:
        if type(m) is int:
            indexes += f"{i}," + " " * (8 - len(str(i)))
            values += f"{m}," + " " * (8 - len(str(m)))
            s += 1
        i += 1
    indexes += "\nkoniec.\n"
    values += "\nkoniec.\n"
    print(f"suma: {s}\n")
    if print_all:
        print(indexes)
        print("\n")
        print(values)


def main(seed: int = SEED, isShown: bool = False):

    IS_SLIPPERY = True

    if isShown:
        print(f"\nSlippery: {IS_SLIPPERY}\n")

    ENV = gym.make(
        "FrozenLake-v1", desc=None, map_name="8x8", is_slippery=IS_SLIPPERY
    )
    ENV_SHOW = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="8x8",
        is_slippery=IS_SLIPPERY,
        render_mode="human",
    )
    ENV_SHOW.metadata["render_fps"] = 90

    Q = QLearning.QL(
        map_shape=QSHAPE,
        learning_rate=0.9,
        discount=0.97,
        policy="Eps-greedy",  # Eps-greedy | Boltzman
        eps=0.95,
        T=1,
    )

    T = 192
    E = int(1e5)
    E_validate = int(1e4)

    QLearning.run(env=ENV, Q=Q, Tmax=T, Emax=E)

    if isShown:
        print_results(Q)
        print(Q.getLastMoves(200))

    Q.rewardCounter = 0
    Q.eps = 0.001
    QLearning.run(env=ENV, Q=Q, Tmax=T, Emax=E_validate)

    if isShown:
        print_results(Q)
        print(Q.getLastMoves(200))

    results = {
        "seed": seed,
        "Q": Q.Q,
        "rewards": Q.rewardCounter,
        "moves": Q.moves,
    }

    return results


if __name__ == "__main__":
    main(seed=SEED, isShown=True)
