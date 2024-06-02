import gym
import QLearning

import numpy as np


SEED = np.random.seed(318407)


mapsize = 64
actionTypes = 4
QSHAPE = (mapsize, actionTypes)


def print_results(Q: QLearning.QL):
    print(f"\nRewards count: {Q.rewardCounter}")
    print(Q.Q)
    indexes = "indeksy:\n"
    values = "moves:\n"
    i = 0
    s = 0
    for m in Q.moves:
        if m > 0:
            indexes += f"{i}," + " "*(8-len(str(i)))
            values += f"{m}," + " "*(8-len(str(m)))
            s += 1
        i += 1
    indexes += "\nkoniec.\n"
    values += "\nkoniec.\n"
    print(f"suma: {s}\n")


if __name__ == "__main__":

    IS_SLIPPERY = True
    print(f"Slippery: {IS_SLIPPERY}")

    ENV = gym.make(
        "FrozenLake-v1", desc=None, map_name="8x8", is_slippery=IS_SLIPPERY
    )
    ENV_SHOW = gym.make(
        "FrozenLake-v1", desc=None, map_name="8x8", is_slippery=IS_SLIPPERY, render_mode='human'  # noqa
    )
    ENV_SHOW.metadata["render_fps"] = 90

    Q = QLearning.QL(
        map_shape=QSHAPE,
        learning_rate=0.9,
        discount=0.97,
        policy="Eps-greedy",  # Eps-greedy | Boltzman
        eps=0.9,
        T=2,
    )

    E = int(1e4)
    T = 192

    QLearning.run(env=ENV, Q=Q, Tmax=T, Emax=E)

    print_results(Q)
    print(Q.getLastMoves(200))

    Q.eps = 0.001
    QLearning.run(env=ENV, Q=Q, Tmax=T, Emax=int(10e4))

    print_results(Q)
    print(Q.getLastMoves(200))
