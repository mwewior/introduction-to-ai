import gym
import QLearning

import numpy as np


SEED = np.random.seed(318407)


mapsize = 64
actionTypes = 4
QSHAPE = (mapsize, actionTypes)


def run(
    env: gym.Env, Q: QLearning.Q = None, Tmax: int = 50, Emax: int = 20000
):

    for e in range(Emax):

        observation, info = env.reset()
        Q.updateState(observation)

        if round(e % 1e3) == 0:
            print(f"Episode: {int(e / 1e3)}e3 | Rewards gained: {Q.rewardCounter}")  # noqa

        for t in range(Tmax):
            action = Q.chooseAction()

            observation, reward, terminated, truncated, info = env.step(action)

            Q.updateQ(observation, action, reward)

            if reward == 1:
                Q.rewardCounter += 1

            if terminated or truncated:
                break

    env.close()


if __name__ == "__main__":

    ENV = gym.make(
        "FrozenLake-v1", desc=None, map_name="8x8", is_slippery=True
    )
    ENV_SHOW = gym.make(
        "FrozenLake-v1", desc=None, map_name="8x8", is_slippery=True, render_mode='human'  # noqa
    )

    Q = QLearning.Q(
        map_shape=QSHAPE,
        learning_rate=0.9,
        discount=0.87,
        policy="Eps-greedy",  # Eps-greedy | Boltzman
        eps=0.6,
        T=2,
    )

    E = int(5e5)
    T = int(5e1)

    run(env=ENV, Q=Q, Tmax=T, Emax=E)
    print(Q.Q)
    print(f"Rewards count: {Q.rewardCounter}")
    # print(f"\n\n\t\tdifference\n{Q.Q - Qready.Q}\n\t\tdifference\n\n")

    Q.eps = 0.9
    run(env=ENV, Q=Q, Tmax=T, Emax=10)
    print(f"Rewards count: {Q.rewardCounter}")
