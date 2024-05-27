import gym
import QLearning

import numpy as np

import time
from multiprocessing import Process, Queue


# SEED = np.random.seed(318407)


mapsize = 64
actionTypes = 4
QSHAPE = (mapsize, actionTypes)


def run(
    seed: int, queue: Queue
):

    np.random.seed(seed)

    env = gym.make(
        "FrozenLake-v1", desc=None, map_name="8x8", is_slippery=True
    )

    Q = QLearning.Q(
        map_shape=QSHAPE,
        learning_rate=0.9,
        discount=0.87,
        policy="Eps-greedy",  # Eps-greedy | Boltzman
        eps=0.6,
        T=2,
    )

    Emax = int(5e5)
    Tmax = int(5e1)

    for e in range(Emax):

        observation, info = env.reset()
        Q.updateState(observation)

        # if round(e % 1e3) == 0:
        #     print(f"Episode: {int(e / 1e3)}e3 | Rewards gained: {Q.rewardCounter}")  # noqa

        for t in range(Tmax):
            action = Q.chooseAction()

            observation, reward, terminated, truncated, info = env.step(action)

            Q.updateQ(observation, action, reward)

            if reward == 1:
                Q.rewardCounter += 1

            if terminated or truncated:
                break

    result = {
        'seed': seed,
        'Q': Q.Q,
        'rewardCounter': Q.rewardCounter
    }
    queue.put(result)

    env.close()


def main():
    seeds = [318407, 27052024, 19122020, 27112002, 99815612]

    queue = Queue()

    processes = []

    tic = time.time()

    for seed in seeds:
        p = Process(target=run, args=(seed, queue))
        processes.append(p)
        p.start()

    toc = time.time()

    for p in processes:
        p.join()

    print("\n")
    results = []
    while not queue.empty():
        result = queue.get()
        results.append(result)
        print(f"Seed: {result['seed']}")
        print(f"Total Rewards: {result['rewardCounter']}\n")
        print(f"Final Q-Table:\n{result['Q']}")

    print(f"\nElapsed time: {toc - tic}")
    return results


if __name__ == "__main__":
    print("started!")
    results = main()
