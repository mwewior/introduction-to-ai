import gym
import QLearning


mapsize = 64
actionTypes = 4
QSHAPE = (mapsize, actionTypes)


def run(
    env: gym.Env,
    Tmax: int = 1000,
    Q: QLearning.Q = None,
    policy: str = "Eps-greedy",
    eps: int = 0.8,
):
    if Q is None:
        Q = QLearning.Q(
            QSHAPE,
            learning_rate=0.1,
            discount=0.87,
            policy=policy,  # Eps-greedy | Boltzman
            eps=eps,
            T=2,
        )

    # observation, info = env.reset(seed=SEED)
    observation, info = env.reset()

    for _ in range(Tmax):
        action = Q.chooseAction()

        observation, reward, terminated, truncated, info = env.step(action)

        Q.updateQ(observation, action, reward)

        if terminated or truncated:
            observation, info = env.reset()

        if round(_ % 1e5) == 0:
            print(_)

    env.close()
    # print("\nfinall Q Table:")
    # print(Q.Q)
    return Q


if __name__ == "__main__":
    env = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="8x8",
        is_slippery=True,
        # render_mode="human",  # jeżeli ma się wyświetlać
    )
    t = int(1e6)
    # Q0 = QLearning.Q(
    #         QSHAPE,
    #         learning_rate=0.1,
    #         discount=0.87,
    #         policy=policy,  # Eps-greedy | Boltzman
    #         eps=eps,
    #         T=2,
    #    )
    Q1 = run(env=env, Tmax=t, policy="Eps-greedy", eps=0.9)
    Q2 = run(env=env, Tmax=t, Q=Q1, policy="Boltzman")
    Q3 = run(env=env, Tmax=t, Q=Q2, policy="Eps-greedy", eps=0.15)

    print("\n\n\tFirst Table:\n")
    print(Q1.Q)
    print("\n\n\tSecond Table:\n")
    print(Q2.Q)
    print("\n\n\tThird Table:\n")
    print(Q3.Q)
    print(Q3.reshapeQ())
