import gym
import numpy as np
from src import MultiArmedBandit, QLearning
from src import matplotlib, plt


def q2(env, title="Plot"):

    trials = 10
    steps = 10000
    num_bins = 100

    bandit_results = np.zeros((trials, num_bins))
    qlearner_results = np.zeros((trials, num_bins))

    for i in range(trials):
        agent = MultiArmedBandit(epsilon=0.2)
        action_values, rewards = agent.fit(env, steps=steps, num_bins=num_bins)
        bandit_results[i] = np.array(rewards)

        agent = QLearning(epsilon=0.2, alpha=0.5, gamma=0.9)
        action_values, rewards = agent.fit(env, steps=steps, num_bins=num_bins)
        qlearner_results[i] = np.array(rewards)

    plt.title(title)
    plt.xlabel(f"Steps ({steps // num_bins} per point)")
    plt.ylabel("Reward")

    plt.plot(np.mean(bandit_results, axis=0), label='Bandit')
    plt.plot(np.mean(qlearner_results, axis=0), label='QLearner')
    plt.legend()
    plt.savefig(f"free_response/2a_{title}.png")
    plt.clf()


def main():
    slots = gym.make('SlotMachines-v0')
    q2(slots, title="SlotMachines_Comparision")
  
    frozen_lake = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False)
    q2(frozen_lake, title="FrozenLake_Comparison")

    frozen_lake = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=True)
    q2(frozen_lake, title="SlipperyFrozenLake_Comparison")

if __name__ == "__main__":
    main()